"""Cost profiling (Phase 7) -- the cost axis of the money figure.

Measures, per architecture, across sequence length:
  * peak VRAM for one forward pass (`torch.cuda.max_memory_allocated`)
  * forward latency (ms), and latency per 1k tokens

This is the "cheap" half of the thesis: a hybrid that recovers dense-level accuracy
is only interesting if it does so at meaningfully lower cost, and a dense model that
OOMs at long context is a *result*, not a crash -- an OOM is recorded as a row with
null cost and `oom=True` so the figure can mark exactly where dense dies.

`run_decode_profile` measures autoregressive DECODE instead: per-token latency
and the exact size of what each model must keep (KV cache vs fixed state) at a
given context length -- the cost axis where linear attention actually differs.

Models are built from the sweep configs (architecture only -- cost does not depend
on trained weights), so this runs without checkpoints.

    python -m iota.profile --out experiments/results/cost_profile.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import time
from typing import Dict, List, Optional

import torch
import yaml

from .data.tokenizer import get_tokenizer
from .util import sweep_config_path
from .models import build_model

ARCHS = ["transformer", "gated_linear", "hybrid"]
SEQ_LENS = (128, 256, 512, 1024, 2048, 4096, 8192)

CSV_FIELDS = [
    "model", "arch", "seq_len", "batch_size", "params_m",
    "peak_vram_mb", "latency_ms", "latency_ms_per_1k_tok", "oom", "device",
]


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def _reset_peak(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device: str) -> Optional[float]:
    if device.startswith("cuda"):
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    return None


@torch.no_grad()
def measure(model, seq_len: int, batch_size: int, vocab_size: int, device: str,
            warmup: int = 3, iters: int = 5) -> Dict:
    """One (model, seq_len) cell: peak VRAM + MEDIAN forward latency.

    BUILD_PLAN §6: >=3 warmup iters, synchronize around every timing, median of
    >=5 runs (the median shrugs off a one-off stall that would skew a mean).
    Raises torch.cuda.OutOfMemoryError to the caller, which records it as a result.
    """
    model.eval()
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    for _ in range(warmup):
        model(x)
    _sync(device)
    _reset_peak(device)

    times = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        model(x)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    dt_ms = sorted(times)[len(times) // 2]

    peak = _peak_mb(device)
    del x
    return {"latency_ms": dt_ms, "peak_vram_mb": peak,
            "latency_ms_per_1k_tok": dt_ms / (batch_size * seq_len / 1000.0)}


def run_profile(
    out_csv: str = "experiments/results/cost_profile.csv",
    device: Optional[str] = None,
    seq_lens=SEQ_LENS,
    batch_size: int = 1,
    archs=ARCHS,
) -> List[Dict]:
    """Profile every architecture across `seq_lens`; write a tidy CSV."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tokenizer()
    rows: List[Dict] = []

    if not device.startswith("cuda"):
        print("[profile] WARNING: no CUDA -- VRAM columns will be empty and latency "
              "is CPU latency, which is not the cost axis the figure wants.", flush=True)

    for arch in archs:
        cfg = yaml.safe_load(open(sweep_config_path(arch)))
        cfg["vocab_size"] = tok.vocab_size
        run_name = cfg.get("train", {}).get("run_name", f"{arch}_sweep")
        model = build_model(cfg).to(device)
        params_m = model.num_params() / 1e6
        oomed = False
        for sl in seq_lens:
            if oomed:
                # Once it dies it stays dead; record the rest as OOM without retrying.
                rows.append(_row(run_name, arch, sl, batch_size, params_m,
                                 None, None, None, True, device))
                print(f"  {arch:13s} seq={sl:<5} skipped (already OOM)", flush=True)
                continue
            _reset_peak(device)
            try:
                m = measure(model, sl, batch_size, tok.vocab_size, device)
            except torch.cuda.OutOfMemoryError:
                oomed = True
                torch.cuda.empty_cache()
                rows.append(_row(run_name, arch, sl, batch_size, params_m,
                                 None, None, None, True, device))
                print(f"  {arch:13s} seq={sl:<5} OOM  <-- recorded as a result", flush=True)
                continue
            rows.append(_row(run_name, arch, sl, batch_size, params_m,
                             m["peak_vram_mb"], m["latency_ms"],
                             m["latency_ms_per_1k_tok"], False, device))
            vram = f"{m['peak_vram_mb']:8.1f}MB" if m["peak_vram_mb"] is not None else "     n/a"
            print(f"  {arch:13s} seq={sl:<5} {vram} {m['latency_ms']:8.2f}ms "
                  f"({m['latency_ms_per_1k_tok']:.2f} ms/1k tok)", flush=True)
        del model
        gc.collect()
        _reset_peak(device)

    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)", flush=True)
    return rows


# ---------------------------------------------------------------------------
# DECODE: per-token generation cost at a given context length. This is where a
# fixed-size state and a KV cache genuinely differ -- the prefill numbers above
# mostly measure kernel quality (fused SDPA vs a PyTorch chunk loop).
# ---------------------------------------------------------------------------
DECODE_LENS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)
DECODE_FIELDS = [
    "model", "arch", "context_len", "batch_size", "params_m",
    "cache_mb", "decode_ms_per_tok", "tokens_per_sec", "peak_vram_mb", "oom", "device",
]


def _cache_mb(cache: list) -> float:
    """Exact bytes held by the decode cache (KV tensors, or S/z state + conv buffers)."""
    total = 0
    for layer in cache:
        for v in layer.values():
            if torch.is_tensor(v):
                total += v.numel() * v.element_size()
    return total / (1024 ** 2)


@torch.no_grad()
def measure_decode(model, context_len: int, batch_size: int, vocab_size: int, device: str,
                   warmup: int = 3, iters: int = 32) -> Dict:
    """Decode `iters` tokens after a synthetic context of `context_len` tokens.

    Each timed step is synchronised individually; latency is the MEDIAN per-token
    time (BUILD_PLAN §6). Raises torch.cuda.OutOfMemoryError to the caller.
    """
    model.eval()
    cache = model.init_decode_cache(batch_size, context_len + warmup + iters, device=device,
                                    prefill_len=context_len)
    cache_mb = _cache_mb(cache)
    tok = torch.randint(0, vocab_size, (batch_size,), device=device)
    for _ in range(warmup):
        tok = model.decode_step(tok, cache).argmax(-1)
    _sync(device)
    _reset_peak(device)
    times = []
    for _ in range(iters):
        _sync(device)
        t0 = time.perf_counter()
        tok = model.decode_step(tok, cache).argmax(-1)
        _sync(device)
        times.append((time.perf_counter() - t0) * 1000.0)
    ms = sorted(times)[len(times) // 2]
    out = {"cache_mb": cache_mb, "decode_ms_per_tok": ms,
           "tokens_per_sec": batch_size * 1000.0 / ms, "peak_vram_mb": _peak_mb(device)}
    del cache
    return out


def run_decode_profile(
    out_csv: str = "experiments/results/decode_profile.csv",
    device: Optional[str] = None,
    context_lens=DECODE_LENS,
    batch_size: int = 1,
    archs=ARCHS,
) -> List[Dict]:
    """Decode cost per architecture across context length; OOM recorded as a result."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = get_tokenizer()
    rows: List[Dict] = []
    print(f"[decode] per-token decode cost, batch {batch_size}, device {device}", flush=True)
    for arch in archs:
        cfg = yaml.safe_load(open(sweep_config_path(arch)))
        cfg["vocab_size"] = tok.vocab_size
        run_name = cfg.get("train", {}).get("run_name", f"{arch}_sweep")
        model = build_model(cfg).to(device)
        params_m = model.num_params() / 1e6
        oomed = False
        for L in context_lens:
            row = {"model": run_name, "arch": arch, "context_len": L, "batch_size": batch_size,
                   "params_m": round(params_m, 3), "oom": False, "device": device}
            if oomed:
                rows.append({**row, "oom": True})
                continue
            _reset_peak(device)
            try:
                m = measure_decode(model, L, batch_size, tok.vocab_size, device)
            except torch.cuda.OutOfMemoryError:
                oomed = True
                torch.cuda.empty_cache()
                rows.append({**row, "oom": True})
                print(f"  {arch:13s} ctx={L:<6} OOM  <-- recorded as a result", flush=True)
                continue
            rows.append({**row, "cache_mb": round(m["cache_mb"], 3),
                         "decode_ms_per_tok": round(m["decode_ms_per_tok"], 3),
                         "tokens_per_sec": round(m["tokens_per_sec"], 1),
                         "peak_vram_mb": None if m["peak_vram_mb"] is None else round(m["peak_vram_mb"], 2)})
            print(f"  {arch:13s} ctx={L:<6} cache {m['cache_mb']:9.3f} MB  "
                  f"{m['decode_ms_per_tok']:7.3f} ms/tok", flush=True)
        del model
        gc.collect()
        _reset_peak(device)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=DECODE_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)", flush=True)
    return rows


def _row(model, arch, seq_len, bs, params_m, vram, lat, lat1k, oom, device) -> Dict:
    return {
        "model": model, "arch": arch, "seq_len": seq_len, "batch_size": bs,
        "params_m": round(params_m, 3),
        "peak_vram_mb": None if vram is None else round(vram, 2),
        "latency_ms": None if lat is None else round(lat, 3),
        "latency_ms_per_1k_tok": None if lat1k is None else round(lat1k, 3),
        "oom": oom, "device": device,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="iota Phase-7 cost profiling")
    ap.add_argument("--out", default="experiments/results/cost_profile.csv")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_seq", type=int, default=8192)
    ap.add_argument("--decode", action="store_true", help="per-token decode cost instead")
    args = ap.parse_args()
    if args.decode:
        run_decode_profile(out_csv=args.out.replace("cost_profile", "decode_profile"),
                           device=args.device, batch_size=args.batch_size)
        return 0
    seqs = tuple(s for s in SEQ_LENS if s <= args.max_seq)
    run_profile(out_csv=args.out, device=args.device, seq_lens=seqs,
                batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
