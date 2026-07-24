"""Cost profiling (Phase 7) -- the cost axis of the money figure.

Measures, per architecture, across sequence length:
  * peak VRAM for one forward pass (`torch.cuda.max_memory_allocated`)
  * forward latency (ms), and latency per 1k tokens

This is the "cheap" half of the thesis: a hybrid that recovers dense-level accuracy
is only interesting if it does so at meaningfully lower cost, and a dense model that
OOMs at long context is a *result*, not a crash -- an OOM is recorded as a row with
null cost and `oom=True` so the figure can mark exactly where dense dies.

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
            warmup: int = 2, iters: int = 5) -> Dict:
    """One (model, seq_len) cell: peak VRAM + mean forward latency.

    Raises torch.cuda.OutOfMemoryError to the caller, which records it as a result.
    """
    model.eval()
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    for _ in range(warmup):
        model(x)
    _sync(device)
    _reset_peak(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    _sync(device)
    dt_ms = (time.perf_counter() - t0) * 1000.0 / iters

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
        cfg = yaml.safe_load(open(f"configs/sweep_{arch}.yaml"))
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
    args = ap.parse_args()
    seqs = tuple(s for s in SEQ_LENS if s <= args.max_seq)
    run_profile(out_csv=args.out, device=args.device, seq_lens=seqs,
                batch_size=args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
