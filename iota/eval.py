"""Eval (Phase 5 milestone slice).

Exact-answer accuracy, verifier-checked (never loss). For Phase 5 this measures
in-distribution accuracy to validate the pipeline. The full length/recall grid
and CSV output is Phase 6 (see BUILD_PLAN.md §6 and the carry-forward TODOs).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch

from .data.dataset import (
    EVAL_OFFSET,
    DataSampler,
    Example,
    SweepExample,
    collate_padded,
    make_sweep_example,
)
from .data.tokenizer import Tokenizer, get_tokenizer
from .data.verifier import check


@torch.no_grad()
def _greedy_generate(model, prompt_id_batch: torch.Tensor, eos_id: int, max_new: int, device) -> List[List[int]]:
    """Greedy-decode `max_new` tokens for a batch of equal-length prompts."""
    model.eval()
    seq = prompt_id_batch.to(device)
    B = seq.shape[0]
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    generated = [[] for _ in range(B)]
    for _ in range(max_new):
        logits = model(seq)[:, -1, :]
        nxt = logits.argmax(dim=-1)
        for b in range(B):
            if not finished[b]:
                tokid = int(nxt[b])
                if tokid == eos_id:
                    finished[b] = True
                else:
                    generated[b].append(tokid)
        if bool(finished.all()):
            break
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
    return generated


@torch.no_grad()
def exact_answer_accuracy(
    model,
    sampler: DataSampler,
    n: int = 256,
    offset: int = EVAL_OFFSET,
    max_new: int = 4,
    device: str = "cpu",
    tok: Tokenizer = None,
    capture_failures: int = 5,
) -> Tuple[float, List[Dict]]:
    """Generate answers for `n` held-out examples; score with the verifier.

    Batches examples that share a prompt length so no padding is needed (correct
    for both attention and recurrent architectures). Returns (accuracy, failures).
    """
    tok = tok or get_tokenizer()
    examples: List[Example] = [sampler.example(i, offset) for i in range(n)]

    groups: Dict[int, List[int]] = defaultdict(list)
    for idx, ex in enumerate(examples):
        groups[ex.true_len].append(idx)

    correct = 0
    failures: List[Dict] = []
    for length, idxs in groups.items():
        batch = torch.stack([torch.tensor(examples[i].tokens[:length], dtype=torch.long) for i in idxs])
        gens = _greedy_generate(model, batch, tok.eos_id, max_new, device)
        for local, gidx in enumerate(idxs):
            ex = examples[gidx]
            pred = tok.decode(gens[local])
            ok = check(ex.prompt, pred)
            correct += int(ok)
            if not ok and len(failures) < capture_failures:
                failures.append(
                    {"target": ex.target, "pred": pred, "true_len": ex.true_len, "n_bindings": ex.meta.get("n_bindings")}
                )
    return correct / max(1, len(examples)), failures


def load_checkpoint(run_name: str, results_dir: str = "experiments/results", device: str = "cpu"):
    """Rebuild a model from a saved run json + weights (safetensors or .pt)."""
    import json
    import os

    import yaml  # noqa: F401

    from .models import build_model

    with open(os.path.join(results_dir, f"{run_name}.json")) as fh:
        run = json.load(fh)
    cfg = run["config"]
    model = build_model(cfg).to(device)
    wpath = os.path.join(results_dir, run["weights"])
    if wpath.endswith(".safetensors"):
        from safetensors.torch import load_file

        state = load_file(wpath)
    else:
        state = torch.load(wpath, map_location=device)
    model.load_state_dict(state)
    # Re-tie explicitly: the checkpoint stores embed.weight and head.weight as two
    # separate (cloned) tensors; after loading we collapse them back to one shared
    # parameter so the tie can't silently drift.
    model.backbone.tie_weights()
    model.eval()
    return model, cfg


# ===========================================================================
# Phase 6 sweep: teacher-forced exact-match eval with CIs and CSV output.
#
# Teacher-forced exact-match is IDENTICAL to greedy generation for these tasks
# (the answer deterministically follows the query; if a digit is wrong both mark
# the query wrong; if earlier digits are right the conditioning is identical), and
# it scores a whole batch in ONE forward pass -- sidestepping the slow per-token
# scan that makes greedy decode expensive for the linear model.
# ===========================================================================
def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


@torch.no_grad()
def teacher_forced_scores(
    model, examples: List[SweepExample], pad_id: int, device: str, minibatch: int = 32
) -> List[List[bool]]:
    """Per-query correctness (list[bool] per example) via teacher forcing.

    A query's answer is correct iff the model's argmax at each of its answer-token
    positions matches the true token. Batched, right-padded, one forward per
    minibatch. Raises torch.cuda.OutOfMemoryError up to the caller (Pass 2 dense).
    """
    model.eval()
    out: List[List[bool]] = []
    for start in range(0, len(examples), minibatch):
        chunk = examples[start : start + minibatch]
        inp = collate_padded(chunk, pad_id).to(device)
        logits = model(inp[:, :-1])               # predict token j from position j-1
        pred = logits.argmax(-1)                   # (B, T-1)
        for i, e in enumerate(chunk):
            q_ok = []
            for (s, en) in e.answer_spans:
                ok = all(int(pred[i, j - 1]) == e.tokens[j] for j in range(s, en))
                q_ok.append(bool(ok))
            out.append(q_ok)
        del logits, pred, inp
    return out


def _cell_examples(tok, cell: Dict, n: int, seed_base: int) -> List[SweepExample]:
    """Build n PAIRED examples for a cell (identical across models)."""
    exs = []
    for idx in range(n):
        exs.append(make_sweep_example(
            tok,
            mode=cell["mode"],
            n_bindings=cell["n_bindings"],
            distractor_density=cell.get("distractor_density", 0.0),
            seq_len=cell["seq_len"],
            seed=seed_base + idx,
            n_queries=cell.get("n_queries", 1),
            query_pos=cell.get("query_pos", "uniform"),
            ops_kinds=cell.get("ops_kinds"),
        ))
    return exs


def cells_for_pass(pass_id: int) -> List[Dict]:
    """The eval grid for each pass (see PHASE6_EVAL_SPEC.md §3)."""
    if pass_id == 1:  # capacity (headline): multi-query, fixed length
        return [
            {"mode": "assoc_recall", "seq_len": 512, "n_bindings": nb,
             "n_queries": min(nb, 16), "query_pos": "uniform", "sweep": nb}
            for nb in (2, 4, 8, 16, 32, 64, 128)
        ]
    if pass_id == 2:  # length-generalization: fixed capacity, swept length
        return [
            {"mode": "assoc_recall", "seq_len": sl, "n_bindings": 8,
             "n_queries": 8, "query_pos": "uniform", "sweep": sl}
            for sl in (128, 256, 512, 1024, 2048, 4096, 8192)
        ]
    if pass_id == 3:  # control: additive state_track (learnable home turf), swept length
        return [
            {"mode": "state_track", "seq_len": sl, "n_bindings": 8,
             "n_queries": 1, "ops_kinds": ["add", "sub"], "sweep": sl}
            for sl in (128, 256, 512, 1024, 2048, 4096, 8192)
        ]
    raise ValueError(f"unknown pass {pass_id}")


CSV_FIELDS = [
    "model", "mode", "pass", "seq_len_nominal", "seq_len_true_tokens",
    "n_bindings", "n_queries", "distractor_density",
    "accuracy_exact", "accuracy_per_query", "ci_low", "ci_high", "n", "seed",
]


def run_pass(
    pass_id: int,
    models: Dict[str, object],
    n: int = 500,
    seed: int = 0,
    device: str = "cpu",
    out_csv: Optional[str] = None,
    tok: Tokenizer = None,
    minibatch: int = 32,
    capture_failures: int = 5,
) -> List[Dict]:
    """Run one eval pass over all models (paired prompts) and write a tidy CSV."""
    tok = tok or get_tokenizer()
    pad = tok.pad_id
    rows: List[Dict] = []
    for ci, cell in enumerate(cells_for_pass(pass_id)):
        seed_base = EVAL_OFFSET + seed * 1_000_000 + ci * 10_000
        examples = _cell_examples(tok, cell, n, seed_base)
        true_len = sum(e.true_len for e in examples) / max(1, len(examples))
        for name, model in models.items():
            try:
                scores = teacher_forced_scores(model, examples, pad, device, minibatch)
            except torch.cuda.OutOfMemoryError:
                if device.startswith("cuda"):
                    torch.cuda.empty_cache()
                print(f"  [OOM] {name} pass{pass_id} cell seq_len={cell['seq_len']} "
                      f"n_bindings={cell['n_bindings']} -> recorded as OOM")
                rows.append(_row(name, pass_id, cell, true_len, None, None, (None, None), n, seed))
                continue
            exact = sum(1 for s in scores if all(s)) / max(1, len(scores))
            flat = [q for s in scores for q in s]
            per_q = sum(flat) / max(1, len(flat))
            k = sum(1 for s in scores if all(s))
            lo, hi = wilson_ci(k, len(scores))
            rows.append(_row(name, pass_id, cell, true_len, exact, per_q, (lo, hi), n, seed))
            print(f"  pass{pass_id} {name:13s} sweep={cell['sweep']:<5} "
                  f"true_len={true_len:6.0f} exact={exact:.3f} per_q={per_q:.3f} "
                  f"CI=[{lo:.3f},{hi:.3f}]")
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)")
    return rows


def _row(name, pass_id, cell, true_len, exact, per_q, ci, n, seed):
    return {
        "model": name, "mode": cell["mode"], "pass": pass_id,
        "seq_len_nominal": cell["seq_len"], "seq_len_true_tokens": round(true_len, 1),
        "n_bindings": cell["n_bindings"], "n_queries": cell.get("n_queries", 1),
        "distractor_density": cell.get("distractor_density", 0.0),
        "accuracy_exact": None if exact is None else round(exact, 4),
        "accuracy_per_query": None if per_q is None else round(per_q, 4),
        "ci_low": None if ci[0] is None else round(ci[0], 4),
        "ci_high": None if ci[1] is None else round(ci[1], 4),
        "n": n, "seed": seed,
    }


def main():
    ap = argparse.ArgumentParser(description="iota Phase 6 eval passes")
    ap.add_argument("--pass", dest="pass_id", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--models", default="transformer,gated_linear,hybrid",
                    help="comma-separated run names in experiments/results/")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--minibatch", type=int, default=32)
    ap.add_argument("--out", default=None)
    ap.add_argument("--results_dir", default="experiments/results")
    args = ap.parse_args()

    models = {}
    for name in args.models.split(","):
        name = name.strip()
        model, _ = load_checkpoint(name, results_dir=args.results_dir, device=args.device)
        models[name] = model
    out = args.out or f"{args.results_dir}/pass{args.pass_id}.csv"
    run_pass(args.pass_id, models, n=args.n, seed=args.seed, device=args.device,
             out_csv=out, minibatch=args.minibatch)


if __name__ == "__main__":
    main()
