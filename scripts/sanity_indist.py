"""Sanity check: evaluate each checkpoint on its EXACT training distribution.

The Phase-6 eval passes deliberately probe OUT of distribution (capacity to 128
bindings, length to 8192). That makes a low score ambiguous: a bad checkpoint and a
genuine capacity limit look identical. This script removes the ambiguity by scoring
each model on the distribution it actually trained on, split BY TASK MODE:

    assoc_pq   -- in-distribution associative-recall per-query accuracy
    state_pq   -- in-distribution state-tracking (control) per-query accuracy

A healthy model is high on both. `state_pq` near chance (~0.01) means the control
never learned, and no sweep figure is trustworthy until that is fixed -- the pooled
per-query metric hides this, because assoc emits several queries per example while
state_track emits one.

Ranges are read from configs/sweep_{arch}.yaml rather than hardcoded, so this can
never silently drift out of sync with the curriculum again.

    python -m scripts.sanity_indist [--n 500] [--only transformer]
"""

from __future__ import annotations

import argparse
import random

import torch
import yaml

from iota.data.dataset import EVAL_OFFSET, make_sweep_example
from iota.data.tokenizer import get_tokenizer
from iota.eval import load_checkpoint, teacher_forced_scores

ARCHS = ["transformer", "gated_linear", "hybrid"]
RESULTS_DIR = "experiments/results"


def _rng_range(spec, default):
    """Read an int / list / {min,max} curriculum spec as an inclusive (lo, hi)."""
    if isinstance(spec, dict):
        return int(spec["min"]), int(spec["max"])
    if isinstance(spec, (list, tuple)):
        return min(spec), max(spec)
    if spec is None:
        return default
    return int(spec), int(spec)


def build_set(tok, comp: dict, n: int, salt: int):
    """n examples drawn the way CurriculumSampler draws them at full difficulty."""
    mode = comp["mode"]
    nb_lo, nb_hi = _rng_range(comp.get("n_bindings"), (2, 8))
    sl_lo, sl_hi = _rng_range(comp.get("seq_len"), (64, 256))
    nq_lo, nq_hi = _rng_range(comp.get("n_queries"), (1, 1))
    dens = float(comp.get("distractor_density", 0.0))
    exs = []
    for idx in range(n):
        r = random.Random(EVAL_OFFSET + salt + idx)
        nb = r.randint(nb_lo, nb_hi)
        seq_len = r.randint(sl_lo, sl_hi)
        seed = r.randrange(1 << 30)
        if mode == "assoc_recall":
            nq = max(1, min(r.randint(nq_lo, nq_hi), nb))
            exs.append(make_sweep_example(
                tok, mode, nb, dens, seq_len, seed,
                n_queries=nq, query_pos=comp.get("query_pos", "uniform")))
        else:
            exs.append(make_sweep_example(
                tok, mode, nb, dens, seq_len, seed,
                ops_kinds=comp.get("ops_kinds")))
    return exs


def score(model, exs, tok, device, minibatch):
    sc = teacher_forced_scores(model, exs, tok.pad_id, device, minibatch)
    flat = [q for e in sc for q in e]
    per_q = sum(flat) / max(1, len(flat))
    exact = sum(1 for e in sc if all(e)) / max(1, len(sc))
    return per_q, exact


def main() -> int:
    ap = argparse.ArgumentParser(description="in-distribution checkpoint sanity check")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--only", choices=ARCHS)
    ap.add_argument("--minibatch", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = get_tokenizer()
    archs = [args.only] if args.only else ARCHS

    print(f"{'model':16s} {'assoc_pq':>9} {'assoc_ex':>9} {'state_pq':>9} {'state_ex':>9}")
    print("-" * 58)
    for arch in archs:
        cfg = yaml.safe_load(open(f"configs/sweep_{arch}.yaml"))
        run_name = cfg.get("train", {}).get("run_name", f"{arch}_sweep")
        try:
            model, _ = load_checkpoint(run_name, results_dir=RESULTS_DIR, device=args.device)
        except Exception as e:
            print(f"{arch:16s} -- no usable checkpoint ({type(e).__name__}: {e})")
            continue
        # Each curriculum component IS the training distribution for its mode.
        out = {}
        for salt, comp in enumerate(cfg["curriculum"]):
            exs = build_set(tok, comp, args.n, salt * 1_000_000)
            out[comp["mode"]] = score(model, exs, tok, args.device, args.minibatch)
        a_pq, a_ex = out.get("assoc_recall", (float("nan"),) * 2)
        s_pq, s_ex = out.get("state_track", (float("nan"),) * 2)
        print(f"{arch:16s} {a_pq:9.3f} {a_ex:9.3f} {s_pq:9.3f} {s_ex:9.3f}")
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\nHealthy = both per-query columns high. state_pq ~0.01 means the control\n"
          "never learned and no sweep figure is trustworthy yet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
