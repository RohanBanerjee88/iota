"""Sanity check: evaluate each recovered checkpoint on its EXACT training
distribution, bucketed by task mode.

Motivation: the Phase-6 eval passes (cells_for_pass) run at seq_len=512 with
distractor_density=0 -- i.e. ~2x the max training length (64..256) that every
model saw. That entangles length-extrapolation with capacity and can tank the
attention-based models before capacity even matters. Before trusting any sweep
number we must confirm the checkpoints actually reproduce their ~0.85 training
accuracy IN-distribution. This script does exactly that and separates assoc_recall
(the real task) from state_track (the control) so we can see, per model:

    assoc_pq   -- in-distribution associative-recall per-query accuracy
    state_pq   -- in-distribution state-tracking per-query accuracy

Run on the Kaggle box after the checkpoints are in experiments/results/.
"""

import random

import torch

from iota.data.dataset import EVAL_OFFSET, make_sweep_example, collate_padded
from iota.data.tokenizer import get_tokenizer
from iota.eval import load_checkpoint, teacher_forced_scores

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N = 500          # examples per (model, mode)
tok = get_tokenizer()

# --- exact training ranges (from configs/sweep_*.yaml, identical across arches) ---
NB = (2, 8)          # n_bindings min,max
SEQ = (64, 256)      # seq_len min,max
NQ = (1, 3)          # n_queries min,max (assoc only)
DENS = 0.1
OPS = ["add", "sub"]


def build_set(mode, n):
    """n in-distribution examples for `mode`, drawn like the curriculum does."""
    exs = []
    for idx in range(n):
        r = random.Random(EVAL_OFFSET + idx + (0 if mode == "assoc_recall" else 7_000_000))
        nb = r.randint(*NB)
        seq_len = r.randint(*SEQ)
        seed = r.randrange(1 << 30)
        if mode == "assoc_recall":
            nq = max(1, min(r.randint(*NQ), nb))
            exs.append(make_sweep_example(
                tok, "assoc_recall", nb, DENS, seq_len, seed,
                n_queries=nq, query_pos="uniform"))
        else:
            exs.append(make_sweep_example(
                tok, "state_track", nb, DENS, seq_len, seed, ops_kinds=OPS))
    return exs


def per_q(model, exs):
    sc = teacher_forced_scores(model, exs, tok.pad_id, DEVICE, minibatch=64)
    flat = [q for e in sc for q in e]
    exact = sum(1 for e in sc if all(e)) / max(1, len(sc))
    return sum(flat) / max(1, len(flat)), exact


assoc = build_set("assoc_recall", N)
state = build_set("state_track", N)
print(f"built {N} assoc + {N} state_track in-distribution examples "
      f"(seq {SEQ}, nb {NB}, density {DENS})\n")

RUNS = ["transformer_sweep", "gated_linear_sweep", "hybrid_sweep"]
print(f"{'model':22s} {'assoc_pq':>9} {'assoc_exact':>12} {'state_pq':>9} {'state_exact':>12}")
print("-" * 70)
for run in RUNS:
    model, _ = load_checkpoint(run, device=DEVICE)
    a_pq, a_ex = per_q(model, assoc)
    s_pq, s_ex = per_q(model, state)
    print(f"{run:22s} {a_pq:9.3f} {a_ex:12.3f} {s_pq:9.3f} {s_ex:12.3f}")
    del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
