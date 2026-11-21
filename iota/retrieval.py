"""Pass 11: which facts can a trained model retrieve?

Every trained GLA behaves as if it retrieves a roughly fixed number of facts k: per-query recall is about k/N once
N > k (PROGRESS.md 2026-11-21). Stalled runs sit at k ~ 4-8 at every load, most learned runs at k ~ 25-45, the best
at 56-104. This pass asks WHICH facts are retrievable. Prompts follow pass 1's layout: all SET lines first, in
random key order, then a filler block, then interleaved `GET k = dd` queries. Each query is binned by

    facts_after   how many facts were written after the queried one (0 = the last SET line)
                  -> a recency buffer shows as high recall at small facts_after only
    key           the queried key (0-127)      -> a fixed set of retrievable keys shows as bimodal per-key recall
    value         the answer (0-96)            -> a value prior shows as recall concentrated on a few values
    query_rank    the query's place in the suffix (earlier answers sit in context, teacher-forced)

Cells: 8 / 32 / 64 / 128 facts at the pass-1 floor (256 nominal tokens), plus 8 facts at 640 nominal tokens (the
longest training length). The two 8-fact cells differ only in the filler between the last fact and the queries, so
they separate "number of later facts" from "token distance" without leaving the trained range.
Output: one row per (model, cell, dimension, bin) with hits and total; analysis aggregates further.
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

from .data.dataset import EVAL_OFFSET
from .data.tokenizer import Tokenizer, get_tokenizer
from .eval import _cell_examples, _scores_with_backoff

CELLS = ((8, 256), (8, 640), (32, 256), (64, 256), (128, 256))
SEED_BASE = 11_000_000            # + EVAL_OFFSET: disjoint from every other pass's prompt seeds
DIMS = ("facts_after", "key", "value", "query_rank")
FIELDS = ["model", "n_bindings", "seq_len", "true_len_mean", "dim", "bin", "hits", "total", "n", "seed"]


def query_meta(e) -> List[Dict]:
    """Per query of an assoc_recall example, in answer order: facts written after it, key, value, rank."""
    sets = [line.split() for line in e.prompt.splitlines() if line.startswith("SET ")]
    index = {s[1]: i for i, s in enumerate(sets)}
    n = len(sets)
    return [{"facts_after": n - 1 - index[k], "key": int(k), "value": int(v), "query_rank": r}
            for r, (k, v) in enumerate(zip(e.meta["query_keys"], e.meta["answers"]))]


def tally(examples, scores) -> Dict[tuple, List[int]]:
    """{(dim, bin): [hits, total]} over every query of every example."""
    agg: Dict[tuple, List[int]] = {}
    for e, sc in zip(examples, scores):
        for q, ok in zip(query_meta(e), sc):
            for d in DIMS:
                h = agg.setdefault((d, q[d]), [0, 0])
                h[0] += int(ok)
                h[1] += 1
    return agg


def _recall(agg, dim, lo, hi=None) -> Optional[float]:
    h = sum(v[0] for (d, b), v in agg.items() if d == dim and b >= lo and (hi is None or b < hi))
    t = sum(v[1] for (d, b), v in agg.items() if d == dim and b >= lo and (hi is None or b < hi))
    return h / t if t else None


def run_retrieval_pass(models: Dict[str, object], n: int = 500, seed: int = 0, device: str = "cpu",
                       out_csv: Optional[str] = None, tok: Tokenizer = None, minibatch: int = 32,
                       cells=CELLS) -> List[Dict]:
    tok = tok or get_tokenizer()
    rows: List[Dict] = []
    for ci, (nb, seq_len) in enumerate(cells):
        cell = {"mode": "assoc_recall", "seq_len": seq_len, "n_bindings": nb,
                "n_queries": min(nb, 16), "query_pos": "uniform"}
        examples = _cell_examples(tok, cell, n, EVAL_OFFSET + SEED_BASE + seed * 1_000_000 + ci * 10_000)
        true_len = round(sum(e.true_len for e in examples) / len(examples), 1)
        for name, model in models.items():
            scores = _scores_with_backoff(model, examples, tok.pad_id, device, minibatch, name)
            if scores is None:
                continue
            agg = tally(examples, scores)
            for (d, b), (h, t) in sorted(agg.items(), key=lambda kv: (DIMS.index(kv[0][0]), kv[0][1])):
                rows.append({"model": name, "n_bindings": nb, "seq_len": seq_len, "true_len_mean": true_len,
                             "dim": d, "bin": b, "hits": h, "total": t, "n": n, "seed": seed})
            pq = _recall(agg, "facts_after", 0)
            last4, older = _recall(agg, "facts_after", 0, 4), _recall(agg, "facts_after", 4)
            print(f"  pass11 {name:18s} nb={nb:<3} len={true_len:<6} recall={pq:.3f} k={pq * nb:5.1f} "
                  f"last4={last4:.3f} older={older if older is None else round(older, 3)}", flush=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)", flush=True)
    return rows
