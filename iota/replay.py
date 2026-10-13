"""Pass 10: gate-replay controls -- does the TIMING of forgetting matter, or only its amount?

Pass 9's `mean` clamp swapped a layer's gates for their arithmetic mean. Gates
multiply, so that also changes how much is forgotten (0.2 * 0.8 = 0.16, but
0.5 * 0.5 = 0.25): it could not separate "what is forgotten when" from "how much".
This pass records each GLA layer's gate trace on the baseline forward and then
re-runs the same prompts with that ONE layer's trace replaced, over the context
span (facts + distractors, before the first question); the query/answer suffix
keeps its recorded gates:

    replay    the recorded trace, unchanged      -> must reproduce the baseline
                                                  exactly (the harness check)
    shuffle   the same values, permuted in time per prompt and head (several
              random permutations) -> same distribution and the same product
              over the span; only the alignment with tokens changes
    geomean   one constant per prompt and head, exp(mean log gamma) over the
              span -> the same product over the span, no timing at all
    arith     the arithmetic mean over the span (pass 9's clamp, per prompt and
              span) -> forgets LESS in total; geomean vs arith shows the cost of
              that mismatch

Reading it: shuffle ~ replay -> timing does not matter; shuffle << replay ->
WHEN the layer forgets matters, beyond how much (geomean should then fall too).
Matching the span product does not match each fact's remaining decay (a
permutation moves gates before/after a write), so a drop says alignment
matters under this intervention -- not a full decomposition of selectivity.
Gates are handled as logits and in the forward's own effective form
(log of the sigmoid floored at 1e-6), so "the same product" is exact.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Optional

import torch

from .clamp import _paired_delta, _short
from .data.dataset import collate_padded
from .data.tokenizer import Tokenizer, get_tokenizer
from .eval import EVAL_OFFSET, _cell_examples
from .gates import _gla_layers
from .models.gated_linear import EPS

CELLS = (8, 32)
SEED_BASE = 10_000_000          # + EVAL_OFFSET: disjoint from every other pass's prompt seeds
N_SHUFFLES = 3
CONDITIONS = ("replay", "shuffle", "geomean", "arith")
FIELDS = ["model", "n_bindings", "layer", "condition", "per_query", "delta", "delta_ci_low", "delta_ci_high",
          "shuffle_min", "shuffle_max", "ctx_log10_product", "n", "seed"]


def effective_log_gamma(logits: torch.Tensor) -> torch.Tensor:
    """log gamma exactly as the chunk-parallel forward uses it."""
    return torch.log(torch.sigmoid(logits.float()).clamp_min(EPS))


def _logit_from_log_gamma(lg: torch.Tensor) -> torch.Tensor:
    lg = lg.clamp(max=-1e-7)                     # gamma < 1 strictly
    return lg - torch.log(-torch.expm1(lg))      # log g - log(1 - g)


def modify(logits: torch.Tensor, spans: List[int], kind: str, gen: Optional[torch.Generator] = None) -> torch.Tensor:
    """logits (B, T, H) of one layer -> the same with positions [0, spans[i]) replaced."""
    out = logits.clone()
    if kind == "replay":
        return out
    for i, L in enumerate(spans):
        seg = logits[i, :L]                                            # (L, H)
        if kind == "shuffle":
            perm = torch.argsort(torch.rand(L, seg.shape[1], generator=gen), dim=0).to(seg.device)
            out[i, :L] = torch.gather(seg, 0, perm)
        elif kind == "geomean":
            lg = effective_log_gamma(seg).mean(0, keepdim=True)        # (1, H)
            out[i, :L] = _logit_from_log_gamma(lg).to(out.dtype).expand(L, -1)
        elif kind == "arith":
            g = torch.sigmoid(seg.float()).mean(0, keepdim=True)
            out[i, :L] = _logit_from_log_gamma(torch.log(g.clamp_min(EPS))).to(out.dtype).expand(L, -1)
        else:
            raise ValueError(kind)
    return out


def _ctx_log10_product(logits: torch.Tensor, spans: List[int]) -> float:
    """Mean over prompts and heads of log10 prod gamma over the context span."""
    vals = [effective_log_gamma(logits[i, :L]).sum(0).mean() / math.log(10) for i, L in enumerate(spans)]
    return float(torch.stack(vals).mean())


@torch.no_grad()
def _score_chunk(model, chunk, pad_id, device, layer_mods=None, mods=None, capture=None):
    """Teacher-forced (hits, total) per example for one batch (same rule as eval.teacher_forced_scores),
    optionally replacing one layer's gate logits (`mods[name]`) or capturing all of them."""
    hooks = []
    for name, m in (layer_mods or {}).items():
        if capture is not None:
            hooks.append(m.g_proj.register_forward_hook(
                lambda mod, inp, o, name=name: capture.__setitem__(name, o.detach().clone())))
        elif mods is not None and name in mods:
            hooks.append(m.g_proj.register_forward_hook(lambda mod, inp, o, t=mods[name]: t))
    try:
        inp = collate_padded(chunk, pad_id).to(device)
        pred = model(inp[:, :-1]).argmax(-1)
    finally:
        for h in hooks:
            h.remove()
    res = []
    for i, e in enumerate(chunk):
        hits = sum(all(int(pred[i, j - 1]) == e.tokens[j] for j in range(s, en)) for (s, en) in e.answer_spans)
        res.append((hits, len(e.answer_spans)))
    return res


def run_replay_pass(models: Dict[str, object], n: int = 500, seed: int = 0, device: str = "cpu",
                    out_csv: Optional[str] = None, tok: Tokenizer = None, minibatch: int = 32,
                    cells=CELLS, n_shuffles: int = N_SHUFFLES) -> List[Dict]:
    tok = tok or get_tokenizer()
    rows: List[Dict] = []
    for ci, nb in enumerate(cells):
        cell = {"mode": "assoc_recall", "seq_len": 256, "n_bindings": nb,
                "n_queries": min(nb, 16), "query_pos": "uniform"}
        examples = _cell_examples(tok, cell, n, EVAL_OFFSET + SEED_BASE + seed * 1_000_000 + ci * 10_000)
        for name, model in models.items():
            model.eval()
            layers = dict(_gla_layers(model))
            if not layers:
                print(f"  pass10 {name:18s} no gated-linear layers -- skipped", flush=True)
                continue
            gen = torch.Generator().manual_seed(seed * 7919 + ci)
            base: List[tuple] = []
            per = {(ln, c): [] for ln in layers for c in CONDITIONS}
            shuf_each = {ln: [[] for _ in range(n_shuffles)] for ln in layers}
            prod = {(ln, c): [] for ln in layers for c in ("baseline",) + CONDITIONS}
            for start in range(0, len(examples), minibatch):
                chunk = examples[start: start + minibatch]
                spans = [e.true_len for e in chunk]
                cap: Dict[str, torch.Tensor] = {}
                base += _score_chunk(model, chunk, tok.pad_id, device, layers, capture=cap)
                for ln in layers:
                    rec = cap[ln]
                    prod[(ln, "baseline")].append(_ctx_log10_product(rec, spans))
                    for c in CONDITIONS:
                        if c == "shuffle":
                            runs = []
                            for k in range(n_shuffles):
                                t = modify(rec, spans, "shuffle", gen)
                                if k == 0:
                                    prod[(ln, c)].append(_ctx_log10_product(t, spans))
                                sc = _score_chunk(model, chunk, tok.pad_id, device, layers, mods={ln: t})
                                shuf_each[ln][k] += sc
                                runs.append(sc)
                            per[(ln, c)] += [(sum(r[j][0] for r in runs) / n_shuffles, runs[0][j][1])
                                             for j in range(len(chunk))]
                        else:
                            t = modify(rec, spans, c)
                            prod[(ln, c)].append(_ctx_log10_product(t, spans))
                            per[(ln, c)] += _score_chunk(model, chunk, tok.pad_id, device, layers, mods={ln: t})

            def pq(xs):
                return sum(h for h, _ in xs) / max(1, sum(t for _, t in xs))

            mean_prod = lambda k: round(sum(prod[k]) / len(prod[k]), 3)
            for ln in layers:
                rows.append({"model": name, "n_bindings": nb, "layer": _short(ln), "condition": "baseline",
                             "per_query": round(pq(base), 4), "delta": 0.0, "delta_ci_low": 0.0, "delta_ci_high": 0.0,
                             "shuffle_min": "", "shuffle_max": "", "ctx_log10_product": mean_prod((ln, "baseline")),
                             "n": n, "seed": seed})
                for c in CONDITIONS:
                    d, lo, hi = _paired_delta(per[(ln, c)], base, seed=seed)
                    sh = [pq(x) for x in shuf_each[ln]] if c == "shuffle" else []
                    rows.append({"model": name, "n_bindings": nb, "layer": _short(ln), "condition": c,
                                 "per_query": round(pq(per[(ln, c)]), 4), "delta": round(d, 4),
                                 "delta_ci_low": round(lo, 4), "delta_ci_high": round(hi, 4),
                                 "shuffle_min": round(min(sh), 4) if sh else "",
                                 "shuffle_max": round(max(sh), 4) if sh else "",
                                 "ctx_log10_product": mean_prod((ln, c)), "n": n, "seed": seed})
                msg = " ".join(f"{c}={pq(per[(ln, c)]):.3f}" for c in CONDITIONS)
                print(f"  pass10 {name:18s} nb={nb:<3} {_short(ln):3s} baseline={pq(base):.3f} {msg}", flush=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)", flush=True)
    return rows
