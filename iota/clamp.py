"""Pass 9: gate-clamp intervention -- which layers' forgetting does recall need?

Pass 8 showed WHAT the gates do: the legacy-init models (r04-r06) learned a
fast-forgetting layer 0 and, in places, layers that drop distractors but keep
facts, while r07's gates sit at ~1 everywhere. That is correlation. This pass
intervenes on a trained model at eval time, one GLA layer at a time, replacing
its gate gamma_t = sigmoid(g_proj(x_t)) with a constant per head:

    keep   gamma = sigmoid(6) ~ 0.9975 (the configured init: "remember everything")
           -> removes BOTH the layer's forgetting and its selectivity
    mean   gamma = that head's own mean gamma on these prompts
           -> keeps how much the layer forgets on average, removes only the
              input-dependence (what it chooses to forget)

plus `keep` on every layer at once. Per-query recall (teacher-forced, pass 1's
prompt layout) is compared with the unclamped model on the SAME prompts; the
delta has a paired bootstrap CI over examples.

Reading it: a big drop under `keep` but not `mean` -> the amount of forgetting
matters, not its selectivity; a drop under `mean` too -> the layer's choice of
what to forget matters. On r07 every `keep` clamp should be ~a no-op (its gates
are already ~0.9975): the built-in control.
"""

from __future__ import annotations

import csv
import math
import os
import random
from typing import Dict, List, Optional

import torch

from .data.dataset import collate_padded
from .data.tokenizer import Tokenizer, get_tokenizer
from .eval import EVAL_OFFSET, _cell_examples, _scores_with_backoff
from .gates import _gla_layers

KEEP_LOGIT = 6.0                 # sigmoid(6) ~ 0.9975, the configured gate init
CELLS = (8, 32, 64)              # easy / where r07 collapsed / high load
SEED_BASE = 9_000_000            # + EVAL_OFFSET: disjoint from every other pass's prompt seeds
FIELDS = ["model", "n_bindings", "condition", "layer", "clamp", "gamma_clamped", "per_query",
          "delta", "delta_ci_low", "delta_ci_high", "n", "seed"]


def _short(name: str) -> str:
    """backbone.blocks.3.mixer -> L3"""
    parts = name.split(".")
    return f"L{parts[parts.index('blocks') + 1]}" if "blocks" in parts else name


@torch.no_grad()
def head_mean_gamma(model, examples, pad_id: int, device: str, minibatch: int = 32) -> Dict[str, torch.Tensor]:
    """Per GLA layer: mean gamma per head over every real (non-pad) token. -> {name: (H,)}"""
    model.eval()
    sums, cnt, captured = {}, 0, {}
    layers = _gla_layers(model)
    hooks = [m.g_proj.register_forward_hook(
        lambda mod, inp, o, name=name: captured.__setitem__(name, torch.sigmoid(o.float())))
        for name, m in layers]
    try:
        for start in range(0, len(examples), minibatch):
            chunk = examples[start: start + minibatch]
            model(collate_padded(chunk, pad_id).to(device))
            mask = torch.zeros(len(chunk), max(len(e.tokens) for e in chunk), 1)
            for i, e in enumerate(chunk):
                mask[i, : len(e.tokens)] = 1
            for name in captured:
                g = captured[name].cpu()                                  # (B, T, H)
                sums[name] = sums.get(name, 0) + (g * mask).sum((0, 1))
            cnt += int(mask.sum())
    finally:
        for h in hooks:
            h.remove()
    return {name: s / cnt for name, s in sums.items()}


class clamp_gates:
    """Context manager: replace the gate logits of the named GLA layers.

    `targets` maps layer name -> (H,) tensor of gamma values in (0,1) to impose."""

    def __init__(self, model, targets: Dict[str, torch.Tensor]):
        self.mods = dict(_gla_layers(model))
        self.targets = targets
        self.hooks = []

    def __enter__(self):
        for name, gamma in self.targets.items():
            logit = torch.logit(gamma.double().clamp(1e-6, 1 - 1e-6)).float()

            def hook(mod, inp, out, logit=logit):
                return logit.to(out.device, out.dtype).expand_as(out).clone()
            self.hooks.append(self.mods[name].g_proj.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.hooks:
            h.remove()
        return False


def _per_example(scores: List[List[bool]]) -> List[tuple]:
    return [(sum(s), len(s)) for s in scores]


def _paired_delta(a: List[tuple], b: List[tuple], n_boot: int = 2000, seed: int = 0):
    """Per-query accuracy of a minus b, CI by resampling EXAMPLES (paired, clustered)."""
    def acc(rows, idx):
        h = sum(rows[i][0] for i in idx)
        t = sum(rows[i][1] for i in idx)
        return h / t if t else 0.0
    n = len(a)
    full = range(n)
    d = acc(a, full) - acc(b, full)
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        boots.append(acc(a, idx) - acc(b, idx))
    boots.sort()
    return d, boots[int(0.025 * n_boot)], boots[min(n_boot - 1, int(0.975 * n_boot))]


def run_clamp_pass(models: Dict[str, object], n: int = 500, seed: int = 0, device: str = "cpu",
                   out_csv: Optional[str] = None, tok: Tokenizer = None, minibatch: int = 32,
                   cells=CELLS) -> List[Dict]:
    tok = tok or get_tokenizer()
    keep = 1 / (1 + math.exp(-KEEP_LOGIT))
    rows: List[Dict] = []
    for ci, nb in enumerate(cells):
        cell = {"mode": "assoc_recall", "seq_len": 256, "n_bindings": nb,
                "n_queries": min(nb, 16), "query_pos": "uniform"}
        examples = _cell_examples(tok, cell, n, EVAL_OFFSET + SEED_BASE + seed * 1_000_000 + ci * 10_000)
        for name, model in models.items():
            layers = [ln for ln, _ in _gla_layers(model)]
            if not layers:
                print(f"  pass9 {name:18s} no gated-linear layers -- skipped", flush=True)
                continue
            H = dict(_gla_layers(model))[layers[0]].n_heads
            means = head_mean_gamma(model, examples, tok.pad_id, device, minibatch)
            conds = [("none", "", "none", {})]
            conds.append(("keep_all", "all", "keep", {ln: torch.full((H,), keep) for ln in layers}))
            for ln in layers:
                conds.append((f"keep_{_short(ln)}", _short(ln), "keep", {ln: torch.full((H,), keep)}))
                conds.append((f"mean_{_short(ln)}", _short(ln), "mean", {ln: means[ln]}))
            base = None
            for cond, layer, kind, targets in conds:
                with clamp_gates(model, targets):
                    sc = _scores_with_backoff(model, examples, tok.pad_id, device, minibatch, name)
                if sc is None:
                    continue
                pe = _per_example(sc)
                if base is None:
                    base = pe
                pq = sum(h for h, _ in pe) / max(1, sum(t for _, t in pe))
                d, lo, hi = _paired_delta(pe, base, seed=seed)
                gam = "" if kind == "none" else " ".join(f"{float(g):.3f}" for g in next(iter(targets.values())))
                rows.append({"model": name, "n_bindings": nb, "condition": cond, "layer": layer, "clamp": kind,
                             "gamma_clamped": gam, "per_query": round(pq, 4), "delta": round(d, 4),
                             "delta_ci_low": round(lo, 4), "delta_ci_high": round(hi, 4), "n": n, "seed": seed})
                print(f"  pass9 {name:18s} nb={nb:<3} {cond:10s} recall={pq:.3f} delta={d:+.3f} "
                      f"[{lo:+.3f}, {hi:+.3f}]", flush=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)", flush=True)
    return rows
