"""Pass 8: what do the decay gates actually do? (gate statistics, no training)

r07 (the configured gate init: bias 6, zero weight, gamma ~ 0.9975) lost most of
the recall that r04 (the overwritten init: bias 0, random weight, gamma ~ 0.5) had.
The hypothesis: at bias 6 the sigmoid is saturated (gradient sigma'(6) ~ 0.0025,
100x smaller than at 0), so the gate never learns to depend on the input and the
state accumulates everything with almost no forgetting.

This pass measures it directly. For every GLA layer of a trained checkpoint it
records gamma_t = sigmoid(g_proj(x_t)) on pass-1-style recall prompts, split by
token role:

    set_key    "SET", the key digits and "=" of a fact line
    set_value  the value digits of a fact line            (what must be remembered)
    distractor "DISTRACTOR" lines                          (what may be forgotten)
    query      "GET", the key digits and "=" of a question
    answer     the answer digits after a question

and, for each fact value token, how much of it survives to the first question:
retention = prod_{s=t+1..end} gamma_s (reported as mean log10). Selective
forgetting shows up as gamma(distractor) < gamma(set_value) and a wide spread of
gamma across tokens; a frozen gate shows gamma ~ constant with std ~ 0.
"""

from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Optional

import torch

from .data.dataset import collate_padded
from .data.tokenizer import Tokenizer, get_tokenizer
from .eval import EVAL_OFFSET, _cell_examples
from .models.gated_linear import GatedLinearAttention

ROLES = ("set_key", "set_value", "distractor", "query", "answer")
CELLS = (8, 32)  # n_bindings: easy for both inits vs where r07 collapsed (0.88 -> 0.36)
SEED_BASE = 8_000_000  # + EVAL_OFFSET: disjoint from every other pass's prompt seeds
FIELDS = ["model", "n_bindings", "layer", "role", "n_tokens", "gamma_mean", "gamma_std",
          "gamma_p05", "gamma_p50", "gamma_p95", "head_mean_min", "head_mean_max",
          "fact_log10_retention", "bias_mean", "weight_norm", "n", "seed"]


def token_roles(tokens: List[int], tok: Tokenizer) -> List[str]:
    """Role of every token (see module docstring); anything else is 'other'."""
    roles, line, phase = [], None, None
    for t in tokens:
        s = tok.itos[t]
        if s in ("SET", "GET"):
            line, phase = s, "key"
            roles.append("set_key" if s == "SET" else "query")
        elif s == "DISTRACTOR":
            line, phase = "noise", None
            roles.append("distractor")
        elif s == "=" and line in ("SET", "GET"):
            roles.append("set_key" if line == "SET" else "query")
            phase = "value"
        elif s.isdigit() and line in ("SET", "GET"):
            if line == "SET":
                roles.append("set_key" if phase == "key" else "set_value")
            else:
                roles.append("query" if phase == "key" else "answer")
        elif line == "noise" and s not in (tok.itos[tok.pad_id], tok.itos[tok.eos_id]):
            roles.append("distractor")
        else:
            if s in ("START", tok.itos[tok.eos_id]):
                line, phase = None, None
            roles.append("other")
    return roles


def _gla_layers(model) -> List[tuple]:
    """[(layer_name, GatedLinearAttention)] in depth order."""
    return [(name, m) for name, m in model.named_modules() if isinstance(m, GatedLinearAttention)]


def _quantile(xs: torch.Tensor, q: float) -> float:
    if xs.numel() > 1_000_000:  # torch.quantile has an input-size limit; a sample is plenty
        xs = xs[torch.randperm(xs.numel(), generator=torch.Generator().manual_seed(0))[:1_000_000]]
    return float(torch.quantile(xs, q))


@torch.no_grad()
def gate_stats(model, examples, tok: Tokenizer, device: str = "cpu", minibatch: int = 32) -> Dict:
    """-> {layer_name: {"gamma": {role: 1-D tensor of gammas (tokens x heads)},
                        "head_means": {role: (H,) tensor}, "retention": [log10 per fact token],
                        "bias_mean": float, "weight_norm": float}}"""
    model.eval()
    layers = _gla_layers(model)
    out = {name: {"gamma": {r: [] for r in ROLES}, "head_sum": {r: 0.0 for r in ROLES},
                  "head_cnt": {r: 0 for r in ROLES}, "retention": [],
                  "bias_mean": float(m.g_proj.bias.float().mean()),
                  "weight_norm": float(m.g_proj.weight.float().norm(dim=1).mean())}
           for name, m in layers}
    if not layers:
        return {}
    captured: Dict[str, torch.Tensor] = {}
    hooks = [m.g_proj.register_forward_hook(
        lambda mod, inp, o, name=name: captured.__setitem__(name, torch.sigmoid(o.float())))
        for name, m in layers]
    try:
        for start in range(0, len(examples), minibatch):
            chunk = examples[start: start + minibatch]
            model(collate_padded(chunk, tok.pad_id).to(device))
            for name in out:
                g = captured[name].cpu()                     # (B, T, H)
                lg = torch.log(g.clamp_min(1e-12))
                cum = torch.cumsum(lg, dim=1)                # (B, T, H)
                for i, ex in enumerate(chunk):
                    roles = token_roles(ex.tokens, tok)
                    end = ex.true_len - 1                    # last context token, before the 1st question
                    for r in ROLES:
                        idx = [t for t, rr in enumerate(roles) if rr == r]
                        if not idx:
                            continue
                        sel = g[i, idx]                      # (n_r, H)
                        out[name]["gamma"][r].append(sel.reshape(-1))
                        out[name]["head_sum"][r] = out[name]["head_sum"][r] + sel.sum(0)
                        out[name]["head_cnt"][r] += sel.shape[0]
                    for t in (t for t, rr in enumerate(roles) if rr == "set_value" and t < end):
                        # fraction of token t's write left at `end`: prod gamma_{t+1..end}
                        out[name]["retention"].append(
                            float(((cum[i, end] - cum[i, t]) / math.log(10)).mean()))
    finally:
        for h in hooks:
            h.remove()
    for name in out:
        d = out[name]
        d["gamma"] = {r: torch.cat(v) if v else torch.empty(0) for r, v in d["gamma"].items()}
        d["head_means"] = {r: d["head_sum"][r] / d["head_cnt"][r] if d["head_cnt"][r] else None
                           for r in ROLES}
    return out


def run_gate_pass(models: Dict[str, object], n: int = 200, seed: int = 0, device: str = "cpu",
                  out_csv: Optional[str] = None, tok: Tokenizer = None, minibatch: int = 32,
                  cells=CELLS) -> List[Dict]:
    tok = tok or get_tokenizer()
    rows: List[Dict] = []
    for ci, nb in enumerate(cells):
        cell = {"mode": "assoc_recall", "seq_len": 256, "n_bindings": nb,
                "n_queries": min(nb, 16), "query_pos": "uniform"}
        examples = _cell_examples(tok, cell, n, EVAL_OFFSET + SEED_BASE + seed * 1_000_000 + ci * 10_000)
        for name, model in models.items():
            stats = gate_stats(model, examples, tok, device, minibatch)
            if not stats:
                print(f"  pass8 {name:18s} no gated-linear layers -- skipped", flush=True)
                continue
            for layer, d in stats.items():
                for r in ROLES:
                    g = d["gamma"][r]
                    if g.numel() == 0:
                        continue
                    hm = d["head_means"][r]
                    ret = d["retention"] if r == "set_value" else []
                    rows.append({
                        "model": name, "n_bindings": nb, "layer": layer, "role": r, "n_tokens": g.numel(),
                        "gamma_mean": round(float(g.mean()), 5), "gamma_std": round(float(g.std()), 5),
                        "gamma_p05": round(_quantile(g, 0.05), 5), "gamma_p50": round(_quantile(g, 0.5), 5),
                        "gamma_p95": round(_quantile(g, 0.95), 5),
                        "head_mean_min": round(float(hm.min()), 5), "head_mean_max": round(float(hm.max()), 5),
                        "fact_log10_retention": round(sum(ret) / len(ret), 3) if ret else "",
                        "bias_mean": round(d["bias_mean"], 4), "weight_norm": round(d["weight_norm"], 4),
                        "n": n, "seed": seed})
                sv = [x for x in rows if x["model"] == name and x["layer"] == layer
                      and x["n_bindings"] == nb and x["role"] in ("set_value", "distractor")]
                msg = " ".join(f"{x['role']}={x['gamma_mean']:.3f}±{x['gamma_std']:.3f}" for x in sv)
                print(f"  pass8 {name:18s} nb={nb:<3} {layer:22s} {msg} "
                      f"bias={d['bias_mean']:+.2f} |W|={d['weight_norm']:.3f}", flush=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)", flush=True)
    return rows
