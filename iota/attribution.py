"""Pass 12: whose value does a wrong answer give? (plus the snapshot trajectory used by the seed sweep)

Pass 11 found that recall depends on the queried key's digits, and a zero-parameter collision model fits it: models
behave as if they identify a key only by a code. In-between models fit "digit count + last digit" (r35: predicted
0.54, observed 0.54), and stalled models look like "digit count only" (PROGRESS.md 2026-11-23). If that is how they
fail, a wrong answer should be ANOTHER fact's value, and that fact's key should share the queried key's code. This
pass decodes each query's predicted two-digit value. The first digit is teacher-forced; the second is conditioned on
the model's own first digit, which costs one extra forward pass. For every wrong answer it asks:

    in_prompt     is the predicted value one of the other facts' values?  chance = distinct other values / 96
    relation      if so, does that fact's key share the queried key's
                    len        digit count
                    last       last digit
                    len_last   digit count AND last digit      (the in-between models' code)
                    first      first digit
                    len_first  digit count AND first digit
                  or sit next to it in the definitions (adjacent)?
                  Each share is compared with its baseline: the share of the other facts in that relation.

Ties (several facts with the predicted value) split their weight. One row per (model, cell) with the counts, the
attributed relation weights and their baselines (lift = rel / base).
"""

from __future__ import annotations

import csv
import glob
import os
from typing import Dict, List, Optional

import torch

from .data.dataset import EVAL_OFFSET, collate_padded
from .data.tokenizer import Tokenizer, get_tokenizer
from .eval import _cell_examples

CELLS = ((8, 256), (32, 256), (64, 256))
SEED_BASE = 12_000_000            # + EVAL_OFFSET: disjoint from every other pass's prompt seeds
RELS = {
    "len": lambda a, b: len(a) == len(b),
    "last": lambda a, b: a[-1] == b[-1],
    "len_last": lambda a, b: len(a) == len(b) and a[-1] == b[-1],
    "first": lambda a, b: a[0] == b[0],
    "len_first": lambda a, b: len(a) == len(b) and a[0] == b[0],
}
REL_NAMES = list(RELS) + ["adjacent"]
COUNTS = ["queries", "correct", "wrong", "non_digit", "in_prompt", "in_prompt_chance",
          "q_len1", "c_len1", "q_len2", "c_len2", "q_len3", "c_len3"]
FIELDS = (["model", "n_bindings", "seq_len"] + COUNTS + [f"rel_{r}" for r in REL_NAMES]
          + [f"base_{r}" for r in REL_NAMES] + ["n", "seed"])


def fact_list(e) -> List[tuple]:
    """[(key string, value int)] in definition order (the prompt's SET lines)."""
    return [(s[1], int(s[3])) for s in (line.split() for line in e.prompt.splitlines() if line.startswith("SET "))]


@torch.no_grad()
def _predict_chunk(model, chunk, tok: Tokenizer, device: str) -> List[List[Optional[int]]]:
    digit = {tok.stoi[str(d)]: d for d in range(10)}
    inp = collate_padded(chunk, tok.pad_id).to(device)
    pred1 = model(inp[:, :-1]).argmax(-1)                 # pred1[i, j-1] predicts token j
    inp2 = inp.clone()
    for i, e in enumerate(chunk):
        for (s, _) in e.answer_spans:
            inp2[i, s] = pred1[i, s - 1]                  # the model's own first digit
    pred2 = model(inp2[:, :-1]).argmax(-1)
    out = []
    for i, e in enumerate(chunk):
        vals = []
        for (s, _) in e.answer_spans:
            d1, d2 = int(pred1[i, s - 1]), int(pred2[i, s])
            vals.append(digit[d1] * 10 + digit[d2] if d1 in digit and d2 in digit else None)
        out.append(vals)
    return out


def predict_values(model, examples, tok: Tokenizer, device: str, minibatch: int = 32) -> List[List[Optional[int]]]:
    """Per example, per query: the model's predicted two-digit value (None if a predicted token is not a digit).
    Later queries see earlier queries' predicted first digits in context (the second digits stay the true ones)."""
    model.eval()
    out: List[List[Optional[int]]] = []
    start, mb = 0, minibatch
    while start < len(examples):
        try:
            out += _predict_chunk(model, examples[start: start + mb], tok, device)
            start += mb
        except torch.cuda.OutOfMemoryError:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            if mb == 1:
                raise
            mb = max(1, mb // 2)
    return out


def attribute(examples, preds) -> Dict[str, float]:
    """Counts over every query; relation weights over wrong answers that are another fact's value."""
    c: Dict[str, float] = {k: 0 for k in COUNTS}
    for r in REL_NAMES:
        c[f"rel_{r}"] = 0.0
        c[f"base_{r}"] = 0.0
    for e, pv in zip(examples, preds):
        facts = fact_list(e)
        pos = {k: i for i, (k, _) in enumerate(facts)}
        for (k, v), p in zip(zip(e.meta["query_keys"], e.meta["answers"]), pv):
            c["queries"] += 1
            c[f"q_len{len(k)}"] += 1
            if p == v:
                c["correct"] += 1
                c[f"c_len{len(k)}"] += 1
                continue
            c["wrong"] += 1
            if p is None:
                c["non_digit"] += 1
                continue
            others = [(kk, vv) for kk, vv in facts if kk != k]
            c["in_prompt_chance"] += len({vv for _, vv in others if vv != v}) / 96
            matches = [kk for kk, vv in others if vv == p]
            if not matches:
                continue
            c["in_prompt"] += 1
            for name, rel in RELS.items():
                c[f"rel_{name}"] += sum(rel(k, kk) for kk in matches) / len(matches)
                c[f"base_{name}"] += sum(rel(k, kk) for kk, _ in others) / len(others)
            c["rel_adjacent"] += sum(abs(pos[kk] - pos[k]) == 1 for kk in matches) / len(matches)
            c["base_adjacent"] += sum(abs(pos[kk] - pos[k]) == 1 for kk, _ in others) / len(others)
    return c


def _cell(nb: int, seq_len: int) -> Dict:
    return {"mode": "assoc_recall", "seq_len": seq_len, "n_bindings": nb, "n_queries": min(nb, 16),
            "query_pos": "uniform"}


def summarize(c: Dict[str, float]) -> str:
    wrong = max(1, c["wrong"] - c["non_digit"])
    lift = lambda r: c[f"rel_{r}"] / c[f"base_{r}"] if c[f"base_{r}"] else float("nan")
    return (f"recall={c['correct'] / max(1, c['queries']):.3f} wrong_in_prompt={c['in_prompt'] / wrong:.2f} "
            f"(chance {c['in_prompt_chance'] / wrong:.2f}) lift len={lift('len'):.2f} last={lift('last'):.2f} "
            f"len_last={lift('len_last'):.2f} first={lift('first'):.2f} adjacent={lift('adjacent'):.2f}")


def run_attribution_pass(models: Dict[str, object], n: int = 500, seed: int = 0, device: str = "cpu",
                         out_csv: Optional[str] = None, tok: Tokenizer = None, minibatch: int = 32,
                         cells=CELLS) -> List[Dict]:
    tok = tok or get_tokenizer()
    rows: List[Dict] = []
    for ci, (nb, seq_len) in enumerate(cells):
        examples = _cell_examples(tok, _cell(nb, seq_len), n,
                                  EVAL_OFFSET + SEED_BASE + seed * 1_000_000 + ci * 10_000)
        for name, model in models.items():
            c = attribute(examples, predict_values(model, examples, tok, device, minibatch))
            rows.append({"model": name, "n_bindings": nb, "seq_len": seq_len, **c, "n": n, "seed": seed})
            print(f"  pass12 {name:18s} nb={nb:<3} {summarize(c)}", flush=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)", flush=True)
    return rows


# ---------------------------------------------------------------------------
# trajectory: the same measurement on snapshots saved during training (scripts/seed_sweep.py --snapshot-every)
# ---------------------------------------------------------------------------
TRAJ_FIELDS = ["step"] + FIELDS[1:]


def trajectory(results_dir: str, cfg: Dict, run_name: str, device: str, n: int = 200, facts: int = 32,
               seed: int = 0, out_csv: Optional[str] = None) -> List[Dict]:
    """Pass 12 on every `snapshots/{run_name}_{step}.safetensors`, one row per step (same prompts throughout)."""
    from safetensors.torch import load_file

    from .models import build_model

    tok = get_tokenizer()
    examples = _cell_examples(tok, _cell(facts, 256), n, EVAL_OFFSET + SEED_BASE + 900_000_000 + seed * 1_000_000)
    model = build_model(dict(cfg, vocab_size=tok.vocab_size)).to(device)
    rows = []
    for path in sorted(glob.glob(os.path.join(results_dir, "snapshots", f"{run_name}_*.safetensors"))):
        step = int(os.path.basename(path)[len(run_name) + 1: -len(".safetensors")])
        model.load_state_dict(load_file(path))
        model.backbone.tie_weights()
        c = attribute(examples, predict_values(model, examples, tok, device))
        rows.append({"step": step, "n_bindings": facts, "seq_len": 256, **c, "n": n, "seed": seed})
        print(f"  traj {run_name} step {step:5d} {summarize(c)}", flush=True)
    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=TRAJ_FIELDS)
            w.writeheader()
            w.writerows(rows)
    return rows
