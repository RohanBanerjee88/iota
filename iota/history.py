"""Pass 7: does answering earlier questions change later recall? (matched histories)

Every query and answer token also updates a recurrent model's state, so asking a
question is not a read-only operation. This pass tests whether that matters,
without confounding it with distance.

For each item we fix a fact table F (n_bindings SET lines + distractors, padded to
exactly PREFIX_LEN tokens), a held-out target key q*, and a history region of
exactly T tokens before the same final query "GET q* =". Only the history differs:

    neutral    distractor tokens, exactly T of them (no questions at all)
    oracle     n_prior other questions, each followed by its CORRECT 2-digit answer
    generated  the same questions, each followed by the MODEL'S OWN greedy answer
    reordered  the oracle question/answer blocks in a different order

The target key never appears in the history and no history block introduces or
changes a fact, so the final query asks for exactly the same thing at exactly the
same absolute position in every condition. Contrasts are paired per item:
    neutral - oracle     does a history of (correctly answered) use hurt recall?
    generated - oracle   additional effect of the model's own answers/mistakes
    reordered - oracle   order sensitivity of the same history
Dense attention only APPENDS to its cache when asked a question, the hybrid's
recurrent layers overwrite some state, the linear model overwrites all of it --
so the three architectures form a natural "how much does a read write" ladder.

This is a screening experiment on existing checkpoints. A positive neutral-oracle
gap is a behavioural effect to explain (Stage 2), not a mechanism.
"""

from __future__ import annotations

import csv
import os
import random
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from .data.dataset import ANSWER_WIDTH, EVAL_OFFSET, _encode_value
from .data.dsl import MAX_BINDINGS, MOD, NOISE_TOKENS
from .data.tokenizer import Tokenizer, get_tokenizer
from .eval import wilson_ci

PREFIX_LEN = 512            # facts + distractors; nb=64 SETs fit, total stays <= ~610 (< 640 trained)
CELLS = [(nb, q) for nb in (8, 32, 64) for q in (0, 3, 7, 15) if q < nb]
CONDITIONS = ("neutral", "oracle", "generated", "reordered")
SEED_BASE = EVAL_OFFSET + 7_000_000  # disjoint from every other pass's prompt seeds


def _noise(tok: Tokenizer, r: random.Random, n: int) -> List[int]:
    """Exactly n distractor tokens, as DISTRACTOR lines like the generator emits."""
    out: List[int] = []
    while len(out) < n:
        left = n - len(out)
        k = min(left - 1, r.randint(3, 8))
        out += tok.encode("DISTRACTOR " + " ".join(r.choice(NOISE_TOKENS) for _ in range(k)))
    assert len(out) == n
    return out


def make_item(tok: Tokenizer, n_bindings: int, n_prior: int, seed: int) -> Dict:
    """One matched item: the token sequences for every condition + scoring spans."""
    r = random.Random(seed)
    keys = r.sample(range(MAX_BINDINGS), n_bindings)
    values = [r.randint(0, MOD - 1) for _ in keys]
    sets: List[int] = []
    for k, v in zip(keys, values):
        sets += tok.encode(f"SET {k} = {v}")
    if len(sets) > PREFIX_LEN:
        raise ValueError(f"{n_bindings} bindings need {len(sets)} tokens > PREFIX_LEN {PREFIX_LEN}")
    prefix = sets + _noise(tok, r, PREFIX_LEN - len(sets))

    target = r.randrange(n_bindings)
    prior = r.sample([i for i in range(n_bindings) if i != target], n_prior)
    blocks = [(tok.encode(f"GET {keys[i]} ="), _encode_value(tok, values[i])) for i in prior]
    hist_len = sum(len(q) + len(a) for q, a in blocks)
    final_q = tok.encode(f"GET {keys[target]} =")
    final_a = _encode_value(tok, values[target])

    def assemble(order):
        seq, spans = list(prefix), []
        for j in order:
            q, a = blocks[j]
            seq += q
            spans.append((len(seq), len(seq) + len(a)))
            seq += a
        return seq, spans

    oracle, oracle_spans = assemble(range(n_prior))
    perm = list(range(n_prior))
    if n_prior > 1:
        while perm == list(range(n_prior)):
            r.shuffle(perm)
    reordered, reordered_spans = assemble(perm)
    neutral = prefix + _noise(tok, r, hist_len)
    final_pos = PREFIX_LEN + hist_len  # where "GET q* =" starts, identical everywhere
    for s in (oracle, reordered, neutral):
        assert len(s) == final_pos
    return {
        "keys": keys, "values": values, "target": target, "prior": prior,
        "history_values": [values[i] for i in prior],
        "contexts": {"neutral": neutral + final_q, "oracle": oracle + final_q,
                     "reordered": reordered + final_q},
        "oracle_spans": oracle_spans, "final_pos": final_pos,
        "answer": final_a, "answer_value": values[target],
    }


def _last_logits(model, seqs: List[List[int]], pad_id: int, device: str) -> torch.Tensor:
    """Logits at each sequence's last real position (right-padded batch)."""
    L = max(len(s) for s in seqs)
    batch = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        batch[i, : len(s)] = torch.tensor(s)
    logits = model(batch.to(device))
    idx = torch.tensor([len(s) - 1 for s in seqs], device=logits.device)
    return logits[torch.arange(len(seqs), device=logits.device), idx].float()


@torch.no_grad()
def fill_generated(model, items: List[Dict], pad_id: int, device: str, minibatch: int = 32) -> None:
    """Build items[i]["contexts"]["generated"]: oracle history with the model's own
    greedy answers in place of the true ones (earlier mistakes stay in context)."""
    model.eval()
    for start in range(0, len(items), minibatch):
        chunk = items[start: start + minibatch]
        seqs = [list(it["contexts"]["oracle"]) for it in chunk]
        n_prior = max(len(it["oracle_spans"]) for it in chunk)
        for j in range(n_prior):
            active = [i for i, it in enumerate(chunk) if j < len(it["oracle_spans"])]
            for d in range(ANSWER_WIDTH):
                ctx = [seqs[i][: chunk[i]["oracle_spans"][j][0] + d] for i in active]
                pred = _last_logits(model, ctx, pad_id, device).argmax(-1).tolist()
                for i, p in zip(active, pred):
                    seqs[i][chunk[i]["oracle_spans"][j][0] + d] = p
        for it, s in zip(chunk, seqs):
            it["contexts"]["generated"] = s


@torch.no_grad()
def score_final(model, items: List[Dict], cond: str, tok: Tokenizer, device: str,
                minibatch: int = 32) -> List[Dict]:
    """Greedy 2-digit final answer + teacher-forced log-prob of the true answer."""
    model.eval()
    pad_id = tok.pad_id
    out: List[Dict] = []
    for start in range(0, len(items), minibatch):
        chunk = items[start: start + minibatch]
        ctx = [it["contexts"][cond] for it in chunk]
        true = [it["answer"] for it in chunk]
        lp1 = F.log_softmax(_last_logits(model, ctx, pad_id, device), -1)
        d1 = lp1.argmax(-1).tolist()
        lp2 = F.log_softmax(_last_logits(model, [c + [t[0]] for c, t in zip(ctx, true)], pad_id, device), -1)
        d2 = lp2.argmax(-1).tolist()
        wrong1 = [i for i in range(len(chunk)) if d1[i] != true[i][0]]
        if wrong1:  # greedy second digit after the model's own (wrong) first digit
            alt = _last_logits(model, [ctx[i] + [d1[i]] for i in wrong1], pad_id, device).argmax(-1).tolist()
            for i, a in zip(wrong1, alt):
                d2[i] = a
        for i, it in enumerate(chunk):
            logprob = float(lp1[i, true[i][0]] + lp2[i, true[i][1]])
            pred = [d1[i], d2[i]]
            ok = pred == list(true[i])
            digits = "".join(tok.itos[x] for x in pred)
            pval = int(digits) if digits.isdigit() else None
            out.append({"correct": ok, "logprob": logprob, "pred_value": pval,
                        "copy": (not ok) and pval is not None and pval in it["history_values"]})
    return out


FIELDS = ["model", "n_bindings", "n_prior", "condition", "accuracy", "ci_low", "ci_high",
          "mean_logprob", "error_copy_rate", "delta_vs_oracle", "delta_ci_low", "delta_ci_high",
          "final_pos", "n", "seed"]


def _paired_delta_ci(a: List[bool], b: List[bool], n_boot: int = 2000, seed: int = 0):
    """Mean of (a - b) per item with a bootstrap CI that resamples ITEMS (paired)."""
    diffs = [int(x) - int(y) for x, y in zip(a, b)]
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return sum(diffs) / n, means[int(0.025 * n_boot)], means[min(n_boot - 1, int(0.975 * n_boot))]


def run_history_pass(
    models: Dict[str, object], n: int = 500, seed: int = 0, device: str = "cpu",
    out_csv: Optional[str] = None, tok: Tokenizer = None, minibatch: int = 32, cells=CELLS,
) -> List[Dict]:
    tok = tok or get_tokenizer()
    rows: List[Dict] = []
    for ci, (nb, q) in enumerate(cells):
        base = SEED_BASE + seed * 1_000_000 + ci * 10_000
        for name, model in models.items():
            items = [make_item(tok, nb, q, base + i) for i in range(n)]
            conds = CONDITIONS if q > 0 else ("oracle",)  # q=0: every condition is identical
            if q > 0:
                fill_generated(model, items, tok.pad_id, device, minibatch)
            scored = {c: score_final(model, items, c, tok, device, minibatch) for c in conds}
            ref = [s["correct"] for s in scored["oracle"]]
            for c in conds:
                sc = scored[c]
                k = sum(s["correct"] for s in sc)
                lo, hi = wilson_ci(k, len(sc))
                errs = [s for s in sc if not s["correct"]]
                d, dlo, dhi = _paired_delta_ci([s["correct"] for s in sc], ref, seed=seed)
                rows.append({
                    "model": name, "n_bindings": nb, "n_prior": q,
                    "condition": c if q > 0 else "none", "accuracy": round(k / len(sc), 4),
                    "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                    "mean_logprob": round(sum(s["logprob"] for s in sc) / len(sc), 4),
                    "error_copy_rate": round(sum(s["copy"] for s in errs) / len(errs), 4) if errs else 0.0,
                    "delta_vs_oracle": round(d, 4), "delta_ci_low": round(dlo, 4), "delta_ci_high": round(dhi, 4),
                    "final_pos": items[0]["final_pos"] if q == 0 else round(
                        sum(it["final_pos"] for it in items) / len(items), 1),
                    "n": n, "seed": seed,
                })
            msg = " ".join(f"{c}={sum(s['correct'] for s in scored[c]) / n:.3f}" for c in conds)
            print(f"  pass7 {name:18s} nb={nb:<3} prior={q:<3} {msg}", flush=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)", flush=True)
    return rows
