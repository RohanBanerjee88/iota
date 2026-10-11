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
import random
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
    """Rebuild a model from a saved run json + weights (safetensors or .pt).

    Robust to a missing run json: if `{run_name}.json` isn't there but the weights
    are, reconstruct the config from `configs/sweep_{arch}.yaml` ($IOTA_CONFIG_DIR) (the arch is the
    run_name minus its trailing `_sweep`/`_milestone` suffix).
    """
    import json
    import os

    import yaml

    from .data.tokenizer import get_tokenizer
    from .models import build_model

    json_path = os.path.join(results_dir, f"{run_name}.json")
    if os.path.exists(json_path):
        with open(json_path) as fh:
            run = json.load(fh)
        cfg = run["config"]
        weights = run["weights"]
    else:
        # Fallback: derive arch from run_name and load the matching sweep config.
        arch = run_name
        for suffix in ("_sweep", "_milestone"):
            if arch.endswith(suffix):
                arch = arch[: -len(suffix)]
                break
        from .util import sweep_config_path
        cfg_path = sweep_config_path(arch)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"no {run_name}.json and no {cfg_path} to reconstruct config from"
            )
        cfg = yaml.safe_load(open(cfg_path))
        cfg["vocab_size"] = get_tokenizer().vocab_size
        weights = f"{run_name}.safetensors"
        print(f"  [load_checkpoint] {run_name}.json missing -> rebuilt config from {cfg_path}")

    model = build_model(cfg).to(device)
    wpath = os.path.join(results_dir, weights)
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


def cluster_bootstrap_ci(
    scores: List[List[bool]], n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> Tuple[float, float]:
    """CI for PER-QUERY accuracy, resampling whole EXAMPLES (the cluster).

    A Wilson interval over flattened queries would assume every query is
    independent, but queries inside one example share a prompt and a model state
    and are strongly correlated -- that understates the interval. Resampling at the
    example level respects the clustering. Used for the headline capacity curve;
    `wilson_ci` still covers the exact-match (one Bernoulli per example) column.
    """
    if not scores:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(scores)
    hits = [sum(1 for q in s if q) for s in scores]
    tots = [len(s) for s in scores]
    if sum(tots) == 0:
        return (0.0, 0.0)
    idx_range = range(n)
    means = []
    for _ in range(n_boot):
        pick = [rng.choice(idx_range) for _ in range(n)]
        h = sum(hits[i] for i in pick)
        t = sum(tots[i] for i in pick)
        means.append(h / t if t else 0.0)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return (max(0.0, lo), min(1.0, hi))


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


@torch.no_grad()
def free_running_scores(
    model, examples: List[SweepExample], pad_id: int, device: str, minibatch: int = 32
) -> List[List[bool]]:
    """Per-query correctness when the model's OWN answers stay in the context.

    Teacher forcing scores query q with the CORRECT earlier answers in the
    context. Here the harness still supplies each query ("GET k ="), but the
    answer digits are generated greedily and kept, so an early mistake can
    affect later queries. Query 0 is identical under both protocols.
    """
    model.eval()
    out: List[List[bool]] = []
    for start in range(0, len(examples), minibatch):
        chunk = examples[start: start + minibatch]
        # per example: the harness text between answers, and the true answers
        prefixes, pieces, truths = [], [], []
        for e in chunk:
            spans = e.answer_spans
            prefixes.append(list(e.tokens[: spans[0][0]]))
            pieces.append([list(e.tokens[spans[i][1]: spans[i + 1][0]]) for i in range(len(spans) - 1)])
            truths.append([e.tokens[s:en] for s, en in spans])
        cur = prefixes
        ok: List[List[bool]] = [[] for _ in chunk]
        n_q = max(len(t) for t in truths)
        for q in range(n_q):
            active = [i for i in range(len(chunk)) if q < len(truths[i])]
            gen = {i: [] for i in active}
            width = len(truths[active[0]][q])
            for _ in range(width):
                L = max(len(cur[i]) for i in active)
                batch = torch.full((len(active), L), pad_id, dtype=torch.long)
                for r, i in enumerate(active):
                    batch[r, : len(cur[i])] = torch.tensor(cur[i])
                logits = model(batch.to(device))
                for r, i in enumerate(active):
                    tok_id = int(logits[r, len(cur[i]) - 1].argmax())
                    gen[i].append(tok_id)
                    cur[i] = cur[i] + [tok_id]
            for i in active:
                ok[i].append(gen[i] == list(truths[i][q]))
                if q + 1 < len(truths[i]):
                    cur[i] = cur[i] + pieces[i][q]
        out.extend(ok)
    return out


FREERUN_FIELDS = ["model", "n_bindings", "n_queries", "per_query_teacher_forced", "per_query_free_running",
                  "exact_teacher_forced", "exact_free_running", "queries_flipped", "n", "seed"]


def run_freerun_pass(
    models: Dict[str, object], n: int = 500, seed: int = 0, device: str = "cpu",
    out_csv: Optional[str] = None, tok: Tokenizer = None, minibatch: int = 32,
) -> List[Dict]:
    """Pass 6: free-running vs teacher-forced scoring on Pass 1's prompts (first n per cell)."""
    tok = tok or get_tokenizer()
    rows: List[Dict] = []
    for ci, cell in enumerate(cells_for_pass(1)):
        examples = _cell_examples(tok, cell, n, EVAL_OFFSET + seed * 1_000_000 + ci * 10_000)
        for name, model in models.items():
            tf = _scores_with_backoff(model, examples, tok.pad_id, device, minibatch, name)
            mb = minibatch
            fr = None
            while fr is None:
                try:
                    fr = free_running_scores(model, examples, tok.pad_id, device, mb)
                except torch.cuda.OutOfMemoryError:
                    if device.startswith("cuda"):
                        torch.cuda.empty_cache()
                    if mb == 1:
                        break
                    mb = max(1, mb // 2)
            if tf is None or fr is None:
                continue
            pq = lambda sc: sum(q for s in sc for q in s) / max(1, sum(len(s) for s in sc))
            ex = lambda sc: sum(1 for s in sc if all(s)) / max(1, len(sc))
            flipped = sum(1 for a, b in zip(tf, fr) for x, y in zip(a, b) if x != y)
            rows.append({"model": name, "n_bindings": cell["n_bindings"], "n_queries": cell["n_queries"],
                         "per_query_teacher_forced": round(pq(tf), 4), "per_query_free_running": round(pq(fr), 4),
                         "exact_teacher_forced": round(ex(tf), 4), "exact_free_running": round(ex(fr), 4),
                         "queries_flipped": flipped, "n": n, "seed": seed})
            print(f"  pass6 {name:18s} nb={cell['n_bindings']:<4} teacher-forced={pq(tf):.3f} "
                  f"free-running={pq(fr):.3f} flipped={flipped}", flush=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FREERUN_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)", flush=True)
    return rows


def _cell_examples(tok, cell: Dict, n: int, seed_base: int) -> List[SweepExample]:
    """Build n PAIRED examples for a cell (identical across models).

    The generator budgets length in whitespace WORDS, but numbers split into one
    token per digit, so more bindings mean more tokens than the nominal seq_len
    (128 bindings at "1024" is ~1270 real tokens). Cells with exact_true_len
    re-generate with a corrected nominal length until the true prompt length is
    within a few tokens of seq_len, so a grid column really holds length fixed.
    """
    def make(seq_len, seed):
        return make_sweep_example(
            tok,
            mode=cell["mode"],
            n_bindings=cell["n_bindings"],
            distractor_density=cell.get("distractor_density", 0.0),
            seq_len=seq_len,
            seed=seed,
            n_queries=cell.get("n_queries", 1),
            query_pos=cell.get("query_pos", "uniform"),
            ops_kinds=cell.get("ops_kinds"),
        )

    exs = []
    for idx in range(n):
        target = cell["seq_len"]
        e = make(target, seed_base + idx)
        if cell.get("exact_true_len"):
            nominal = target
            for _ in range(4):
                gap = e.true_len - target
                if abs(gap) <= 8:
                    break
                nominal = max(1, nominal - gap)
                e = make(nominal, seed_base + idx)
        exs.append(e)
    return exs


def cells_for_pass(pass_id: int) -> List[Dict]:
    """The eval grid for each pass (see PHASE6_EVAL_SPEC.md §3)."""
    if pass_id == 1:  # capacity (headline): multi-query, length held in-distribution
        # seq_len=256 is a length the models train on. The distractor budget's
        # length-floor pads every low/mid-nb example UP to ~256 tokens (bindings
        # buried in distractors, exactly the training regime), so the crossover
        # region (nb up to ~50) varies ONLY capacity, not length. Past ~50 bindings
        # the SET lines themselves exceed 256 tokens and true_len grows to ~773 at
        # nb=128 -- a mild 1.2x RoPE extrapolation for a 640-trained model, and the
        # regime where linear attention is expected dead anyway. The old grid
        # (seq_len=512) forced even nb=2 to ~510 tokens, ~2x the training length,
        # tanking the attention models on LENGTH before capacity mattered --
        # inverting the headline.
        return [
            {"mode": "assoc_recall", "seq_len": 256, "n_bindings": nb,
             "n_queries": min(nb, 16), "query_pos": "uniform", "sweep": nb}
            for nb in (2, 4, 8, 16, 32, 64, 128)
        ]
    if pass_id == 2:  # length-generalization: fixed capacity, swept length
        return [
            {"mode": "assoc_recall", "seq_len": sl, "n_bindings": 8,
             "n_queries": 8, "query_pos": "uniform", "sweep": sl}
            for sl in (128, 256, 512, 1024, 2048, 4096, 8192)
        ]
    if pass_id == 3:  # control: overwrite state_track (hold ONE value), swept length
        return [
            {"mode": "state_track", "seq_len": sl, "n_bindings": 8,
             "n_queries": 1, "ops_kinds": ["set"], "sweep": sl}
            for sl in (128, 256, 512, 1024, 2048, 4096, 8192)
        ]
    if pass_id == 5:  # JOINT load x distance grid: both axes at once, 8 queries every cell
        # Passes 1 and 2 vary load and distance separately; a model can pass both
        # and still fail where they meet. Every cell asks the same number of
        # questions (8), and 1024 tokens already fits the 128-binding definitions,
        # so within a column the context length is held fixed.
        return [
            {"mode": "assoc_recall", "seq_len": sl, "n_bindings": nb, "n_queries": 8,
             "query_pos": "uniform", "exact_true_len": True, "sweep": f"{nb}x{sl}"}
            for nb in (8, 32, 64, 128) for sl in (1024, 2048, 4096, 8192)
        ]
    raise ValueError(f"unknown pass {pass_id}")


KEYLEN_FIELDS = ["model", "n_bindings", "key_digits", "accuracy_per_query", "ci_low", "ci_high",
                 "n_queries", "seed"]


def run_keylen_pass(
    models: Dict[str, object], n: int = 1000, seed: int = 0, device: str = "cpu",
    out_csv: Optional[str] = None, tok: Tokenizer = None, minibatch: int = 32,
    loads=(16, 32, 64),
) -> List[Dict]:
    """Diagnostic (pass 4): recall accuracy split by how many DIGITS the queried key has.

    Keys are 1-3 digit numbers in [0,128). A short causal conv can fuse only a few
    neighbouring tokens, so if a model's errors pile up on 3-digit keys -- and get
    worse as more look-alike keys share the context -- the "capacity" drop is
    partly a key-resolution problem, not a memory limit. Same cells and seeds as
    Pass 1, so the prompts are identical to the headline curve's.
    """
    tok = tok or get_tokenizer()
    pass1 = {c["n_bindings"]: (ci, c) for ci, c in enumerate(cells_for_pass(1))}
    rows: List[Dict] = []
    for nb in loads:
        ci, cell = pass1[nb]
        examples = _cell_examples(tok, cell, n, EVAL_OFFSET + seed * 1_000_000 + ci * 10_000)
        for name, model in models.items():
            scores = _scores_with_backoff(model, examples, tok.pad_id, device, minibatch, name)
            if scores is None:
                continue
            hit: Dict[int, int] = defaultdict(int)
            tot: Dict[int, int] = defaultdict(int)
            for sc, ex in zip(scores, examples):
                for ok, key in zip(sc, ex.meta["query_keys"]):
                    d = len(str(key))
                    hit[d] += int(ok)
                    tot[d] += 1
            for d in sorted(tot):
                lo, hi = wilson_ci(hit[d], tot[d])  # per-query (approximate: queries share prompts)
                rows.append({"model": name, "n_bindings": nb, "key_digits": d,
                             "accuracy_per_query": round(hit[d] / tot[d], 4),
                             "ci_low": round(lo, 4), "ci_high": round(hi, 4),
                             "n_queries": tot[d], "seed": seed})
                print(f"  pass4 {name:18s} nb={nb:<3} key_digits={d} acc={hit[d]/tot[d]:.3f} "
                      f"(n_q={tot[d]})", flush=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=KEYLEN_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)", flush=True)
    return rows


CSV_FIELDS = [
    "model", "mode", "pass", "seq_len_nominal", "seq_len_true_tokens",
    "n_bindings", "n_queries", "distractor_density",
    "accuracy_exact", "accuracy_per_query",
    # ci_low/ci_high are the Wilson interval for accuracy_EXACT; the per-query
    # column has its own cluster-bootstrap interval. Pairing a per-query point with
    # the exact-match band would be a category error, so they are kept separate.
    "ci_low", "ci_high", "ci_low_pq", "ci_high_pq", "n", "seed",
]


def _scores_with_backoff(model, examples, pad_id, device, minibatch, name=""):
    """teacher_forced_scores, halving the eval batch on OOM down to batch 1.

    An OOM at eval batch 32 says nothing about the architecture -- it is a choice
    of eval batch size. Only a model that cannot fit ONE sequence has truly hit
    its memory wall (that is the result Panel C reports). Returns None then.
    """
    mb = minibatch
    while True:
        try:
            return teacher_forced_scores(model, examples, pad_id, device, mb)
        except torch.cuda.OutOfMemoryError:
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
            if mb == 1:
                return None
            mb = max(1, mb // 2)
            print(f"  [OOM] {name}: retrying cell at eval batch {mb}", flush=True)


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
            scores = _scores_with_backoff(model, examples, pad, device, minibatch, name)
            if scores is None:
                print(f"  [OOM] {name} pass{pass_id} cell seq_len={cell['seq_len']} "
                      f"n_bindings={cell['n_bindings']} -> OOM even at batch 1, recorded as OOM")
                rows.append(_row(name, pass_id, cell, true_len, None, None,
                                 (None, None), (None, None), n, seed))
                continue
            exact = sum(1 for s in scores if all(s)) / max(1, len(scores))
            flat = [q for s in scores for q in s]
            per_q = sum(flat) / max(1, len(flat))
            k = sum(1 for s in scores if all(s))
            lo, hi = wilson_ci(k, len(scores))
            pq_lo, pq_hi = cluster_bootstrap_ci(scores, seed=seed)
            rows.append(_row(name, pass_id, cell, true_len, exact, per_q,
                             (lo, hi), (pq_lo, pq_hi), n, seed))
            print(f"  pass{pass_id} {name:13s} sweep={cell['sweep']:<5} "
                  f"true_len={true_len:6.0f} exact={exact:.3f} per_q={per_q:.3f} "
                  f"CI_pq=[{pq_lo:.3f},{pq_hi:.3f}]", flush=True)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {out_csv} ({len(rows)} rows)")
    return rows


def _row(name, pass_id, cell, true_len, exact, per_q, ci, ci_pq, n, seed):
    return {
        "model": name, "mode": cell["mode"], "pass": pass_id,
        "seq_len_nominal": cell["seq_len"], "seq_len_true_tokens": round(true_len, 1),
        "n_bindings": cell["n_bindings"], "n_queries": cell.get("n_queries", 1),
        "distractor_density": cell.get("distractor_density", 0.0),
        "accuracy_exact": None if exact is None else round(exact, 4),
        "accuracy_per_query": None if per_q is None else round(per_q, 4),
        "ci_low": None if ci[0] is None else round(ci[0], 4),
        "ci_high": None if ci[1] is None else round(ci[1], 4),
        "ci_low_pq": None if ci_pq[0] is None else round(ci_pq[0], 4),
        "ci_high_pq": None if ci_pq[1] is None else round(ci_pq[1], 4),
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
