"""iota task generator (Phase 1) — the foundation, built first.

Two task modes (see BUILD_PLAN.md §3). The discriminating axis is *recall load*
(`n_bindings`), not sequence length alone.

  Mode A — `state_track`  (control / sanity)
      A single accumulator `x` updated step by step with affine ops mod M.
      Tests whether a model can carry state at all. Linear models are expected
      to do well here — that is the point of having it as a control.
      For this mode, `n_bindings` is interpreted as the number of update ops.

  Mode B — `assoc_recall` (the real test — MQAR-style)
      Define many variables early, bury them under distractors, then retrieve a
      specific one (optionally with one arithmetic op over two of them). A
      fixed-size recurrent state must hold all bindings simultaneously, so
      accuracy is expected to degrade as `n_bindings` grows for linear attention
      while a transformer holds.

Public API:
    gen(mode, n_bindings, distractor_density, seq_len, seed) -> (prompt, target, meta)

`gen` is deterministic per (args, seed). Length is measured in whitespace tokens
(the tokenizer in Phase 3 is word/keyword level, so this is a faithful proxy;
numbers are the only multi-token words once digit-split). `seq_len` acts as a
length target (a floor); `distractor_density` acts as a minimum noise ratio.
"""

from __future__ import annotations

import argparse
import random
from typing import Dict, List, Tuple

# ----------------------------------------------------------------------------
# Constants / vocabulary building blocks (shared with the Phase-3 tokenizer)
# ----------------------------------------------------------------------------
MOD = 97                      # modular arithmetic base; values live in [0, MOD)
MODES = ("state_track", "assoc_recall")

ACC = "x"                     # the single accumulator name used by Mode A
MAX_BINDINGS = 64             # Mode B variable pool size
VAR_NAMES = [f"v{i}" for i in range(MAX_BINDINGS)]
NOISE_TOKENS = ["qx", "lk", "mn", "zz", "rr", "pp", "tt", "ww", "gg", "hh"]
KEYWORDS = ["START", "SET", "GET", "ANSWER", "DISTRACTOR", "mod"]
OPERATORS = ["=", "(", ")", "+", "-", "*"]


# ----------------------------------------------------------------------------
# Determinism helper
# ----------------------------------------------------------------------------
def _mk_rng(mode: str, n_bindings: int, density: float, seq_len: int, seed: int) -> random.Random:
    """Build a local RNG seeded deterministically from the call arguments.

    Avoids touching global RNG state and avoids hashing strings (whose hash is
    salted per-process), so output is reproducible across processes.
    """
    mode_id = MODES.index(mode)
    s = (
        int(seed) * 1_000_003
        + mode_id * 100_003
        + int(n_bindings) * 10_007
        + int(round(float(density) * 1000)) * 101
        + int(seq_len or 0)
    ) & 0x7FFFFFFF
    return random.Random(s)


def _distractor_budget(struct_len: int, density: float, seq_len: int) -> int:
    """Number of distractor tokens to add.

    `seq_len` sets a length floor; `density` sets a noise-ratio floor. We take
    the max so the length sweep (low density, large seq_len) and the density
    sweep (fixed seq_len, high density) both behave sensibly.
    """
    from_len = max(0, int(seq_len or 0) - struct_len)
    density = float(density)
    if density > 0:
        density = min(density, 0.95)
        from_den = int(round(density * struct_len / (1.0 - density)))
    else:
        from_den = 0
    return max(from_len, from_den)


def _emit_distractor_lines(rng: random.Random, budget: int) -> List[List[str]]:
    """Emit DISTRACTOR lines consuming exactly `budget` whitespace tokens.

    Each line is `DISTRACTOR w1 w2 ...` where the keyword counts as one token.
    """
    lines: List[List[str]] = []
    remaining = budget
    while remaining > 0:
        if remaining == 1:
            lines.append(["DISTRACTOR"])  # bare keyword, 1 token
            remaining -= 1
            break
        # line length = 1 (keyword) + k noise words; cap k by what's left
        k = min(remaining - 1, rng.randint(3, 8))
        words = [rng.choice(NOISE_TOKENS) for _ in range(k)]
        lines.append(["DISTRACTOR"] + words)
        remaining -= 1 + k
    return lines


# ----------------------------------------------------------------------------
# Mode A — state_track
# ----------------------------------------------------------------------------
def _gen_state_track(
    rng: random.Random, n_ops: int, density: float, seq_len: int
) -> Tuple[List[List[str]], str, Dict]:
    n_ops = max(1, int(n_ops))
    state = rng.randint(0, MOD - 1)
    val = state
    start_line = ["START", ACC, "=", str(state)]

    ops: List[List[str]] = []
    for _ in range(n_ops):
        kind = rng.choice(["add", "sub", "mul", "affine"])
        if kind == "add":
            b = rng.randint(0, MOD - 1)
            ops.append([ACC, "=", "(", ACC, "+", str(b), ")", "mod", str(MOD)])
            val = (val + b) % MOD
        elif kind == "sub":
            b = rng.randint(0, MOD - 1)
            ops.append([ACC, "=", "(", ACC, "-", str(b), ")", "mod", str(MOD)])
            val = (val - b) % MOD
        elif kind == "mul":
            a = rng.randint(1, MOD - 1)
            ops.append([ACC, "=", "(", ACC, "*", str(a), ")", "mod", str(MOD)])
            val = (val * a) % MOD
        else:  # affine
            a = rng.randint(1, MOD - 1)
            b = rng.randint(0, MOD - 1)
            ops.append([ACC, "=", "(", ACC, "*", str(a), "+", str(b), ")", "mod", str(MOD)])
            val = (val * a + b) % MOD

    answer_line = ["ANSWER", ACC]
    target = str(val)

    struct_len = len(start_line) + sum(len(o) for o in ops) + len(answer_line)
    budget = _distractor_budget(struct_len, density, seq_len)
    dlines = _emit_distractor_lines(rng, budget)

    # Interleave distractor lines among the op gaps so state must be carried
    # across distance (gap i precedes op i; the last gap precedes ANSWER).
    n_gaps = len(ops)
    buckets: List[List[List[str]]] = [[] for _ in range(n_gaps + 1)]
    for dl in dlines:
        buckets[rng.randint(0, n_gaps)].append(dl)

    out: List[List[str]] = [start_line]
    for i, op in enumerate(ops):
        out.extend(buckets[i])
        out.append(op)
    out.extend(buckets[n_gaps])
    out.append(answer_line)

    meta = {"query": {"type": "state_track", "var": ACC}, "n_ops": n_ops}
    return out, target, meta


# ----------------------------------------------------------------------------
# Mode B — assoc_recall
# ----------------------------------------------------------------------------
def _gen_assoc_recall(
    rng: random.Random, n_bindings: int, density: float, seq_len: int
) -> Tuple[List[List[str]], str, Dict]:
    n = int(n_bindings)
    if n < 2:
        n = 2
    if n > MAX_BINDINGS:
        raise ValueError(
            f"assoc_recall supports at most {MAX_BINDINGS} bindings (got {n_bindings})"
        )

    names = VAR_NAMES[:n]
    values = [rng.randint(0, MOD - 1) for _ in range(n)]
    set_lines = [["SET", names[i], "=", str(values[i])] for i in range(n)]

    # Query a binding from the *early* region so retrieval genuinely spans
    # distance (all distractors and later bindings sit between def and query).
    early = max(1, n // 2)
    use_op = rng.random() < 0.5
    if use_op:
        i = rng.randint(0, early - 1)
        j = rng.randint(0, early - 1)
        op = rng.choice(["+", "-"])
        tval = (values[i] + values[j]) % MOD if op == "+" else (values[i] - values[j]) % MOD
        query_line = ["ANSWER", "(", names[i], op, names[j], ")", "mod", str(MOD)]
        target = str(tval)
        query = {"type": "op", "op": op, "vars": [names[i], names[j]]}
    else:
        i = rng.randint(0, early - 1)
        query_line = ["GET", names[i]]
        target = str(values[i] % MOD)
        query = {"type": "get", "var": names[i]}

    struct_len = sum(len(s) for s in set_lines) + len(query_line)
    budget = _distractor_budget(struct_len, density, seq_len)
    dlines = _emit_distractor_lines(rng, budget)

    # SETs first (early), then the distractor block, then the query: this
    # guarantees the queried binding is defined early and recalled across
    # distance — otherwise the recall test would be fake.
    out: List[List[str]] = list(set_lines) + dlines + [query_line]

    meta = {"query": query, "n_bindings": n}
    return out, target, meta


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def gen(
    mode: str,
    n_bindings: int = 8,
    distractor_density: float = 0.0,
    seq_len: int = 256,
    seed: int = 0,
) -> Tuple[str, str, Dict]:
    """Generate one example.

    Returns (prompt_str, target_str, meta). Deterministic per (args, seed).
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")

    rng = _mk_rng(mode, n_bindings, distractor_density, seq_len, seed)
    if mode == "state_track":
        lines, target, extra = _gen_state_track(rng, n_bindings, distractor_density, seq_len)
    else:
        lines, target, extra = _gen_assoc_recall(rng, n_bindings, distractor_density, seq_len)

    prompt = "\n".join(" ".join(line) for line in lines)
    n_tokens = sum(len(line) for line in lines)
    meta = {
        "mode": mode,
        "n_bindings": int(n_bindings),
        "distractor_density": float(distractor_density),
        "seq_len_target": int(seq_len),
        "n_tokens": n_tokens,
        "seed": int(seed),
        "mod": MOD,
        "target": target,
    }
    meta.update(extra)
    return prompt, target, meta


# ----------------------------------------------------------------------------
# CLI / smoke
# ----------------------------------------------------------------------------
def _smoke(seed: int = 0) -> None:
    for mode in MODES:
        prompt, target, meta = gen(mode, n_bindings=4, distractor_density=0.3, seq_len=48, seed=seed)
        print(f"=== mode={mode}  n_tokens={meta['n_tokens']}  target={target} ===")
        print(prompt)
        print(f"--- query: {meta['query']} ---\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="iota DSL task generator")
    ap.add_argument("--smoke", action="store_true", help="print one tiny example of each mode")
    ap.add_argument("--mode", choices=MODES, default="assoc_recall")
    ap.add_argument("--n_bindings", type=int, default=8)
    ap.add_argument("--distractor_density", type=float, default=0.0)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.smoke:
        _smoke(args.seed)
        return

    prompt, target, meta = gen(
        args.mode, args.n_bindings, args.distractor_density, args.seq_len, args.seed
    )
    print(prompt)
    print(f"\nTARGET: {target}")
    print(f"META: {meta}")


if __name__ == "__main__":
    main()
