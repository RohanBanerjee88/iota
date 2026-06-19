"""Symbolic oracle (Phase 2) — the ground-truth answerer.

`solve(prompt)` parses the DSL and computes the correct answer *exactly*, with a
real expression evaluator rather than trusting any bookkeeping from the
generator. Agreement between this oracle and the generator's target (across many
thousands of examples) is what validates both.
"""

from __future__ import annotations

from typing import Dict, List

from .dsl import MOD


def _is_int(tok: str) -> bool:
    return tok.isdigit()


def _eval(tokens: List[str], env: Dict[str, int]) -> int:
    """Evaluate an expression token list with precedence: mod < +/- < * < primary.

    Grammar:
        mod_expr := add_sub ('mod' add_sub)*
        add_sub  := mul (('+'|'-') mul)*
        mul      := primary ('*' primary)*
        primary  := '(' mod_expr ')' | INT | IDENT
    """
    p = [0]

    def peek():
        return tokens[p[0]] if p[0] < len(tokens) else None

    def adv() -> str:
        t = tokens[p[0]]
        p[0] += 1
        return t

    def primary() -> int:
        t = peek()
        if t == "(":
            adv()
            v = mod_expr()
            if peek() != ")":
                raise ValueError("expected ')'")
            adv()
            return v
        if t is None:
            raise ValueError("unexpected end of expression")
        if _is_int(t):
            adv()
            return int(t)
        adv()  # identifier
        if t not in env:
            raise ValueError(f"unknown variable {t!r}")
        return env[t]

    def mul() -> int:
        v = primary()
        while peek() == "*":
            adv()
            v = v * primary()
        return v

    def add_sub() -> int:
        v = mul()
        while peek() in ("+", "-"):
            op = adv()
            r = mul()
            v = v + r if op == "+" else v - r
        return v

    def mod_expr() -> int:
        v = add_sub()
        while peek() == "mod":
            adv()
            m = add_sub()
            v = v % m
        return v

    v = mod_expr()
    if p[0] != len(tokens):
        raise ValueError(f"trailing tokens in expression: {tokens[p[0]:]}")
    return v


def solve(prompt: str) -> str:
    """Parse the DSL prompt and return the exact answer as a string in [0, MOD)."""
    env: Dict[str, int] = {}
    answer = None

    for raw in prompt.strip().splitlines():
        toks = raw.split()
        if not toks:
            continue
        head = toks[0]

        if head == "DISTRACTOR":
            continue
        if head == "START":
            # START <var> = <expr>
            if len(toks) < 4 or toks[2] != "=":
                raise ValueError(f"bad START line: {raw!r}")
            env[toks[1]] = _eval(toks[3:], env)
        elif head == "SET":
            if len(toks) < 4 or toks[2] != "=":
                raise ValueError(f"bad SET line: {raw!r}")
            env[toks[1]] = _eval(toks[3:], env)
        elif head == "GET":
            if len(toks) != 2:
                raise ValueError(f"bad GET line: {raw!r}")
            if toks[1] not in env:
                raise ValueError(f"GET of undefined variable {toks[1]!r}")
            answer = env[toks[1]] % MOD
        elif head == "ANSWER":
            answer = _eval(toks[1:], env) % MOD
        else:
            # generic assignment: <var> = <expr>   (Mode A update lines)
            if len(toks) >= 3 and toks[1] == "=":
                env[toks[0]] = _eval(toks[2:], env)
            else:
                raise ValueError(f"cannot parse line: {raw!r}")

    if answer is None:
        raise ValueError("prompt contained no GET/ANSWER query")
    return str(answer % MOD)


def solve_all(prompt: str) -> List[str]:
    """Return the ordered list of answers for EVERY GET/ANSWER query in the prompt.

    Used to verifier-check multi-query examples (single-query returns a 1-list).
    The oracle is identifier-agnostic, so it handles both integer keys (from the
    generator) and named variables (from the adversarial tests).
    """
    env: Dict[str, int] = {}
    answers: List[str] = []
    for raw in prompt.strip().splitlines():
        toks = raw.split()
        if not toks:
            continue
        head = toks[0]
        if head == "DISTRACTOR":
            continue
        if head in ("START", "SET"):
            if len(toks) < 4 or toks[2] != "=":
                raise ValueError(f"bad {head} line: {raw!r}")
            env[toks[1]] = _eval(toks[3:], env)
        elif head == "GET":
            if len(toks) != 2 or toks[1] not in env:
                raise ValueError(f"bad GET line: {raw!r}")
            answers.append(str(env[toks[1]] % MOD))
        elif head == "ANSWER":
            answers.append(str(_eval(toks[1:], env) % MOD))
        elif len(toks) >= 3 and toks[1] == "=":
            env[toks[0]] = _eval(toks[2:], env)
        else:
            raise ValueError(f"cannot parse line: {raw!r}")
    if not answers:
        raise ValueError("prompt contained no GET/ANSWER query")
    return answers
