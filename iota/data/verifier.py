"""Executable verifier (Phase 2).

`check(prompt, predicted)` recomputes the correct answer via the oracle and
compares it to a prediction. This is the only thing that decides whether a model
output is "correct" anywhere in the project — never a hand-written label.
"""

from __future__ import annotations

from typing import Optional

from . import oracle
from .dsl import MOD


def _to_val(s: str) -> Optional[int]:
    """Extract an integer answer in [0, MOD) from a possibly-noisy prediction.

    Accepts a bare number ("42") or a trailing number ("ANSWER 42"); takes the
    last integer-looking token so generated continuations are tolerated.
    """
    s = str(s).strip()
    for tok in reversed(s.split()):
        t = tok.lstrip("-")
        if t.isdigit():
            return int(tok) % MOD
    return None


def check(prompt: str, predicted: str) -> bool:
    """Return True iff `predicted` matches the oracle's answer for `prompt`."""
    try:
        truth = int(oracle.solve(prompt)) % MOD
    except Exception:
        return False
    val = _to_val(predicted)
    return val is not None and val == truth
