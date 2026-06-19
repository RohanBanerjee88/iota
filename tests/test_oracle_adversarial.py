"""Hand-authored adversarial tests that PROVE the oracle is an independent,
correct evaluator — not a mirror of the generator's bookkeeping.

Every expected answer here is computed by hand (in the comment), never by the
generator. Cases deliberately stress: unusual spacing/blank lines, nested
parentheses, negative intermediate values, mod-boundary results (0 and 96), and
unparenthesized operator precedence (* binds tighter than +, mod binds loosest).
"""

import pytest

from iota.data import oracle
from iota.data import verifier
from iota.data.dsl import MOD

# (name, prompt, expected_answer)  -- expected computed by hand, see comments.
CASES = [
    (
        "unusual_spacing_and_blank_lines",
        # tabs, doubled spaces, leading indent, blank line; GET ignores all of it
        "SET   v0\t=   14\n\n   SET v1 = 3\nDISTRACTOR qx   lk\nGET v0",
        14,  # v0 == 14
    ),
    (
        "nested_parens",
        "SET v0 = 10\nSET v1 = 4\nSET v2 = 3\nANSWER ( ( v0 + v1 ) * v2 ) mod 97",
        42,  # (10+4)*3 = 42 ; 42 mod 97 = 42
    ),
    (
        "negative_intermediate_wraps_positive",
        "START x = 5\nx = ( x - 90 ) mod 97\nANSWER x",
        12,  # 5-90 = -85 ; -85 mod 97 = 12
    ),
    (
        "mod_boundary_zero",
        "SET v0 = 50\nSET v1 = 50\nANSWER ( v0 - v1 ) mod 97",
        0,  # 50-50 = 0
    ),
    (
        "mod_boundary_max_via_wrap",
        "START x = 0\nx = ( x - 1 ) mod 97\nANSWER x",
        96,  # -1 mod 97 = 96
    ),
    (
        "mod_boundary_max_via_product",
        "START x = 96\nx = ( x * 96 + 96 ) mod 97\nANSWER x",
        0,  # 96*96+96 = 9312 = 97*96 -> 9312 mod 97 = 0
    ),
    (
        "unparenthesized_precedence_mul_over_add",
        "SET v0 = 2\nSET v1 = 3\nSET v2 = 4\nANSWER v0 + v1 * v2 mod 97",
        14,  # 2 + (3*4) = 14 ; * before + , mod loosest -> 14 mod 97 = 14
    ),
    (
        "unparenthesized_mul_then_mod_wraps",
        "SET v0 = 20\nSET v1 = 10\nANSWER v0 * v1 mod 97",
        6,  # 20*10 = 200 ; 200 mod 97 = 6
    ),
    (
        "distractors_interspersed_are_ignored",
        "SET v0 = 7\nDISTRACTOR qx lk mn zz\nSET v1 = 8\nDISTRACTOR rr pp\nANSWER ( v0 + v1 ) mod 97",
        15,  # 7+8 = 15
    ),
    (
        "last_write_wins_reassignment",
        "SET v0 = 10\nSET v0 = 20\nGET v0",
        20,  # second SET shadows the first
    ),
]


@pytest.mark.parametrize("name,prompt,expected", CASES, ids=[c[0] for c in CASES])
def test_oracle_matches_hand_computed_answer(name, prompt, expected):
    assert oracle.solve(prompt) == str(expected), f"{name}: oracle disagreed"


@pytest.mark.parametrize("name,prompt,expected", CASES, ids=[c[0] for c in CASES])
def test_verifier_accepts_truth_rejects_corruption(name, prompt, expected):
    assert verifier.check(prompt, str(expected)) is True
    assert verifier.check(prompt, str((expected + 1) % MOD)) is False
