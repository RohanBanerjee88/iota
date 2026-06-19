"""Phase 2 acceptance gate.

Generate 10,000 examples across both modes and all difficulty levels; for every
one, the oracle must reproduce the generator's target exactly (100% agreement),
and the verifier must accept the truth and reject a corruption.

This is the gate that stops the first session. If it passes, the data layer is
trustworthy and model work may begin.
"""

from iota.data import dsl
from iota.data import oracle
from iota.data import verifier

MODES = dsl.MODES
N_BINDINGS = [2, 4, 8, 16, 32, 64]
DENSITIES = [0.0, 0.3, 0.6]
SEQ_LENS = [128, 256, 512]
TARGET_N = 10_000


def _grid():
    """Yield (mode, n_bindings, density, seq_len, seed) tuples until TARGET_N."""
    count = 0
    seed = 0
    while count < TARGET_N:
        for mode in MODES:
            for nb in N_BINDINGS:
                for dens in DENSITIES:
                    for sl in SEQ_LENS:
                        yield (mode, nb, dens, sl, seed)
                        count += 1
                        if count >= TARGET_N:
                            return
        seed += 1


def test_oracle_agrees_and_verifier_catches_corruptions():
    n = 0
    agree = 0
    mismatches = []
    for mode, nb, dens, sl, seed in _grid():
        prompt, target, _ = dsl.gen(mode, nb, dens, sl, seed)
        ans = oracle.solve(prompt)
        n += 1
        if ans == target:
            agree += 1
        else:
            if len(mismatches) < 5:
                mismatches.append((mode, nb, dens, sl, seed, target, ans))
        # verifier accepts truth, rejects a corruption
        assert verifier.check(prompt, target) is True
        corrupt = str((int(target) + 1) % dsl.MOD)
        assert verifier.check(prompt, corrupt) is False

    assert n == TARGET_N
    assert agree == n, f"agreement {agree}/{n}; sample mismatches: {mismatches}"
