"""Phase 2 tests: the verifier accepts the truth and rejects corruptions."""

from iota.data import dsl
from iota.data import oracle
from iota.data import verifier


def _corrupt(target: str) -> str:
    return str((int(target) + 1) % dsl.MOD)


def test_verifier_accepts_oracle_answer():
    for mode in dsl.MODES:
        for seed in range(100):
            prompt, target, _ = dsl.gen(mode, 8, 0.3, 128, seed=seed)
            assert verifier.check(prompt, oracle.solve(prompt)) is True
            assert verifier.check(prompt, target) is True


def test_verifier_rejects_corruption():
    for mode in dsl.MODES:
        for seed in range(100):
            prompt, target, _ = dsl.gen(mode, 8, 0.3, 128, seed=seed)
            assert verifier.check(prompt, _corrupt(target)) is False


def test_verifier_tolerates_trailing_token_format():
    prompt, target, _ = dsl.gen("assoc_recall", 8, 0.0, 64, seed=0)
    assert verifier.check(prompt, f"ANSWER {target}") is True


def test_verifier_handles_garbage_prediction():
    prompt, _, _ = dsl.gen("state_track", 4, 0.0, 64, seed=0)
    assert verifier.check(prompt, "no number here") is False
