"""Phase 1 tests: the DSL generator is deterministic and well-formed."""

import pytest

from iota.data import dsl


def test_smoke_both_modes():
    for mode in dsl.MODES:
        prompt, target, meta = dsl.gen(mode, n_bindings=4, distractor_density=0.2, seq_len=64, seed=1)
        assert isinstance(prompt, str) and prompt.strip()
        assert target.isdigit()
        assert 0 <= int(target) < dsl.MOD
        assert meta["mode"] == mode
        assert meta["n_tokens"] == sum(len(line.split()) for line in prompt.splitlines())


def test_determinism():
    a = dsl.gen("assoc_recall", 8, 0.3, 256, seed=7)
    b = dsl.gen("assoc_recall", 8, 0.3, 256, seed=7)
    assert a == b
    c = dsl.gen("assoc_recall", 8, 0.3, 256, seed=8)
    assert a[0] != c[0]  # different seed -> different prompt


def test_seq_len_floor_respected():
    # With density 0, total length should land at or just above the target.
    for seq_len in (128, 256, 512):
        _, _, meta = dsl.gen("state_track", 4, 0.0, seq_len, seed=0)
        assert meta["n_tokens"] >= seq_len
        assert meta["n_tokens"] <= seq_len + 16  # at most one short trailing line


def test_density_increases_noise():
    _, _, lo = dsl.gen("assoc_recall", 8, 0.0, 64, seed=3)
    _, _, hi = dsl.gen("assoc_recall", 8, 0.8, 64, seed=3)
    assert hi["n_tokens"] > lo["n_tokens"]


def test_assoc_recall_query_defined_early():
    for seed in range(50):
        prompt, _, meta = dsl.gen("assoc_recall", 16, 0.3, 256, seed=seed)
        lines = prompt.splitlines()
        q = meta["query"]
        qkeys = [q["key"]]  # single GET uses integer key
        # the query line is last; every queried key must be SET before it
        last_set_idx = {}
        query_idx = len(lines) - 1
        for idx, line in enumerate(lines):
            toks = line.split()
            if toks and toks[0] == "SET":
                last_set_idx[toks[1]] = idx
        for k in qkeys:
            assert k in last_set_idx, f"{k} never SET"
            assert last_set_idx[k] < query_idx
            # "early": the queried binding lives in the first half of the pool
            assert int(k) < max(1, meta["n_bindings"] // 2)


def test_assoc_recall_rejects_too_many_bindings():
    with pytest.raises(ValueError):
        dsl.gen("assoc_recall", dsl.MAX_BINDINGS + 1, 0.0, 256, seed=0)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        dsl.gen("not_a_mode", 4, 0.0, 64, seed=0)


def test_integer_keys_and_capacity_to_128():
    prompt, _, meta = dsl.gen("assoc_recall", 128, 0.0, 600, seed=0, query_pos="uniform")
    assert meta["n_bindings"] == 128
    # keys are integers, digit-encoded (no atomic v-tokens)
    assert any(line.startswith("SET 100 =") for line in prompt.splitlines())


def test_multi_query_answers_match_oracle():
    from iota.data import oracle

    for seed in range(30):
        prompt, _, meta = dsl.gen("assoc_recall", 16, 0.1, 256, seed=seed,
                                  n_queries=5, query_pos="uniform")
        assert meta["n_queries"] == 5
        assert len(meta["answers"]) == 5
        # oracle.solve_all reproduces every queried answer, in order
        got = [int(a) for a in oracle.solve_all(prompt)]
        assert got == meta["answers"]
