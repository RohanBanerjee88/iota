"""Phase 3 tests: vocab < ~100, lossless round-trip, true-length accounting."""

from iota.data import dsl
from iota.data.tokenizer import DIGITS, get_tokenizer


def test_vocab_size_under_100():
    tok = get_tokenizer()
    assert tok.vocab_size < 100, f"vocab too big: {tok.vocab_size}"
    assert tok.pad_id == 0


def test_round_trip_lossless_1000_examples():
    tok = get_tokenizer()
    for seed in range(500):
        for mode in dsl.MODES:
            prompt, _, _ = dsl.gen(mode, n_bindings=8, distractor_density=0.3, seq_len=128, seed=seed)
            ids = tok.encode(prompt)
            assert tok.decode(ids) == " ".join(prompt.split()), f"round-trip failed: {mode}/{seed}"


def test_numbers_are_digit_split():
    tok = get_tokenizer()
    ids = tok.encode("SET v0 = 96")
    # 96 -> two digit tokens '9','6'
    assert tok.itos[ids[-1]] == "6"
    assert tok.itos[ids[-2]] == "9"
    assert all(tok.itos[i] in DIGITS for i in ids[-2:])


def test_true_length_counts_digits():
    tok = get_tokenizer()
    # "GET v0" -> 2 tokens; "GET v12" -> 2 tokens (v12 is atomic); numbers split
    assert tok.true_length("ANSWER 100") == 1 + 3   # ANSWER + 1 + 0 + 0
    assert tok.true_length("SET v0 = 5") == 4       # SET v0 = 5
    assert tok.true_length("SET v0 = 5", add_eos=True) == 5


def test_true_length_differs_from_word_count():
    tok = get_tokenizer()
    prompt, _, meta = dsl.gen("assoc_recall", 16, 0.2, 256, seed=1)
    true_len = tok.true_length(prompt)
    # digit-splitting can only add tokens, never remove
    assert true_len >= meta["n_tokens"]


def test_encode_number_round_trips():
    tok = get_tokenizer()
    for v in (0, 5, 9, 10, 42, 96):
        ids = tok.encode_number(v)
        assert tok.decode(ids) == str(v)
