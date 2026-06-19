"""Tiny deterministic tokenizer (Phase 3).

Word/keyword level, with one twist: **numbers are split into individual digits**
so the vocabulary stays tiny (< ~100 tokens) instead of needing one id per value
in [0, MOD). Everything else (keywords, operators, variable names, the
accumulator, distractor noise words) is a single atomic token.

Round-trip is lossless on DSL prompts because two number-words are never
adjacent in the DSL (an operator/keyword/newline always separates them), so a
maximal run of digit tokens decodes back to exactly one number.

The tokenizer is also the source of the **true tokenized length** (post
digit-split). Callers thread that through so eval/profile use real token length
as the x-axis, not the nominal whitespace `seq_len`.
"""

from __future__ import annotations

from typing import List

from .dsl import ACC, KEYWORDS, NOISE_TOKENS, OPERATORS, VAR_NAMES

# Special tokens first so PAD == 0.
PAD = "<pad>"
EOS = "<eos>"
SPECIALS = [PAD, EOS]
DIGITS = [str(d) for d in range(10)]


def _build_vocab() -> List[str]:
    # Order is fixed and deterministic; PAD must be id 0.
    toks: List[str] = []
    toks += SPECIALS
    toks += DIGITS
    toks += KEYWORDS          # START SET GET ANSWER DISTRACTOR mod
    toks += OPERATORS         # = ( ) + - *
    toks += [ACC]             # x
    toks += VAR_NAMES         # v0..v63
    toks += NOISE_TOKENS      # qx lk ...
    # sanity: no duplicates
    assert len(toks) == len(set(toks)), "duplicate token in vocab"
    return toks


class Tokenizer:
    """Deterministic, closed-vocabulary tokenizer."""

    def __init__(self) -> None:
        self.itos = _build_vocab()
        self.stoi = {t: i for i, t in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD]
        self.eos_id = self.stoi[EOS]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    # -- encoding -----------------------------------------------------------
    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        """Encode a DSL string to token ids. Numbers are digit-split."""
        ids: List[int] = []
        for word in text.split():
            if word.isdigit():
                for ch in word:
                    ids.append(self.stoi[ch])
            else:
                if word not in self.stoi:
                    raise KeyError(f"token {word!r} not in vocabulary")
                ids.append(self.stoi[word])
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_number(self, value) -> List[int]:
        """Encode a target/answer integer (or its string) as digit tokens."""
        return [self.stoi[ch] for ch in str(value)]

    # -- decoding -----------------------------------------------------------
    def decode(self, ids: List[int], strip_special: bool = True) -> str:
        """Decode ids back to a whitespace-normalized DSL string.

        Consecutive digit tokens are merged into a single number word (valid for
        DSL prompts, where number-words are never adjacent).
        """
        words: List[str] = []
        num_buf: List[str] = []

        def flush():
            if num_buf:
                words.append("".join(num_buf))
                num_buf.clear()

        for i in ids:
            tok = self.itos[i]
            if tok in (PAD, EOS):
                flush()
                if not strip_special:
                    words.append(tok)
                continue
            if tok in DIGITS:
                num_buf.append(tok)
            else:
                flush()
                words.append(tok)
        flush()
        return " ".join(words)

    # -- length -------------------------------------------------------------
    def true_length(self, text: str, add_eos: bool = False) -> int:
        """True tokenized length (post digit-split), for use as the real x-axis."""
        return len(self.encode(text, add_eos=add_eos))


# Single shared instance is fine (stateless, deterministic).
_DEFAULT = Tokenizer()


def get_tokenizer() -> Tokenizer:
    return _DEFAULT


def _smoke() -> None:
    from .dsl import gen

    tok = get_tokenizer()
    print(f"vocab_size = {tok.vocab_size}  (pad={tok.pad_id}, eos={tok.eos_id})")
    for mode in ("state_track", "assoc_recall"):
        prompt, target, meta = gen(mode, 4, 0.2, 48, seed=0)
        ids = tok.encode(prompt)
        round_trip = tok.decode(ids)
        norm = " ".join(prompt.split())
        print(f"\n[{mode}] words={meta['n_tokens']}  true_len={len(ids)}  target={target}")
        print(f"  round-trip lossless: {round_trip == norm}")


if __name__ == "__main__":
    _smoke()
