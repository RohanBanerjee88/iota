"""Dataset plumbing for training/eval (Phase 5).

Each example is `encode(prompt) + encode(target_digits) + [EOS]`. Loss is masked
to the *completion* (answer digits + EOS) only — the prompt contains random
values and distractor noise that are not predictable, so training to predict them
would only add irreducible loss. This is still next-token cross-entropy, just
masked to the answer span.

`true_len` (the post-digit-split tokenized prompt length) is carried on every
example so eval/profile can use real token length as the x-axis.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from . import oracle  # noqa: F401  (kept handy for callers)
from .dsl import MOD, gen
from .tokenizer import Tokenizer, get_tokenizer

# Disjoint example streams for train vs eval (no leakage).
TRAIN_OFFSET = 0
EVAL_OFFSET = 1 << 40

# Every answer is a value in [0, MOD) -> a fixed 2-digit zero-padded field. Fixed
# width makes multi-query answers self-delimiting (no separator token) and lets a
# single teacher-forced forward pass score each query by digit position.
ANSWER_WIDTH = 2


@dataclass
class Example:
    tokens: List[int]
    answer_mask: List[int]  # 1 on answer (completion) tokens
    prompt: str
    target: str
    true_len: int           # tokenized prompt length (post digit-split)
    meta: Dict


def make_example(
    tok: Tokenizer,
    mode: str,
    n_bindings: int,
    distractor_density: float,
    seq_len: int,
    seed: int,
    query_type: Optional[str] = None,
    query_pos: str = "early",
) -> Example:
    prompt, target, meta = gen(
        mode, n_bindings, distractor_density, seq_len, seed,
        query_type=query_type, query_pos=query_pos,
    )
    p_ids = tok.encode(prompt)
    a_ids = tok.encode_number(target) + [tok.eos_id]
    tokens = p_ids + a_ids
    answer_mask = [0] * len(p_ids) + [1] * len(a_ids)
    return Example(tokens, answer_mask, prompt, target, len(p_ids), meta)


class DataSampler:
    """Deterministic on-the-fly sampler over a training/eval distribution.

    Per-example, draws n_bindings from a list and a fresh content seed, so the
    stream is reproducible and (train vs eval) disjoint via the offset.
    """

    def __init__(self, spec: Dict, tok: Optional[Tokenizer] = None):
        self.tok = tok or get_tokenizer()
        self.mode = spec["mode"]
        nb = spec["n_bindings"]
        self.n_bindings = list(nb) if isinstance(nb, (list, tuple)) else [int(nb)]
        self.density = float(spec.get("distractor_density", 0.0))
        self.seq_len = int(spec["seq_len"])
        self.query_type = spec.get("query_type", None)
        self.query_pos = spec.get("query_pos", "early")

    def example(self, index: int, offset: int) -> Example:
        r = random.Random(offset + index)
        nb = r.choice(self.n_bindings)
        content_seed = r.randrange(1 << 30)
        return make_example(
            self.tok, self.mode, nb, self.density, self.seq_len, content_seed,
            query_type=self.query_type, query_pos=self.query_pos,
        )

    def batch_examples(self, indices: List[int], offset: int) -> List[Example]:
        return [self.example(i, offset) for i in indices]


# ---------------------------------------------------------------------------
# Sweep dataset (Phase 6): fixed-width 2-digit answers, multi-query, teacher-
# forced scoring. Kept separate from the milestone path above (which the §0 gate
# reproduces) so nothing here disturbs that.
# ---------------------------------------------------------------------------
@dataclass
class SweepExample:
    tokens: List[int]
    answer_spans: List[tuple]   # [start, end) per query, into `tokens`
    true_len: int               # tokenized prompt length (real x-axis)
    prompt: str
    meta: Dict


def make_sweep_example(
    tok: Tokenizer,
    mode: str,
    n_bindings: int,
    distractor_density: float,
    seq_len: int,
    seed: int,
    n_queries: int = 1,
    query_pos: str = "uniform",
) -> SweepExample:
    prompt, _, meta = gen(
        mode, n_bindings, distractor_density, seq_len, seed,
        n_queries=n_queries, query_pos=query_pos,
    )
    p_ids = tok.encode(prompt)
    a_ids: List[int] = []
    spans: List[tuple] = []
    for a in meta["answers"]:
        s = len(p_ids) + len(a_ids)
        for ch in f"{a % MOD:0{ANSWER_WIDTH}d}":
            a_ids.append(tok.stoi[ch])
        spans.append((s, s + ANSWER_WIDTH))
    a_ids.append(tok.eos_id)
    tokens = p_ids + a_ids
    return SweepExample(tokens, spans, len(p_ids), prompt, meta)


def _draw(r: random.Random, spec):
    """Sample a scalar from an int, a list of choices, or {min,max}."""
    if isinstance(spec, dict):
        return r.randint(int(spec["min"]), int(spec["max"]))
    if isinstance(spec, (list, tuple)):
        return r.choice(list(spec))
    return int(spec)


class CurriculumSampler:
    """Deterministic mixed-curriculum sampler producing SweepExamples.

    `curriculum` is a list of components, each a dict like:
        {mode, n_bindings, seq_len, distractor_density, n_queries, query_pos, weight}
    where n_bindings/seq_len/n_queries may be an int, a list, or {min,max}.
    Per example a component is chosen by weight and its params are sampled.
    """

    def __init__(self, curriculum: List[Dict], tok: Optional[Tokenizer] = None):
        self.tok = tok or get_tokenizer()
        self.components = curriculum
        self.weights = [float(c.get("weight", 1.0)) for c in curriculum]

    def example(self, index: int, offset: int) -> SweepExample:
        r = random.Random(offset + index)
        comp = r.choices(self.components, weights=self.weights, k=1)[0]
        mode = comp["mode"]
        nb = _draw(r, comp.get("n_bindings", 8))
        seq_len = _draw(r, comp.get("seq_len", 256))
        density = float(comp.get("distractor_density", 0.0))
        query_pos = comp.get("query_pos", "uniform")
        nq = _draw(r, comp.get("n_queries", 1)) if mode == "assoc_recall" else 1
        nq = max(1, min(int(nq), int(nb)))
        content_seed = r.randrange(1 << 30)
        return make_sweep_example(
            self.tok, mode, nb, density, seq_len, content_seed,
            n_queries=nq, query_pos=query_pos,
        )

    def batch(self, indices: List[int], offset: int) -> List[SweepExample]:
        return [self.example(i, offset) for i in indices]


def collate_sweep(examples: List[SweepExample], pad_id: int):
    """Right-pad SweepExamples; loss mask covers answer tokens + EOS."""
    maxlen = max(len(e.tokens) for e in examples)
    B = len(examples)
    inp = torch.full((B, maxlen), pad_id, dtype=torch.long)
    amask = torch.zeros((B, maxlen), dtype=torch.float)
    for i, e in enumerate(examples):
        inp[i, : len(e.tokens)] = torch.tensor(e.tokens, dtype=torch.long)
        # answer digit tokens + the EOS right after the last answer
        for (s, en) in e.answer_spans:
            amask[i, s:en] = 1.0
        amask[i, len(e.tokens) - 1] = 1.0  # EOS
    return inp[:, :-1], inp[:, 1:], amask[:, 1:]


def collate_padded(examples: List[SweepExample], pad_id: int):
    """Right-pad SweepExamples into (input_ids,) for teacher-forced eval."""
    maxlen = max(len(e.tokens) for e in examples)
    inp = torch.full((len(examples), maxlen), pad_id, dtype=torch.long)
    for i, e in enumerate(examples):
        inp[i, : len(e.tokens)] = torch.tensor(e.tokens, dtype=torch.long)
    return inp


def collate(examples: List[Example], pad_id: int):
    """Right-pad a list of examples into training tensors.

    Returns input_ids, target_ids, loss_mask  (all (B, T-1)). Causal + right-pad
    is safe: answer tokens precede the padding, so masked-answer loss never sees
    pad. Returns the answer label mask (loss only on completion tokens).
    """
    maxlen = max(len(e.tokens) for e in examples)
    B = len(examples)
    inp = torch.full((B, maxlen), pad_id, dtype=torch.long)
    for i, e in enumerate(examples):
        inp[i, : len(e.tokens)] = torch.tensor(e.tokens, dtype=torch.long)
    amask = torch.zeros((B, maxlen), dtype=torch.float)
    for i, e in enumerate(examples):
        amask[i, : len(e.answer_mask)] = torch.tensor(e.answer_mask, dtype=torch.float)

    input_ids = inp[:, :-1]
    target_ids = inp[:, 1:]
    # loss where the *label* is an answer token
    loss_mask = amask[:, 1:]
    return input_ids, target_ids, loss_mask
