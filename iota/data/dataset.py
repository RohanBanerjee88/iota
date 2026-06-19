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
from .dsl import gen
from .tokenizer import Tokenizer, get_tokenizer

# Disjoint example streams for train vs eval (no leakage).
TRAIN_OFFSET = 0
EVAL_OFFSET = 1 << 40


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
