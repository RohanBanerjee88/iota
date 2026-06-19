"""Hybrid model (Phase 4, see BUILD_PLAN.md §5).

Same backbone as the gated-linear contender, but the layers listed in
`cfg.full_attention_layers` (e.g. [3, 7]) are replaced with standard dense
attention. Expectation to validate (not assume): the hybrid recovers
transformer-level recall accuracy at materially lower cost than full dense.
"""

from __future__ import annotations

from .base import Block, LMBackbone, SeqModel
from .gated_linear import GatedLinearAttention
from .transformer import CausalSelfAttention


class HybridLM(SeqModel):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        d_ff,
        full_attention_layers=(),
        chunk_size=64,
        dropout=0.0,
        **_,
    ):
        super().__init__()
        full = set(full_attention_layers)
        self.full_attention_layers = sorted(full)
        blocks = []
        for i in range(n_layers):
            if i in full:
                mixer = CausalSelfAttention(d_model, n_heads, dropout)
            else:
                mixer = GatedLinearAttention(d_model, n_heads, chunk_size, dropout)
            blocks.append(Block(d_model, mixer, d_ff, dropout))
        self.backbone = LMBackbone(vocab_size, d_model, blocks, dropout)

    def forward(self, tokens):
        return self.backbone(tokens)

    @classmethod
    def from_config(cls, cfg: dict) -> "HybridLM":
        return cls(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            full_attention_layers=cfg.get("full_attention_layers", []),
            chunk_size=cfg.get("chunk_size", 64),
            dropout=cfg.get("dropout", 0.0),
        )
