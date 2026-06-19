"""Dense transformer baseline (Phase 4).

Honest baseline: causal self-attention via `F.scaled_dot_product_attention`, which
dispatches to FlashAttention / memory-efficient kernels when available (and the
math kernel on CPU) — not a naive Python attention loop. Rotary position
embeddings so the same model can later be evaluated past its training length.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Block, LMBackbone, SeqModel, apply_rope, build_rope_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, rope_base: float = 10000.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = dropout
        self.rope_base = rope_base
        self._cos = None
        self._sin = None

    def _rope(self, T: int, device, dtype):
        if self._cos is None or self._cos.shape[0] < T or self._cos.device != device:
            cos, sin = build_rope_cache(max(T, 1), self.head_dim, self.rope_base, device, dtype)
            self._cos, self._sin = cos, sin
        return self._cos, self._sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,Dh)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        cos, sin = self._rope(T, x.device, x.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        o = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        o = o.transpose(1, 2).contiguous().view(B, T, D)
        return self.out(o)


class TransformerLM(SeqModel):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.0, **_):
        super().__init__()
        blocks = [
            Block(d_model, CausalSelfAttention(d_model, n_heads, dropout), d_ff, dropout)
            for _ in range(n_layers)
        ]
        self.backbone = LMBackbone(vocab_size, d_model, blocks, dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.backbone(tokens)

    @classmethod
    def from_config(cls, cfg: dict) -> "TransformerLM":
        return cls(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            dropout=cfg.get("dropout", 0.0),
        )
