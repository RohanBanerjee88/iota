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

from .base import Block, LMBackbone, SeqModel, apply_rope, build_rope_cache, conv_step


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, rope_base: float = 10000.0,
                 short_conv: int = 0):
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
        # OPTIONAL short causal depthwise conv on the input (OFF by default), the
        # same local-mixing knob GatedLinearAttention has. Multi-digit keys
        # ("SET 1 0 0 = ...") need neighbouring tokens fused before attention can
        # match them; with this on for EVERY architecture the comparison stays
        # fair and attention's O(T^2) cost is unchanged.
        self.short_conv = short_conv
        self.conv = (nn.Conv1d(d_model, d_model, short_conv, groups=d_model, bias=True)
                     if short_conv and short_conv > 1 else None)

    def _rope(self, T: int, device, dtype):
        if self._cos is None or self._cos.shape[0] < T or self._cos.device != device:
            cos, sin = build_rope_cache(max(T, 1), self.head_dim, self.rope_base, device, dtype)
            self._cos, self._sin = cos, sin
        return self._cos, self._sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        if self.conv is not None:  # causal: left-pad by kernel-1
            x = self.conv(F.pad(x.transpose(1, 2), (self.short_conv - 1, 0))).transpose(1, 2)
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

    # -- decode: preallocated KV cache --------------------------------------
    def init_cache(self, batch, max_len, device=None, dtype=torch.float32, prefill_len=0):
        D = self.n_heads * self.head_dim
        shape = (batch, self.n_heads, max_len, self.head_dim)
        mk = torch.randn if prefill_len else torch.zeros  # synthetic context = random
        cache = {"t": prefill_len,
                 "k": mk(shape, device=device, dtype=dtype),
                 "v": mk(shape, device=device, dtype=dtype)}
        if self.conv is not None:
            cache["conv"] = torch.zeros(batch, self.short_conv - 1, D, device=device, dtype=dtype)
        self._rope(max_len, device, dtype)  # build the RoPE table once, not per step
        return cache

    def step(self, x: torch.Tensor, cache: dict) -> torch.Tensor:
        """x: (B, D) for the token at position cache["t"] -> (B, D)."""
        B, D = x.shape
        if self.conv is not None:
            x, cache["conv"] = conv_step(self.conv, self.short_conv, cache["conv"], x)
        q, k, v = self.qkv(x).split(D, dim=-1)
        q = q.view(B, self.n_heads, 1, self.head_dim)
        k = k.view(B, self.n_heads, 1, self.head_dim)
        v = v.view(B, self.n_heads, 1, self.head_dim)
        t = cache["t"]
        cos, sin = self._rope(t + 1, x.device, x.dtype)
        q, k = apply_rope(q, cos[t:], sin[t:]), apply_rope(k, cos[t:], sin[t:])  # rotate at position t
        cache["k"][:, :, t] = k[:, :, 0]
        cache["v"][:, :, t] = v[:, :, 0]
        cache["t"] = t + 1
        o = F.scaled_dot_product_attention(q, cache["k"][:, :, : t + 1], cache["v"][:, :, : t + 1])
        return self.out(o.reshape(B, D))


class TransformerLM(SeqModel):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, dropout=0.0, short_conv=0, **_):
        super().__init__()
        blocks = [
            Block(d_model, CausalSelfAttention(d_model, n_heads, dropout, short_conv=short_conv),
                  d_ff, dropout)
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
            short_conv=cfg.get("short_conv", 0),
        )
