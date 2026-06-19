"""Shared SeqModel interface + common building blocks (Phase 4).

`train.py`, `eval.py`, `profile.py` only ever touch the SeqModel interface — they
never know which architecture they hold. All three models (transformer,
gated_linear, hybrid) assemble the same backbone (embedding -> pre-norm blocks ->
norm -> tied head); only the per-block *mixer* differs.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Common layers
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class FeedForward(nn.Module):
    def __init__(self, d: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d, d_ff)
        self.fc2 = nn.Linear(d_ff, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))


class Block(nn.Module):
    """Pre-norm residual block: x + mixer(norm(x)); x + ff(norm(x))."""

    def __init__(self, d: int, mixer: nn.Module, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.mixer = mixer
        self.norm2 = RMSNorm(d)
        self.ff = FeedForward(d, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Rotary positional embeddings (used by the dense-attention mixer)
# ---------------------------------------------------------------------------
def build_rope_cache(seq_len: int, head_dim: int, base: float = 10000.0, device=None, dtype=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
    cos = freqs.cos()
    sin = freqs.sin()
    if dtype is not None:
        cos, sin = cos.to(dtype), sin.to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, Dh)
    T, Dh = x.shape[-2], x.shape[-1]
    cos = cos[:T].view(1, 1, T, Dh // 2)
    sin = sin[:T].view(1, 1, T, Dh // 2)
    x1, x2 = x[..., : Dh // 2], x[..., Dh // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


# ---------------------------------------------------------------------------
# SeqModel interface + backbone assembly
# ---------------------------------------------------------------------------
class SeqModel(nn.Module):
    """Architecture-agnostic interface. See BUILD_PLAN.md §4."""

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # (B,T) -> (B,T,vocab)
        raise NotImplementedError

    @classmethod
    def from_config(cls, cfg: dict) -> "SeqModel":
        raise NotImplementedError

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LMBackbone(nn.Module):
    """Embedding -> blocks -> norm -> tied linear head."""

    def __init__(self, vocab_size: int, d_model: int, blocks: List[nn.Module], dropout: float = 0.0):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(blocks)
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.embed(tokens))
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def build_model(cfg: dict) -> SeqModel:
    """Factory dispatching on cfg['arch']. The only place arch names are resolved."""
    arch = cfg["arch"]
    if arch == "transformer":
        from .transformer import TransformerLM

        return TransformerLM.from_config(cfg)
    if arch == "gated_linear":
        from .gated_linear import GatedLinearLM

        return GatedLinearLM.from_config(cfg)
    if arch == "hybrid":
        from .hybrid import HybridLM

        return HybridLM.from_config(cfg)
    raise ValueError(f"unknown arch {arch!r}")
