"""Gated linear attention — the contender (Phase 4, see BUILD_PLAN.md §5).

Recurrence (data-dependent scalar decay gate gamma_t in (0,1) per head, feature
map phi = elu + 1):

    S_t = gamma_t * S_{t-1} + phi(k_t) v_t^T     # (Dh x Dh) state
    z_t = gamma_t * z_{t-1} + phi(k_t)           # (Dh,) normalizer
    o_t = (phi(q_t)^T S_t) / (phi(q_t)^T z_t + eps)

Two equivalent implementations:
  * `recurrent_forward` — the literal token loop. O(T) wall-clock, slow. Kept
    ONLY as a `--reference` correctness check (small float tolerance).
  * `forward` (default) — chunk-parallel. Within a chunk, intra-chunk attention
    is computed in parallel; the state (S, z) is carried across chunks. The loop
    runs over T/chunk_size chunks, not over T tokens, so wall-clock genuinely
    scales linearly while staying fast at short lengths. All decay factors are
    formed in log-space and are <= 1, so there is no 1/prod(gamma) blow-up.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Block, LMBackbone, SeqModel

EPS = 1e-6


def _phi(x: torch.Tensor) -> torch.Tensor:
    """Feature map phi(x) = elu(x) + 1  (strictly positive)."""
    return F.elu(x) + 1.0


class GatedLinearAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, chunk_size: int = 64, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.chunk_size = chunk_size
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.g_proj = nn.Linear(d_model, n_heads)  # per-head scalar decay logit
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        # bias decay toward ~1 at init so memory persists early in training
        nn.init.constant_(self.g_proj.bias, 2.0)

    # -- shared projections --------------------------------------------------
    def _project(self, x: torch.Tensor):
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim
        q = _phi(self.q_proj(x)).view(B, T, H, Dh).transpose(1, 2)  # (B,H,T,Dh)
        k = _phi(self.k_proj(x)).view(B, T, H, Dh).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2)
        gamma = torch.sigmoid(self.g_proj(x)).transpose(1, 2)        # (B,H,T) in (0,1)
        return q, k, v, gamma

    def _merge(self, o: torch.Tensor) -> torch.Tensor:
        B, H, T, Dh = o.shape
        o = o.transpose(1, 2).contiguous().view(B, T, H * Dh)
        return self.out(self.drop(o))

    # -- chunk-parallel (default) -------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v, gamma = self._project(x)
        B, H, T, Dh = q.shape
        C = self.chunk_size
        log_g = torch.log(gamma.clamp_min(EPS))  # (B,H,T), <= 0

        S = x.new_zeros(B, H, Dh, Dh)
        z = x.new_zeros(B, H, Dh)
        outputs = []

        for start in range(0, T, C):
            end = min(start + C, T)
            qc = q[:, :, start:end]                 # (B,H,c,Dh)
            kc = k[:, :, start:end]
            vc = v[:, :, start:end]
            lg = log_g[:, :, start:end]             # (B,H,c)
            logP = torch.cumsum(lg, dim=-1)         # (B,H,c) cumulative decay
            P = torch.exp(logP)                     # (B,H,c) in (0,1]

            # intra-chunk decay matrix D[i,m] = exp(logP_i - logP_m), causal i>=m.
            # CRITICAL: mask the acausal (upper-triangle) entries to -inf BEFORE the
            # exp. There diff = logP_i - logP_m > 0, so exp(diff) would overflow to
            # +inf; torch.where would hide it in the forward (selecting 0), but in
            # backward autograd evaluates d/dx exp = exp = inf and multiplies it by
            # the zero mask -> 0*inf = NaN, poisoning every gradient. masked_fill
            # before exp keeps exp(-inf)=0 with a clean zero gradient.
            c = end - start
            causal = torch.tril(torch.ones(c, c, device=x.device, dtype=torch.bool))
            diff = logP.unsqueeze(-1) - logP.unsqueeze(-2)   # (B,H,c,c)
            diff = diff.masked_fill(~causal, float("-inf"))
            D = torch.exp(diff)

            A = torch.matmul(qc, kc.transpose(-1, -2)) * D   # (B,H,c,c)
            intra_v = torch.matmul(A, vc)                    # (B,H,c,Dh)

            q_scaled = qc * P.unsqueeze(-1)                  # (B,H,c,Dh)
            inter_v = torch.matmul(q_scaled, S)              # (B,H,c,Dh)

            num = inter_v + intra_v
            den = torch.matmul(q_scaled, z.unsqueeze(-1)).squeeze(-1) + A.sum(-1)  # (B,H,c)
            outputs.append(num / (den.unsqueeze(-1) + EPS))

            # carry state to next chunk
            logP_end = logP[:, :, -1:]                       # (B,H,1)
            decay_to_end = torch.exp(logP_end - logP)        # (B,H,c) in (0,1]
            k_tilde = kc * decay_to_end.unsqueeze(-1)        # (B,H,c,Dh)
            P_end = torch.exp(logP_end).unsqueeze(-1)        # (B,H,1,1)
            S = P_end * S + torch.matmul(k_tilde.transpose(-1, -2), vc)  # (B,H,Dh,Dh)
            z = torch.exp(logP_end) * z + k_tilde.sum(dim=2)            # (B,H,Dh)

        o = torch.cat(outputs, dim=2)
        return self._merge(o)

    # -- recurrent reference (correctness check only) -----------------------
    @torch.no_grad()
    def recurrent_forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v, gamma = self._project(x)
        B, H, T, Dh = q.shape
        S = x.new_zeros(B, H, Dh, Dh)
        z = x.new_zeros(B, H, Dh)
        outs = []
        for t in range(T):
            kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]   # (B,H,Dh)
            g = gamma[:, :, t]                                 # (B,H)
            S = g[..., None, None] * S + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            z = g[..., None] * z + kt
            num = torch.einsum("bhd,bhde->bhe", qt, S)         # (B,H,Dh)
            den = torch.einsum("bhd,bhd->bh", qt, z) + EPS
            outs.append(num / den.unsqueeze(-1))
        o = torch.stack(outs, dim=2)                           # (B,H,T,Dh)
        return self._merge(o)


class GatedLinearLM(SeqModel):
    def __init__(self, vocab_size, d_model, n_layers, n_heads, d_ff, chunk_size=64, dropout=0.0, **_):
        super().__init__()
        blocks = [
            Block(d_model, GatedLinearAttention(d_model, n_heads, chunk_size, dropout), d_ff, dropout)
            for _ in range(n_layers)
        ]
        self.backbone = LMBackbone(vocab_size, d_model, blocks, dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.backbone(tokens)

    @classmethod
    def from_config(cls, cfg: dict) -> "GatedLinearLM":
        return cls(
            vocab_size=cfg["vocab_size"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            n_heads=cfg["n_heads"],
            d_ff=cfg["d_ff"],
            chunk_size=cfg.get("chunk_size", 64),
            dropout=cfg.get("dropout", 0.0),
        )
