"""Phase 4 tests: SeqModel interface, param budget, chunk==recurrent equivalence."""

import torch
import yaml

from iota.models import build_model
from iota.models.gated_linear import GatedLinearAttention
from iota.data.tokenizer import get_tokenizer

VOCAB = get_tokenizer().vocab_size
CONFIGS = ["configs/tiny_transformer.yaml", "configs/gated_linear.yaml", "configs/hybrid.yaml"]


def _load(path):
    cfg = yaml.safe_load(open(path))
    cfg["vocab_size"] = VOCAB
    return cfg


def test_forward_shape_and_param_budget():
    for path in CONFIGS:
        cfg = _load(path)
        model = build_model(cfg)
        B, T = 2, 48
        tokens = torch.randint(0, VOCAB, (B, T))
        logits = model(tokens)
        assert logits.shape == (B, T, VOCAB)
        n = model.num_params()
        assert 1_000_000 <= n <= 3_000_000, f"{path}: {n} params outside 1-3M"


def test_chunked_matches_recurrent_reference():
    # The chunked (fast) path must match the literal recurrence within float tol.
    torch.manual_seed(0)
    d_model, n_heads = 64, 4
    layer = GatedLinearAttention(d_model, n_heads, chunk_size=16).eval()
    x = torch.randn(3, 50, d_model)  # T=50 not a multiple of chunk_size on purpose
    with torch.no_grad():
        fast = layer(x)
        ref = layer.recurrent_forward(x)
    assert torch.allclose(fast, ref, atol=1e-4, rtol=1e-4), (fast - ref).abs().max().item()


def test_chunked_matches_recurrent_various_chunk_sizes():
    torch.manual_seed(1)
    d_model, n_heads = 32, 2
    x = torch.randn(2, 33, d_model)
    base = GatedLinearAttention(d_model, n_heads, chunk_size=1).eval()
    for cs in (1, 4, 8, 64):
        layer = GatedLinearAttention(d_model, n_heads, chunk_size=cs).eval()
        layer.load_state_dict(base.state_dict())
        with torch.no_grad():
            fast = layer(x)
            ref = layer.recurrent_forward(x)
        assert torch.allclose(fast, ref, atol=1e-4, rtol=1e-4), f"chunk={cs}: {(fast-ref).abs().max()}"


def test_gla_backward_is_finite_under_large_activations():
    # Regression: the acausal entries of the intra-chunk decay matrix must be
    # masked BEFORE exp. Otherwise exp(+diff) overflows to inf and, although the
    # forward hides it via masking, backward computes 0*inf = NaN. This guards
    # the exact bug that made gated_linear fail to train.
    torch.manual_seed(0)
    layer = GatedLinearAttention(64, 4, chunk_size=16).train()
    for scale in (1.0, 10.0, 30.0):
        x = (torch.randn(4, 40, 64) * scale).requires_grad_(True)
        out = layer(x)
        assert torch.isfinite(out).all()
        layer.zero_grad()
        out.pow(2).mean().backward()
        for name, p in layer.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"non-finite grad in {name} at scale {scale}"


def test_from_config_round_trips_arch():
    for path in CONFIGS:
        cfg = _load(path)
        model = build_model(cfg)
        assert model.forward(torch.randint(0, VOCAB, (1, 8))).shape[-1] == VOCAB


def test_short_conv_is_causal_for_every_arch():
    # short_conv fuses neighbouring tokens (multi-digit keys); it must never see the future.
    from iota.data.tokenizer import get_tokenizer
    from iota.models import build_model

    V = get_tokenizer().vocab_size
    torch.manual_seed(0)
    for arch in ("transformer", "gated_linear", "hybrid"):
        m = build_model({"arch": arch, "vocab_size": V, "d_model": 32, "n_layers": 3, "n_heads": 4,
                         "d_ff": 64, "chunk_size": 8, "full_attention_layers": [1],
                         "short_conv": 4}).eval()
        x = torch.randint(0, V, (2, 24))
        x2 = x.clone()
        x2[:, 15] = (x2[:, 15] + 1) % V
        with torch.no_grad():
            assert torch.allclose(m(x)[:, :15], m(x2)[:, :15], atol=1e-5), arch


def test_decode_step_matches_full_forward_and_adds_no_weights():
    # The decode benchmark is only meaningful if token-by-token decoding computes
    # exactly what the full forward does -- and it must not change checkpoints.
    from iota.data.tokenizer import get_tokenizer
    from iota.models import build_model

    V = get_tokenizer().vocab_size
    torch.manual_seed(0)
    for arch in ("transformer", "gated_linear", "hybrid"):
        for sc in (0, 4):
            m = build_model({"arch": arch, "vocab_size": V, "d_model": 32, "n_layers": 3,
                             "n_heads": 4, "d_ff": 64, "chunk_size": 8,
                             "full_attention_layers": [1], "short_conv": sc}).eval()
            keys = set(m.state_dict())
            x = torch.randint(0, V, (2, 21))
            with torch.no_grad():
                full = m(x)
            cache = m.init_decode_cache(2, 21)
            steps = torch.stack([m.decode_step(x[:, t], cache) for t in range(21)], dim=1)
            assert torch.allclose(full, steps, atol=1e-5), (arch, sc)
            assert set(m.state_dict()) == keys


def test_decode_profile_state_is_constant_and_kv_grows():
    from iota.profile import run_decode_profile

    rows = run_decode_profile(out_csv=None, device="cpu", context_lens=(64, 512),
                              archs=["transformer", "gated_linear"])
    mb = {(r["arch"], r["context_len"]): r["cache_mb"] for r in rows}
    assert mb[("gated_linear", 64)] == mb[("gated_linear", 512)]          # fixed state
    assert mb[("transformer", 512)] > 5 * mb[("transformer", 64)]        # KV cache grows
