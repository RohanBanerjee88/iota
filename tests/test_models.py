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


def test_from_config_round_trips_arch():
    for path in CONFIGS:
        cfg = _load(path)
        model = build_model(cfg)
        assert model.forward(torch.randint(0, VOCAB, (1, 8))).shape[-1] == VOCAB
