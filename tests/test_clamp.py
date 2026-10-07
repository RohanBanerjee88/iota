"""Pass 9 (gate clamp): the clamp imposes exactly the requested gamma, undoes itself,
is a no-op on a model whose gates already equal it, and the pass reports every condition."""

import math

import torch

from iota.clamp import clamp_gates, head_mean_gamma, run_clamp_pass
from iota.data.dataset import collate_padded
from iota.data.tokenizer import get_tokenizer
from iota.eval import EVAL_OFFSET, _cell_examples
from iota.gates import _gla_layers, gate_stats
from iota.models import build_model

TOK = get_tokenizer()
KEEP = 1 / (1 + math.exp(-6))


def _model(legacy=True, arch="gated_linear"):
    torch.manual_seed(0)
    return build_model({"arch": arch, "vocab_size": TOK.vocab_size, "d_model": 32, "n_layers": 3,
                        "n_heads": 4, "d_ff": 64, "chunk_size": 16, "full_attention_layers": [1],
                        "decay_bias_init": 6.0, "short_conv": 4, "legacy_gate_init": legacy}).eval()


def _examples(n=4, nb=8):
    cell = {"mode": "assoc_recall", "seq_len": 128, "n_bindings": nb, "n_queries": min(nb, 16),
            "query_pos": "uniform"}
    return _cell_examples(TOK, cell, n, EVAL_OFFSET + 99)


@torch.no_grad()
def test_clamp_imposes_gamma_and_restores():
    m, exs = _model(), _examples()
    x = collate_padded(exs, TOK.pad_id)
    before = m(x)
    name = _gla_layers(m)[1][0]
    with clamp_gates(m, {name: torch.full((4,), KEEP)}):
        st = gate_stats(m, exs, TOK)
        clamped = m(x)
    g = torch.cat(list(st[name]["gamma"].values()))
    assert torch.allclose(g, torch.full_like(g, KEEP), atol=1e-5)           # exactly the clamp
    other = _gla_layers(m)[0][0]
    assert torch.cat(list(st[other]["gamma"].values())).std() > 1e-3        # other layers untouched
    assert not torch.allclose(before, clamped)                               # it changed the model...
    assert torch.equal(before, m(x))                                         # ...and fully undoes itself


@torch.no_grad()
def test_keep_clamp_is_a_noop_on_an_intended_init_model():
    # r07's control: its gates already sit at sigmoid(6), so clamping to it must change nothing.
    m, exs = _model(legacy=False), _examples()
    x = collate_padded(exs, TOK.pad_id)
    with clamp_gates(m, {n: torch.full((4,), KEEP) for n, _ in _gla_layers(m)}):
        clamped = m(x)
    assert torch.allclose(m(x), clamped, atol=1e-5)


def test_mean_clamp_uses_each_heads_mean():
    m, exs = _model(), _examples()
    means = head_mean_gamma(m, exs, TOK.pad_id, "cpu")
    name = _gla_layers(m)[2][0]
    with clamp_gates(m, {name: means[name]}):
        st = gate_stats(m, exs, TOK)
    g = torch.cat(list(st[name]["gamma"].values())).view(-1, 4)
    assert torch.allclose(g, means[name].expand_as(g), atol=1e-5)


def test_run_clamp_pass_conditions(tmp_path):
    models = {"gated_linear_sweep": _model(), "hybrid_sweep": _model(arch="hybrid"),
              "transformer_sweep": _model(arch="transformer")}
    rows = run_clamp_pass(models, n=3, out_csv=str(tmp_path / "pass9_clamp.csv"), cells=(8,))
    gla = [r for r in rows if r["model"] == "gated_linear_sweep"]
    hyb = [r for r in rows if r["model"] == "hybrid_sweep"]
    assert len(gla) == 2 + 2 * 3 and len(hyb) == 2 + 2 * 2                # none, keep_all, keep/mean per layer
    assert {r["condition"] for r in hyb} == {"none", "keep_all", "keep_L0", "mean_L0", "keep_L2", "mean_L2"}
    assert all(r["delta"] == 0 for r in rows if r["condition"] == "none")
    assert not any(r["model"] == "transformer_sweep" for r in rows)
    from scripts.sync_results import build_summary
    assert "4g. Gate-clamp" in build_summary(str(tmp_path), "t")
