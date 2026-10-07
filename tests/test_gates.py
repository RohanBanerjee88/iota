"""Pass 8 (gate statistics): role labels, the frozen-gate signature, retention maths."""

import math

import torch

from iota.data.tokenizer import get_tokenizer
from iota.eval import EVAL_OFFSET, _cell_examples
from iota.gates import ROLES, gate_stats, run_gate_pass, token_roles
from iota.models import build_model

TOK = get_tokenizer()


def _model(legacy: bool, arch: str = "gated_linear"):
    torch.manual_seed(0)
    return build_model({"arch": arch, "vocab_size": TOK.vocab_size, "d_model": 32, "n_layers": 3,
                        "n_heads": 4, "d_ff": 64, "chunk_size": 16, "full_attention_layers": [1],
                        "decay_bias_init": 6.0, "short_conv": 4, "legacy_gate_init": legacy})


def _examples(nb=8, n=4):
    cell = {"mode": "assoc_recall", "seq_len": 128, "n_bindings": nb, "n_queries": min(nb, 16),
            "query_pos": "uniform"}
    return _cell_examples(TOK, cell, n, EVAL_OFFSET + 123)


def test_token_roles():
    ids = TOK.encode("SET 12 = 7 DISTRACTOR qx lk GET 12 =") + [TOK.stoi["0"], TOK.stoi["7"], TOK.eos_id]
    assert token_roles(ids, TOK) == (
        ["set_key"] * 4 + ["set_value"] + ["distractor"] * 3 + ["query"] * 4 + ["answer"] * 2 + ["other"])
    ex = _examples()[0]
    roles = token_roles(ex.tokens, TOK)
    for s, e in ex.answer_spans:                      # answer spans line up with the dataset's own
        assert roles[s:e] == ["answer"] * (e - s)
    assert all(r == "set_value" for r, t in zip(roles, ex.tokens) if r == "set_value" and TOK.itos[t].isdigit())


def test_intended_init_is_a_frozen_gate_and_legacy_is_not():
    exs = _examples()
    frozen = gate_stats(_model(legacy=False), exs, TOK)
    for d in frozen.values():
        assert abs(d["bias_mean"] - 6.0) < 1e-6 and d["weight_norm"] == 0
        for r in ROLES:
            g = d["gamma"][r]
            assert torch.allclose(g, torch.full_like(g, 1 / (1 + math.exp(-6))), atol=1e-6)
    legacy = gate_stats(_model(legacy=True), exs, TOK)
    for d in legacy.values():
        g = torch.cat([d["gamma"][r] for r in ROLES])
        assert 0.4 < float(g.mean()) < 0.6 and float(g.std()) > 1e-3   # input-dependent, around 0.5


def test_retention_matches_constant_gate_formula():
    exs = _examples(n=2)
    d = next(iter(gate_stats(_model(legacy=False), exs, TOK).values()))
    lg = math.log10(1 / (1 + math.exp(-6)))
    want = []
    for ex in exs:
        roles = token_roles(ex.tokens, TOK)
        end = ex.true_len - 1
        want += [(end - t) * lg for t, r in enumerate(roles) if r == "set_value" and t < end]
    assert len(want) == len(d["retention"])
    assert all(abs(a - b) < 1e-3 for a, b in zip(d["retention"], want))


def test_run_gate_pass_rows_and_skips_attention_only(tmp_path):
    models = {"gated_linear_sweep": _model(False), "hybrid_sweep": _model(True, "hybrid"),
              "transformer_sweep": _model(True, "transformer")}
    out = tmp_path / "pass8_gates.csv"
    rows = run_gate_pass(models, n=3, out_csv=str(out), cells=(8,))
    assert out.exists()
    assert {r["model"] for r in rows} == {"gated_linear_sweep", "hybrid_sweep"}  # no gates, no rows
    gla_layers = {r["layer"] for r in rows if r["model"] == "gated_linear_sweep"}
    hyb_layers = {r["layer"] for r in rows if r["model"] == "hybrid_sweep"}
    assert len(gla_layers) == 3 and len(hyb_layers) == 2           # hybrid layer 1 is attention
    sv = [r for r in rows if r["role"] == "set_value"]
    assert sv and all(r["fact_log10_retention"] != "" for r in sv)
    from scripts.sync_results import build_summary
    assert "4f. Decay gates" in build_summary(str(tmp_path), "t")
