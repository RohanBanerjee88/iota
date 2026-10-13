"""Pass 10 (gate replay): the harness reproduces the baseline exactly, and each control
changes only what it claims to (order for shuffle, timing for geomean, the product for arith)."""

import math

import torch

from iota.data.dataset import collate_padded
from iota.data.tokenizer import get_tokenizer
from iota.eval import EVAL_OFFSET, _cell_examples, teacher_forced_scores
from iota.gates import _gla_layers
from iota.models import build_model
from iota.replay import _score_chunk, effective_log_gamma, modify, run_replay_pass

TOK = get_tokenizer()


def _model(arch="gated_linear"):
    torch.manual_seed(0)
    return build_model({"arch": arch, "vocab_size": TOK.vocab_size, "d_model": 32, "n_layers": 3,
                        "n_heads": 4, "d_ff": 64, "chunk_size": 16, "full_attention_layers": [1],
                        "decay_bias_init": 0.0, "short_conv": 4, "legacy_gate_init": True}).eval()


def _examples(n=6, nb=8):
    cell = {"mode": "assoc_recall", "seq_len": 128, "n_bindings": nb, "n_queries": min(nb, 16), "query_pos": "uniform"}
    return _cell_examples(TOK, cell, n, EVAL_OFFSET + 321)


def _record(m, exs):
    layers = dict(_gla_layers(m))
    cap = {}
    base = _score_chunk(m, exs, TOK.pad_id, "cpu", layers, capture=cap)
    return layers, cap, base


def test_scorer_matches_eval_and_replay_is_exact():
    m, exs = _model(), _examples()
    layers, cap, base = _record(m, exs)
    ref = teacher_forced_scores(m, exs, TOK.pad_id, "cpu", minibatch=len(exs))
    assert base == [(sum(s), len(s)) for s in ref]
    inp = collate_padded(exs, TOK.pad_id)[:, :-1]
    clean = m(inp)
    for ln in layers:
        t = modify(cap[ln], [e.true_len for e in exs], "replay")
        hook = layers[ln].g_proj.register_forward_hook(lambda mod, i, o, t=t: t)
        try:
            assert torch.equal(m(inp), clean)                      # bit-identical
        finally:
            hook.remove()
        assert _score_chunk(m, exs, TOK.pad_id, "cpu", layers, mods={ln: t}) == base


def test_shuffle_permutes_only_the_context_span():
    m, exs = _model(), _examples()
    _, cap, _ = _record(m, exs)
    rec = next(iter(cap.values()))
    spans = [e.true_len for e in exs]
    t = modify(rec, spans, "shuffle", torch.Generator().manual_seed(0))
    for i, L in enumerate(spans):
        for h in range(rec.shape[-1]):
            assert torch.equal(torch.sort(t[i, :L, h]).values, torch.sort(rec[i, :L, h]).values)  # same values
        assert not torch.equal(t[i, :L], rec[i, :L])                                             # new order
        assert torch.equal(t[i, L:], rec[i, L:])                                                 # suffix kept


def test_geomean_keeps_the_span_product_and_arith_does_not():
    m, exs = _model(), _examples()
    _, cap, _ = _record(m, exs)
    rec = next(iter(cap.values()))
    spans = [e.true_len for e in exs]
    g, a = modify(rec, spans, "geomean"), modify(rec, spans, "arith")
    for i, L in enumerate(spans):
        base = effective_log_gamma(rec[i, :L]).sum(0)
        assert torch.allclose(effective_log_gamma(g[i, :L]).sum(0), base, rtol=1e-4, atol=1e-3)
        assert torch.all(g[i, :L] == g[i, 0])                                  # one constant per head
        assert torch.all(effective_log_gamma(a[i, :L]).sum(0) >= base - 1e-3)  # Jensen: arith forgets less
        assert torch.equal(g[i, L:], rec[i, L:]) and torch.equal(a[i, L:], rec[i, L:])


def test_run_replay_pass_rows(tmp_path):
    models = {"gated_linear_sweep": _model(), "hybrid_sweep": _model("hybrid"),
              "transformer_sweep": _model("transformer")}
    rows = run_replay_pass(models, n=4, out_csv=str(tmp_path / "pass10_replay.csv"), cells=(8,), n_shuffles=2)
    gla = [r for r in rows if r["model"] == "gated_linear_sweep"]
    assert len(gla) == 3 * 5 and not any(r["model"] == "transformer_sweep" for r in rows)
    assert {r["layer"] for r in rows if r["model"] == "hybrid_sweep"} == {"L0", "L2"}
    for r in rows:
        if r["condition"] == "replay":
            assert r["delta"] == 0.0
    by = {(r["model"], r["layer"], r["condition"]): r for r in rows}
    for (mname, layer, c), r in by.items():
        if c in ("shuffle", "geomean"):                    # same product over the span as the baseline
            assert math.isclose(r["ctx_log10_product"], by[(mname, layer, "baseline")]["ctx_log10_product"],
                                rel_tol=1e-3, abs_tol=1e-2)
    from scripts.sync_results import build_summary
    assert "4h. Gate replay" in build_summary(str(tmp_path), "t")
