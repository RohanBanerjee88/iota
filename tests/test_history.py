"""Pass 7 (matched histories): the conditions must differ ONLY in the history region."""

import torch

from iota.data.tokenizer import get_tokenizer
from iota.history import CELLS, PREFIX_LEN, fill_generated, make_item, run_history_pass, score_final
from iota.models import build_model
from iota.util import seed_everything

TOK = get_tokenizer()
GET = TOK.stoi["GET"]


def _get_keys(seq):
    """Keys asked by every GET in a token sequence."""
    keys, i = [], 0
    while i < len(seq):
        if seq[i] == GET:
            j = i + 1
            while TOK.itos[seq[j]].isdigit():
                j += 1
            keys.append(int("".join(TOK.itos[t] for t in seq[i + 1: j])))
            i = j
        else:
            i += 1
    return keys


def test_conditions_match_except_history():
    for ci, (nb, q) in enumerate(CELLS):
        for i in range(20):
            it = make_item(TOK, nb, q, 1000 * ci + i)
            ctx = it["contexts"]
            lens = {len(s) for s in ctx.values()}
            assert len(lens) == 1                                      # same total length
            final_q = TOK.encode(f"GET {it['keys'][it['target']]} =")
            for s in ctx.values():
                assert s[: PREFIX_LEN] == ctx["oracle"][: PREFIX_LEN]   # identical fact prefix
                assert s[it["final_pos"]:] == final_q                    # same final query, same position
            target = it["keys"][it["target"]]
            hist = _get_keys(ctx["oracle"][PREFIX_LEN: it["final_pos"]])
            assert target not in hist and len(set(hist)) == len(hist) == q   # held-out target, distinct
            assert sorted(_get_keys(ctx["reordered"][PREFIX_LEN: it["final_pos"]])) == sorted(hist)
            assert _get_keys(ctx["neutral"][PREFIX_LEN: it["final_pos"]]) == []  # no questions at all
            assert _get_keys(ctx["oracle"][:PREFIX_LEN]) == []


def test_generated_history_only_changes_answer_digits():
    seed_everything(0)
    model = build_model({"arch": "gated_linear", "vocab_size": TOK.vocab_size, "d_model": 32,
                         "n_layers": 2, "n_heads": 4, "d_ff": 64, "chunk_size": 16})
    items = [make_item(TOK, 8, q, 50 + q) for q in (0, 3, 7)]
    fill_generated(model, items, TOK.pad_id, "cpu", minibatch=2)
    for it in items:
        o, g = it["contexts"]["oracle"], it["contexts"]["generated"]
        assert len(o) == len(g)
        in_span = {p for s, e in it["oracle_spans"] for p in range(s, e)}
        assert all(a == b for p, (a, b) in enumerate(zip(o, g)) if p not in in_span)
        # unconstrained greedy: whatever the model emits (an untrained model may not emit digits)
        assert all(0 <= g[p] < TOK.vocab_size for p in in_span)


def test_scoring_runs_and_q0_conditions_collapse(tmp_path):
    seed_everything(0)
    model = build_model({"arch": "transformer", "vocab_size": TOK.vocab_size, "d_model": 32,
                         "n_layers": 2, "n_heads": 4, "d_ff": 64})
    rows = run_history_pass({"m": model}, n=6, device="cpu", minibatch=4,
                            cells=[(8, 0), (8, 3)], out_csv=str(tmp_path / "p7.csv"))
    q0 = [r for r in rows if r["n_prior"] == 0]
    assert len(q0) == 1 and q0[0]["condition"] == "none"
    q3 = {r["condition"]: r for r in rows if r["n_prior"] == 3}
    assert set(q3) == {"neutral", "oracle", "generated", "reordered"}
    assert q3["oracle"]["delta_vs_oracle"] == 0.0
    assert all(0.0 <= r["accuracy"] <= 1.0 for r in rows)
    it = make_item(TOK, 8, 3, 7)
    sc = score_final(model, [it], "oracle", TOK, "cpu")[0]
    assert sc["logprob"] <= 0 and isinstance(sc["correct"], bool)
