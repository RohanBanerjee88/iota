"""Pass 11 (which facts are retrievable): query bookkeeping, aggregation, the pass itself and its SUMMARY section."""

import csv
import os

from iota.data.dataset import make_sweep_example
from iota.data.tokenizer import get_tokenizer
from iota.models import build_model
from iota.retrieval import FIELDS, query_meta, run_retrieval_pass, tally
from scripts.sync_results import build_summary

TOK = get_tokenizer()


def _example(nb, seed, seq=256):
    return make_sweep_example(TOK, "assoc_recall", nb, 0.0, seq, seed, n_queries=min(nb, 16), query_pos="uniform")


def test_query_meta_matches_the_prompt_and_the_answer_tokens():
    e = _example(32, seed=3)
    sets = [line.split() for line in e.prompt.splitlines() if line.startswith("SET ")]
    keys = [s[1] for s in sets]
    qm = query_meta(e)
    assert len(qm) == len(e.answer_spans) == 16
    assert [q["query_rank"] for q in qm] == list(range(16))
    for q, (s, en) in zip(qm, e.answer_spans):
        i = keys.index(str(q["key"]))
        assert q["facts_after"] == len(sets) - 1 - i
        assert int(sets[i][3]) == q["value"]
        assert TOK.decode(e.tokens[s:en]).replace(" ", "") == f"{q['value']:02d}"


def test_tally_bins_a_recency_buffer_by_facts_after():
    exs = [_example(32, seed=s) for s in range(20)]
    scores = [[q["facts_after"] < 4 for q in query_meta(e)] for e in exs]     # a model that keeps the last 4 facts
    agg = tally(exs, scores)
    for (dim, b), (h, t) in agg.items():
        if dim == "facts_after":
            assert h == (t if b < 4 else 0)
    n_queries = sum(len(e.answer_spans) for e in exs)
    for dim in ("facts_after", "key", "value", "query_rank"):
        assert sum(t for (d, _), (_, t) in agg.items() if d == dim) == n_queries


def test_pass_runs_on_a_tiny_model_and_the_summary_reports_it(tmp_path):
    cfg = {"arch": "gated_linear", "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64, "chunk_size": 16,
           "short_conv": 4, "legacy_gate_init": False, "decay_bias_init": 0.0, "vocab_size": TOK.vocab_size}
    out = str(tmp_path / "pass11_retrieval.csv")
    rows = run_retrieval_pass({"gated_linear_sweep": build_model(cfg)}, n=4, out_csv=out, cells=((8, 256), (8, 640)))
    assert set(next(csv.DictReader(open(out))).keys()) == set(FIELDS)
    for sl in (256, 640):
        fa = [r for r in rows if r["seq_len"] == sl and r["dim"] == "facts_after"]
        assert sorted(r["bin"] for r in fa) == list(range(8)) and sum(r["total"] for r in fa) == 4 * 8
    long = [r for r in rows if r["seq_len"] == 640][0]["true_len_mean"]
    short = [r for r in rows if r["seq_len"] == 256][0]["true_len_mean"]
    assert long > short + 300                       # same 8 facts, only the filler grows
    md = build_summary(str(tmp_path), "rT")
    assert "4i. Which facts are retrievable" in md and "| 8 |" in md and os.path.exists(out)
