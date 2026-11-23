"""Pass 12 (whose value does a wrong answer give): two-pass decoding, attribution, the pass and its SUMMARY section."""

import torch

from iota.attribution import RELS, attribute, fact_list, predict_values, run_attribution_pass
from iota.data.dataset import make_sweep_example
from iota.data.tokenizer import get_tokenizer
from iota.models import build_model
from scripts.sync_results import build_summary

TOK = get_tokenizer()


def _examples(nb, k):
    return [make_sweep_example(TOK, "assoc_recall", nb, 0.0, 256, 100 + s, n_queries=min(nb, 16),
                               query_pos="uniform") for s in range(k)]


class Oracle(torch.nn.Module):
    """Predicts every true next token, except that query q of example x answers `wrong[(x, q)]` instead."""

    def __init__(self, examples, wrong):
        super().__init__()
        self.examples, self.wrong = examples, wrong
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inp):
        B, T = inp.shape
        out = torch.full((B, T, TOK.vocab_size), -1.0)
        for i in range(B):
            x = next(j for j, e in enumerate(self.examples)
                     if e.tokens[: e.answer_spans[0][0]] == inp[i, : e.answer_spans[0][0]].tolist())
            e = self.examples[x]
            for j in range(min(T, len(e.tokens) - 1)):
                out[i, j, e.tokens[j + 1]] = 1.0
            for q, (s, _) in enumerate(e.answer_spans):
                if (x, q) in self.wrong:
                    v = self.wrong[(x, q)]
                    out[i, s - 1].fill_(-1.0)
                    out[i, s - 1, TOK.stoi[str(v // 10)]] = 1.0
                    out[i, s].fill_(-1.0)                      # the second digit, given the model's own first
                    out[i, s, TOK.stoi[str(v % 10)]] = 1.0
        return out


def _colliding_wrong_answers(examples):
    """For each query whose key shares digit count + last digit with another fact: answer that fact's value."""
    wrong = {}
    for x, e in enumerate(examples):
        facts = fact_list(e)
        for q, (k, v) in enumerate(zip(e.meta["query_keys"], e.meta["answers"])):
            mates = [vv for kk, vv in facts if kk != k and RELS["len_last"](k, kk) and vv != v]
            if mates:
                wrong[(x, q)] = mates[0]
    return wrong


def test_two_pass_decoding_recovers_true_and_designated_answers():
    exs = _examples(32, 6)
    wrong = _colliding_wrong_answers(exs)
    assert wrong                                                   # some collisions exist at 32 facts
    preds = predict_values(Oracle(exs, wrong), exs, TOK, "cpu", minibatch=4)
    for x, e in enumerate(exs):
        for q, v in enumerate(e.meta["answers"]):
            assert preds[x][q] == wrong.get((x, q), v)


def test_attribution_finds_the_colliding_fact():
    exs = _examples(32, 20)
    wrong = _colliding_wrong_answers(exs)
    c = attribute(exs, predict_values(Oracle(exs, wrong), exs, TOK, "cpu", minibatch=8))
    assert c["queries"] == sum(len(e.answer_spans) for e in exs)
    assert c["wrong"] == len(wrong) == c["in_prompt"] and c["correct"] == c["queries"] - c["wrong"]
    assert c["rel_len_last"] >= 0.75 * c["in_prompt"]              # a value shared by other facts splits its weight
    assert c["rel_len"] >= c["rel_len_last"] and c["rel_last"] >= c["rel_len_last"]
    assert c["rel_len_last"] / c["base_len_last"] > 5               # far above the share of colliding facts
    assert sum(c[f"q_len{d}"] for d in (1, 2, 3)) == c["queries"]


def test_pass_runs_on_a_tiny_model_and_the_summary_reports_it(tmp_path):
    cfg = {"arch": "gated_linear", "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64, "chunk_size": 16,
           "short_conv": 4, "legacy_gate_init": False, "decay_bias_init": 0.0, "vocab_size": TOK.vocab_size}
    out = str(tmp_path / "pass12_attribution.csv")
    rows = run_attribution_pass({"gated_linear_sweep": build_model(cfg)}, n=4, out_csv=out, cells=((8, 256),))
    assert len(rows) == 1 and rows[0]["queries"] == 4 * 8
    assert rows[0]["correct"] + rows[0]["wrong"] == rows[0]["queries"]
    assert "4j. Whose value does a wrong answer give" in build_summary(str(tmp_path), "rT")
