"""Phase 6 sweep machinery: multi-query encoding, teacher-forced scoring, passes."""

import torch
import torch.nn.functional as F

from iota.data.dataset import (
    TRAIN_OFFSET,
    CurriculumSampler,
    collate_sweep,
    make_sweep_example,
)
from iota.data.tokenizer import get_tokenizer
from iota.eval import cells_for_pass, run_pass, teacher_forced_scores, wilson_ci
from iota.models import build_model
from iota.util import seed_everything

TOK = get_tokenizer()


def test_multi_query_spans_and_2digit():
    e = make_sweep_example(TOK, "assoc_recall", 8, 0.0, 128, seed=1, n_queries=4, query_pos="uniform")
    assert len(e.answer_spans) == 4
    for (s, en) in e.answer_spans:
        assert en - s == 2  # 2-digit fixed width
    # decoded answer digits match meta answers (zero-padded)
    digits = "".join(TOK.itos[e.tokens[j]] for s, en in e.answer_spans for j in range(s, en))
    assert digits == "".join(f"{a:02d}" for a in e.meta["answers"])


def test_wilson_ci_bounds():
    lo, hi = wilson_ci(50, 100)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    assert wilson_ci(0, 0) == (0.0, 0.0)
    lo, hi = wilson_ci(100, 100)
    assert hi == 1.0 or abs(hi - 1.0) < 1e-9


def test_teacher_forced_scores_shape():
    seed_everything(0)
    model = build_model({"arch": "transformer", "vocab_size": TOK.vocab_size,
                         "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64})
    exs = [make_sweep_example(TOK, "assoc_recall", 4, 0.0, 48, seed=s, n_queries=3) for s in range(5)]
    scores = teacher_forced_scores(model, exs, TOK.pad_id, "cpu", minibatch=2)
    assert len(scores) == 5
    assert all(len(s) == 3 for s in scores)
    assert all(isinstance(b, bool) for s in scores for b in s)


def test_teacher_forced_matches_training_objective():
    # An overfit model should score its memorized examples correct under the same
    # teacher-forced rule that eval uses (scoring is consistent with training).
    seed_everything(0)
    model = build_model({"arch": "transformer", "vocab_size": TOK.vocab_size,
                         "d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 128})
    exs = [make_sweep_example(TOK, "assoc_recall", 4, 0.0, 32, seed=s, n_queries=2) for s in range(8)]
    inp, tgt, mask = collate_sweep(exs, TOK.pad_id)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for _ in range(150):
        logits = model(inp)
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                             reduction="none").view_as(tgt)
        loss = (ce * mask).sum() / mask.sum().clamp_min(1.0)
        opt.zero_grad(); loss.backward(); opt.step()
    scores = teacher_forced_scores(model, exs, TOK.pad_id, "cpu")
    exact = sum(1 for s in scores if all(s)) / len(scores)
    assert exact > 0.5, f"overfit model should score its own data; got {exact}"


def test_cells_for_pass_grids():
    assert [c["n_bindings"] for c in cells_for_pass(1)] == [2, 4, 8, 16, 32, 64, 128]
    assert [c["seq_len"] for c in cells_for_pass(2)] == [128, 256, 512, 1024, 2048, 4096, 8192]
    assert all(c["mode"] == "state_track" for c in cells_for_pass(3))


def test_run_pass_writes_rows(tmp_path):
    seed_everything(0)
    models = {
        "m": build_model({"arch": "transformer", "vocab_size": TOK.vocab_size,
                          "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64}),
    }
    out = str(tmp_path / "p1.csv")
    rows = run_pass(1, models, n=8, device="cpu", out_csv=out, minibatch=4)
    assert len(rows) == 7  # 7 capacity cells x 1 model
    assert all(0.0 <= r["accuracy_exact"] <= 1.0 for r in rows)
    import os
    assert os.path.exists(out)


def test_curriculum_sampler_deterministic():
    s = CurriculumSampler([
        {"mode": "assoc_recall", "n_bindings": {"min": 2, "max": 16}, "seq_len": {"min": 64, "max": 128},
         "n_queries": {"min": 1, "max": 4}, "weight": 0.7},
        {"mode": "state_track", "n_bindings": {"min": 2, "max": 8}, "seq_len": 64, "weight": 0.3},
    ], TOK)
    a = s.example(5, TRAIN_OFFSET)
    b = s.example(5, TRAIN_OFFSET)
    assert a.tokens == b.tokens  # deterministic per index
    modes = {s.example(i, TRAIN_OFFSET).meta["mode"] for i in range(40)}
    assert modes == {"assoc_recall", "state_track"}  # both components appear
