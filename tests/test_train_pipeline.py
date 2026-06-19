"""Phase 5 tests: the train/eval pipeline runs on CPU in seconds and learns.

This does NOT assert the near-100% milestone (too slow for unit tests) — it
checks that masked CE training reduces loss and that verifier-checked eval runs
end to end and returns a sane accuracy.
"""

import torch

from iota.data.dataset import DataSampler, TRAIN_OFFSET, collate
from iota.data.tokenizer import get_tokenizer
from iota.eval import exact_answer_accuracy
from iota.models import build_model
from iota.util import seed_everything


def _tiny_cfg():
    return {
        "arch": "transformer",
        "vocab_size": get_tokenizer().vocab_size,
        "d_model": 64,
        "n_layers": 2,
        "n_heads": 4,
        "d_ff": 128,
        "dropout": 0.0,
    }


def _spec():
    return {"mode": "assoc_recall", "n_bindings": [2], "distractor_density": 0.0,
            "seq_len": 32, "query_type": "get"}


def test_masked_training_reduces_loss():
    seed_everything(0)
    tok = get_tokenizer()
    model = build_model(_tiny_cfg())
    sampler = DataSampler(_spec(), tok)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    def step(cursor):
        ex = sampler.batch_examples(list(range(cursor, cursor + 16)), TRAIN_OFFSET)
        inp, tgt, mask = collate(ex, tok.pad_id)
        logits = model(inp)
        ce = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1), reduction="none"
        ).view_as(tgt)
        loss = (ce * mask).sum() / mask.sum().clamp_min(1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        return loss.item()

    first = step(0)
    last = first
    for c in range(1, 60):
        last = step(c * 16)
    assert last < first, f"loss did not drop: {first:.3f} -> {last:.3f}"


def test_exact_answer_accuracy_runs():
    seed_everything(0)
    model = build_model(_tiny_cfg())
    sampler = DataSampler(_spec(), get_tokenizer())
    acc, failures = exact_answer_accuracy(model, sampler, n=16, max_new=4)
    assert 0.0 <= acc <= 1.0
    assert isinstance(failures, list)
