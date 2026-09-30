"""The Kaggle -> GitHub results link, and the eval OOM backoff.

The link is tested against a local bare repo (file:// remote), so no network or
token is needed: it must create the branch on first sync, MERGE on later syncs
(Session B must not wipe Session A), and never push weights.
"""

import json
import os
import subprocess

import torch

from iota.data.tokenizer import get_tokenizer
from iota.eval import _scores_with_backoff, run_pass
from iota.data.dataset import make_sweep_example
from iota.models import build_model
from iota.util import seed_everything
from scripts.sync_results import build_summary, sync

TOK = get_tokenizer()


def _git_show(bare, path):
    return subprocess.run(["git", "--git-dir", bare, "show", f"kaggle-results:{path}"],
                          capture_output=True, text=True)


def _fake_run_json(results, arch, assoc, state):
    hist = {"eval_acc": [{"step": 500, "balanced": (assoc + state) / 2, "per_query": assoc,
                          "exact": assoc / 2, "by_mode": {"assoc_recall": assoc, "state_track": state}}]}
    cfg = {"arch": arch, "train": {"lr": 0.0015}}
    with open(os.path.join(results, f"{arch}_sweep.json"), "w") as fh:
        json.dump({"config": cfg, "history": hist, "weights": f"{arch}_sweep.safetensors"}, fh)


def test_sync_creates_branch_merges_and_skips_weights(tmp_path):
    bare = str(tmp_path / "remote.git")
    subprocess.run(["git", "init", "--bare", "--quiet", bare], check=True)
    results = tmp_path / "results"
    (results / "logs").mkdir(parents=True)
    _fake_run_json(str(results), "transformer", 0.9, 0.8)
    (results / "transformer_sweep.safetensors").write_bytes(b"weights")
    (results / "logs" / "train.log").write_text("step 500 ...\n")

    assert sync(str(results), "r01", stage="train", remote_url=f"file://{bare}")
    assert _git_show(bare, "runs/r01/transformer_sweep.json").returncode == 0
    assert _git_show(bare, "runs/r01/logs/train.log").returncode == 0
    assert _git_show(bare, "runs/r01/transformer_sweep.safetensors").returncode != 0
    assert "r01" in _git_show(bare, "README.md").stdout

    # "Session B": a fresh disk that only has a new CSV -- Session A's files must survive.
    results_b = tmp_path / "results_b"
    results_b.mkdir()
    (results_b / "cost_profile.csv").write_text(
        "model,arch,seq_len,batch_size,params_m,peak_vram_mb,latency_ms,"
        "latency_ms_per_1k_tok,oom,device\n"
        "transformer_sweep,transformer,128,1,2.1,50.0,1.5,11.7,False,cuda\n")
    assert sync(str(results_b), "r01", stage="profile", remote_url=f"file://{bare}")
    assert _git_show(bare, "runs/r01/transformer_sweep.json").returncode == 0
    summary = _git_show(bare, "runs/r01/SUMMARY.md").stdout
    assert "transformer" in summary and "50.0 MB" in summary
    assert _git_show(bare, "runs/r01/syncs.log").stdout.count("stage=") == 2


def test_sync_failure_never_prints_the_token(tmp_path, capsys):
    secret = "ghp_SUPERSECRET123"
    ok = sync(str(tmp_path), "r01", token=secret, retries=1,
              remote_url=f"file:///nonexistent/{secret}/remote.git")
    out = capsys.readouterr()
    assert ok is False
    assert secret not in out.out + out.err
    assert "FAILED" in out.out


def test_sync_without_token_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    assert sync(str(tmp_path), "r01") is False


def test_summary_flags_dead_control_and_tabulates_passes(tmp_path):
    _fake_run_json(str(tmp_path), "gated_linear", 0.85, 0.01)
    seed_everything(0)
    model = build_model({"arch": "transformer", "vocab_size": TOK.vocab_size,
                         "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64})
    run_pass(1, {"transformer_sweep": model}, n=4, device="cpu",
             out_csv=str(tmp_path / "pass1_capacity.csv"), minibatch=4)
    s = build_summary(str(tmp_path), "r01")
    assert "control at chance" in s
    assert "| 128 |" in s  # every n_bindings cell of pass 1 is a row
    assert "pass2_length.csv` not synced yet" in s


class _OOMAbove(torch.nn.Module):
    """Wraps a model and raises CUDA OOM for any batch larger than `limit`."""

    def __init__(self, model, limit):
        super().__init__()
        self.model, self.limit = model, limit

    def forward(self, x):
        if x.shape[0] > self.limit:
            raise torch.cuda.OutOfMemoryError("simulated")
        return self.model(x)


def test_eval_backs_off_batch_instead_of_recording_oom():
    seed_everything(0)
    base = build_model({"arch": "transformer", "vocab_size": TOK.vocab_size,
                        "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64})
    exs = [make_sweep_example(TOK, "assoc_recall", 4, 0.0, 48, seed=s, n_queries=2) for s in range(6)]
    ref = _scores_with_backoff(base, exs, TOK.pad_id, "cpu", 8)
    got = _scores_with_backoff(_OOMAbove(base, 2), exs, TOK.pad_id, "cpu", 8)
    assert got == ref  # same scores, just computed at a smaller batch
    assert _scores_with_backoff(_OOMAbove(base, 0), exs, TOK.pad_id, "cpu", 8) is None
