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


def test_summary_shows_lr_probe_with_best_marked(tmp_path):
    (tmp_path / "tune.csv").write_text(
        "arch,lr,grad_clip,steps,best_step,best_balanced,assoc,state,exact,final_balanced,seconds\n"
        "transformer,0.00075,1.0,4000,4000,0.40,0.5,0.3,0.2,0.40,600\n"
        "transformer,0.0015,1.0,4000,3500,0.62,0.7,0.54,0.4,0.60,600\n")
    s = build_summary(str(tmp_path), "r02")
    assert "LR probe" in s
    assert "| transformer ★ | 0.0015 |" in s and "| transformer | 0.00075 |" in s


def test_checkpoint_without_conv_is_stale_under_conv_config(tmp_path, monkeypatch):
    # The fingerprint must cover model keys, not just the curriculum: a checkpoint
    # trained with short_conv 0 must never be reused for a short_conv 4 config.
    import yaml
    import scripts.run_all as ra

    monkeypatch.setattr(ra, "RESULTS_DIR", str(tmp_path))
    cfg = yaml.safe_load(open("configs/sweep_transformer.yaml"))
    old = dict(cfg, short_conv=0, vocab_size=99)
    (tmp_path / "transformer_sweep.safetensors").write_bytes(b"x")
    hist = {"eval_acc": [{"step": 5000}]}
    (tmp_path / "transformer_sweep.json").write_text(json.dumps({"config": old, "history": hist}))
    assert ra.checkpoint_status("transformer")[0] == "stale"
    same = dict(cfg, vocab_size=99)  # identical model + curriculum, lr changes are fine
    same["train"] = dict(cfg["train"], lr=0.123)
    (tmp_path / "transformer_sweep.json").write_text(json.dumps({"config": same, "history": hist}))
    assert ra.checkpoint_status("transformer")[0] == "ok"


def test_aggregate_seeds_means_ranges_and_keeps_every_seed(tmp_path):
    from scripts.aggregate_seeds import aggregate

    hdr = ("model,mode,pass,seq_len_nominal,seq_len_true_tokens,n_bindings,n_queries,distractor_density,"
           "accuracy_exact,accuracy_per_query,ci_low,ci_high,ci_low_pq,ci_high_pq,n,seed\n")
    for run, v, oom in (("r04", 0.489, False), ("r05", 0.754, False), ("r06", None, True)):
        d = tmp_path / run
        d.mkdir()
        cell = "" if oom else f"{v}"
        (d / "pass1_capacity.csv").write_text(
            hdr + f"gated_linear_sweep,assoc_recall,1,256,386,64,16,0.0,0.0,{cell},,,,,1000,0\n")
    out = tmp_path / "agg"
    res = aggregate({r: str(tmp_path / r) for r in ("r04", "r05", "r06")}, str(out), make_plot=False)
    row = res["pass1_capacity.csv"][0]
    assert row["n_seeds"] == 2                                   # the OOM seed is not averaged in
    assert abs(row["accuracy_per_query"] - (0.489 + 0.754) / 2) < 1e-4
    assert (row["ci_low_pq"], row["ci_high_pq"]) == (0.489, 0.754)  # band = seed range
    assert "r04=0.489" in row["per_seed"] and "r06=OOM" in row["per_seed"]
    assert "r05=0.754" in (out / "SEEDS.md").read_text()


def test_fixinit_configs_change_only_the_gate_init(monkeypatch):
    # r07 = r04 with the intended gate init: same seed/lr/data/size, one flag flipped.
    # The transformer has no gate, so its fixinit config must match r04's exactly
    # (that is what lets --reuse take r04's checkpoint instead of retraining it).
    import yaml
    import scripts.run_all as ra
    from iota.util import sweep_config_path

    assert sweep_config_path("hybrid") == os.path.join("configs", "sweep_hybrid.yaml")
    monkeypatch.setenv("IOTA_CONFIG_DIR", "configs/fixinit")
    assert sweep_config_path("hybrid") == os.path.join("configs/fixinit", "sweep_hybrid.yaml")
    assert ra._cfg_path("hybrid") == sweep_config_path("hybrid")
    for arch in ("gated_linear", "hybrid", "transformer"):
        main = yaml.safe_load(open(f"configs/sweep_{arch}.yaml"))
        fix = yaml.safe_load(open(f"configs/fixinit/sweep_{arch}.yaml"))
        assert fix["train"] == dict(main["train"], seed=0), arch   # r04's seed, nothing else
        fm, ff = ra._fingerprint(main), ra._fingerprint(fix)
        diff = {k for k in set(fm) | set(ff) if fm.get(k) != ff.get(k)}
        assert diff == (set() if arch == "transformer" else {"legacy_gate_init"}), arch
        assert ff["legacy_gate_init"] is (arch == "transformer")


def _fake_ckpt(d, cfg, steps=5000):
    (d / "transformer_sweep.safetensors").write_bytes(b"x")
    (d / "transformer_sweep.json").write_text(
        json.dumps({"config": cfg, "history": {"eval_acc": [{"step": steps}]}}))


def test_train_reuse_takes_a_matching_checkpoint_and_rejects_a_stale_one(tmp_path, monkeypatch):
    import argparse
    import yaml
    import scripts.run_all as ra

    monkeypatch.setattr(ra, "RESULTS_DIR", str(tmp_path))
    monkeypatch.setattr(ra, "_publish", lambda *a: True)
    cfg = dict(yaml.safe_load(open("configs/sweep_transformer.yaml")), vocab_size=99)
    pushed, trained = [], []
    remote = {"r07": None, "r04": cfg}

    def pull(run_name, repo, token):
        if remote[repo] is None:
            return False
        _fake_ckpt(tmp_path, remote[repo])
        return True

    monkeypatch.setattr(ra, "hf_pull", pull)
    monkeypatch.setattr(ra, "hf_push", lambda run_name, repo, token: pushed.append(repo))
    import iota.train
    monkeypatch.setattr(iota.train, "train", lambda c, smoke=False: trained.append(c) or {"best_acc": 1.0})
    args = argparse.Namespace(only="transformer", no_hf=False, repo="r07", token=None, force=False,
                              smoke=False, reuse="transformer=r04")
    ra.stage_train(args)
    assert trained == [] and pushed == ["r07"]          # reused, and copied to the new repo
    assert ra.checkpoint_status("transformer")[0] == "ok"

    for f in tmp_path.iterdir():
        f.unlink()
    pushed.clear()
    remote["r04"] = dict(cfg, short_conv=0)              # a different model: must not be reused
    ra.stage_train(args)
    assert len(trained) == 1 and pushed == ["r07"]      # trained from scratch, then backed up


def test_bias2_configs_change_only_the_gate_bias():
    # r08 = r07 with the gate started unsaturated: decay_bias_init 6 -> 2, nothing else.
    import yaml
    import scripts.run_all as ra
    from iota.data.tokenizer import get_tokenizer
    from iota.models import build_model
    from iota.models.gated_linear import GatedLinearAttention

    for arch in ("gated_linear", "hybrid", "transformer"):
        r07 = yaml.safe_load(open(f"configs/fixinit/sweep_{arch}.yaml"))
        r08 = yaml.safe_load(open(f"configs/bias2/sweep_{arch}.yaml"))
        assert r08["train"] == r07["train"], arch
        f7, f8 = ra._fingerprint(r07), ra._fingerprint(r08)
        diff = {k for k in set(f7) | set(f8) if f7.get(k) != f8.get(k)}
        assert diff == (set() if arch == "transformer" else {"decay_bias_init"}), arch
        if arch != "transformer":
            model = build_model(dict(r08, vocab_size=get_tokenizer().vocab_size))
            gates = [m.g_proj for m in model.modules() if isinstance(m, GatedLinearAttention)]
            assert gates and all(torch.all(g.bias == 2.0) and torch.all(g.weight == 0) for g in gates)
