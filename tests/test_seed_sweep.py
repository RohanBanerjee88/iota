"""The seed sweep (scripts/seed_sweep.py): statistics, the summary, and an end-to-end run.

The end-to-end test runs two tiny CPU jobs AT ONCE (separate worker processes), publishes to a local bare repo
(no token, no network), and checks that a second invocation skips the published jobs.
"""

import json
import os
import subprocess
import sys

import yaml

from scripts.seed_sweep import (REPO_ROOT, collect_rows, fisher_exact, parse_seeds, sweep_summary)


def test_fisher_exact_and_seed_parsing():
    assert abs(fisher_exact(3, 6, 7, 2) - 0.1534) < 1e-3          # learned 3/9 vs 7/9 (PROGRESS 2026-10-26)
    assert abs(fisher_exact(0, 5, 5, 0) - 0.00794) < 1e-4
    assert abs(fisher_exact(2, 2, 2, 2) - 1.0) < 1e-9
    assert parse_seeds("5-7,12,6") == [5, 6, 7, 12]


def _write_job(d, cell, seed, assoc_curve, status="stopped", rc=0):
    name = f"{cell}_s{seed:02d}"
    evals = [{"step": s, "by_mode": {"assoc_recall": a, "state_track": 1.0}} for s, a in assoc_curve]
    json.dump({"status": status, "eval_acc": evals}, open(os.path.join(d, f"{name}_history.json"), "w"))
    json.dump({"job": name, "cell": cell, "seed": seed, "returncode": rc, "status": status, "seconds": 3600,
               "gpu": "0", "lr": 0.0015, "decay_bias_init": 0.0 if cell == "bias0" else 2.0},
              open(os.path.join(d, f"{name}_meta.json"), "w"))


def test_summary_counts_on_track_jobs_and_writes_csv(tmp_path):
    d = str(tmp_path)
    _write_job(d, "bias0", 5, [(1500, 0.02), (3000, 0.2), (6000, 0.55)])
    _write_job(d, "bias0", 6, [(1500, 0.03), (3000, 0.1), (6000, 0.31)])
    _write_job(d, "bias2", 5, [(1500, 0.08), (3000, 0.11), (6000, 0.12)])
    _write_job(d, "bias2", 6, [(1500, 0.07), (3000, 0.3)], status="running", rc=1)   # crashed -> failed
    rows = {r["job"]: r for r in collect_rows(d)}
    assert rows["bias0_s05"]["on_track"] == 1 and rows["bias0_s05"]["first_step_ge_thresh"] == 6000
    assert rows["bias2_s05"]["on_track"] == 0 and rows["bias2_s06"]["status"] == "failed"
    md = sweep_summary(d, "swT")
    assert "| bias0 | 0.0015 | 0.0 | 2 | 0 | 2/2 |" in md
    assert "| bias2 | 0.0015 | 2.0 | 1 | 1 | 0/1 |" in md
    assert "Fisher exact" in md and os.path.exists(os.path.join(d, "sweep.csv"))


def _tiny_cfg(path, bias):
    cfg = {"arch": "gated_linear", "d_model": 32, "n_layers": 2, "n_heads": 4, "d_ff": 64, "chunk_size": 16,
           "short_conv": 4, "legacy_gate_init": False, "decay_bias_init": bias,
           "curriculum": [{"mode": "assoc_recall", "n_bindings": {"min": 2, "max": 4},
                           "seq_len": {"min": 32, "max": 48}, "n_queries": {"min": 1, "max": 2},
                           "distractor_density": 0.1, "query_pos": "uniform", "weight": 1.0}],
           "train": {"lr": 0.0015, "seed": 0, "max_steps": 15000}}
    yaml.safe_dump(cfg, open(path, "w"))


def _git_show(bare, path):
    return subprocess.run(["git", "--git-dir", bare, "show", f"kaggle-results:{path}"],
                          capture_output=True, text=True)


def test_sweep_runs_jobs_in_parallel_publishes_and_resumes(tmp_path):
    bare = str(tmp_path / "remote.git")
    subprocess.run(["git", "init", "--bare", "--quiet", bare], check=True)
    _tiny_cfg(tmp_path / "a.yaml", 0.0)
    _tiny_cfg(tmp_path / "b.yaml", 2.0)
    cmd = [sys.executable, "-m", "scripts.seed_sweep", "--sweep-id", "swT", "--seeds", "3",
           "--cells", f"bias0={tmp_path / 'a.yaml'},bias2={tmp_path / 'b.yaml'}", "--gpus", "cpu,cpu",
           "--stop-at", "20", "--smoke", "--poll", "0.5", "--out", str(tmp_path / "out"),
           "--remote-url", f"file://{bare}"]
    env = {k: v for k, v in os.environ.items() if k != "GH_TOKEN"}
    out = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "2 jobs to run" in out.stdout and "2 at once" in out.stdout
    for job in ("bias0_s03", "bias2_s03"):
        h = json.loads(_git_show(bare, f"runs/swT/{job}_history.json").stdout)
        assert h["status"] == "stopped" and h["last_step"] == 20          # the 60-step smoke schedule, stopped
        meta = json.loads(_git_show(bare, f"runs/swT/{job}_meta.json").stdout)
        assert meta["returncode"] == 0 and meta["seed"] == 3
        assert _git_show(bare, f"runs/swT/logs/{job}.log").returncode == 0
        assert _git_show(bare, f"runs/swT/{job}.safetensors").returncode != 0   # weights never published
    assert "| bias2 |" in _git_show(bare, "runs/swT/SUMMARY.md").stdout
    assert _git_show(bare, "runs/swT/sweep.csv").stdout.count("\n") == 3

    # a fresh disk (new session): everything published is skipped
    cmd[cmd.index("--out") + 1] = str(tmp_path / "out2")
    again = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert again.returncode == 0 and "0 jobs to run (2 already done)" in again.stdout, again.stdout
