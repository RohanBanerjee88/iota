"""Seed sweep: many short GLA runs, several at once, to estimate how often each gate init learns recall.

Why (PROGRESS.md 2026-10-26): with full 15k-step runs (~3 h each) we have 4-5 seeds per gate init, too few to
tell their success rates apart (learned 3/9 vs 7/9, Fisher p ~ 0.15). Two savings make ~20 seeds per init
affordable:

  * train.stop_at keeps the full schedule (lr cosine, curriculum ramp, eval points all follow max_steps) but
    stops at --stop-at (default 6000), so a job is exactly the first 6k steps of the full run. Held-out recall
    >= 0.30 at 6k matched the 15k outcome (>= 0.70 at 32 facts) in 21 of 22 full runs (r27 is the miss).
  * Kaggle gives two T4s and one run uses one: --gpus 0,1 runs two jobs at once (--per-gpu 2: four).

One coordinator process launches the workers (each with its own results folder, CUDA_VISIBLE_DEVICES and
thread budget) and is the ONLY process that publishes -- once after each job -- so no two pushes race.
Everything lands under runs/<sweep-id>/ on the kaggle-results branch: per job `<job>_history.json` (every
eval), `<job>_meta.json` (gpu, seconds, status) and `logs/<job>.log`; plus SUMMARY.md and sweep.csv, rebuilt
on every publish from ALL jobs there (earlier sessions included). Re-running skips jobs already published, so a
session that times out loses only the jobs in progress; --budget-h stops launching jobs that would not finish.
Weights are not kept (not needed for the success-rate question).

    python -m scripts.seed_sweep --sweep-id sw1 --seeds 5-12 --gpus 0,1 --stop-at 6000 --budget-h 11 \\
        --cells bias0=configs/zerow0/sweep_gated_linear.yaml,bias2=configs/bias2/sweep_gated_linear.yaml
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from math import comb
from typing import Dict, List, Optional

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
THRESH = 0.30                     # held-out recall at the stop step that counts as "on track"
DONE = ("stopped", "max_steps", "plateau")
CHECKPOINTS = (1500, 3000, 4500)  # extra steps reported in the summary (the early-start effect)
FIELDS = ["job", "cell", "seed", "lr", "decay_bias_init", "status", "last_step", "assoc_1500", "assoc_3000",
          "assoc_4500", "assoc_last", "on_track", "first_step_ge_thresh", "seconds", "gpu"]


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------
def parse_seeds(spec: str) -> List[int]:
    """'5-9,12' -> [5, 6, 7, 8, 9, 12]"""
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def parse_cells(spec: str) -> Dict[str, str]:
    """'bias0=a.yaml,bias2=b.yaml' -> {'bias0': 'a.yaml', 'bias2': 'b.yaml'} (order kept)"""
    cells = {}
    for part in spec.split(","):
        name, path = part.split("=", 1)
        cells[name.strip()] = path.strip()
    return cells


def job_name(cell: str, seed: int) -> str:
    return f"{cell}_s{seed:02d}"


# ---------------------------------------------------------------------------
# statistics + summary (pure: tested without training anything)
# ---------------------------------------------------------------------------
def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """Two-sided Fisher exact p for [[a, b], [c, d]] (rows = cells, columns = on track / not)."""
    n1, n2, k = a + b, c + d, a + c
    n = n1 + n2
    if n == 0:
        return 1.0

    def p(x):
        return comb(k, x) * comb(n - k, n1 - x) / comb(n, n1)
    obs = p(a)
    return min(1.0, sum(p(x) for x in range(max(0, k - n2), min(k, n1) + 1) if p(x) <= obs * (1 + 1e-9)))


def _assoc(e: Dict) -> float:
    return float(e.get("by_mode", {}).get("assoc_recall", e.get("per_query", 0.0)))


def job_row(meta: Dict, hist: Optional[Dict]) -> Dict:
    evals = (hist or {}).get("eval_acc", [])
    at = {e["step"]: _assoc(e) for e in evals}
    last = _assoc(evals[-1]) if evals else None
    first = next((e["step"] for e in evals if _assoc(e) >= THRESH), "")
    status = (hist or {}).get("status", meta.get("status", "failed")) if meta.get("returncode", 1) == 0 else "failed"
    r = {"job": meta["job"], "cell": meta["cell"], "seed": meta["seed"], "lr": meta.get("lr", ""),
         "decay_bias_init": meta.get("decay_bias_init", ""), "status": status,
         "last_step": evals[-1]["step"] if evals else 0,
         "assoc_last": round(last, 4) if last is not None else "",
         "on_track": int(last is not None and status in DONE and last >= THRESH),
         "first_step_ge_thresh": first, "seconds": meta.get("seconds", ""), "gpu": meta.get("gpu", "")}
    for s in CHECKPOINTS:
        r[f"assoc_{s}"] = round(at[s], 4) if s in at else ""
    return r


def collect_rows(run_dir: str) -> List[Dict]:
    rows = []
    for mp in sorted(glob.glob(os.path.join(run_dir, "*_meta.json"))):
        meta = json.load(open(mp))
        hp = os.path.join(run_dir, f"{meta['job']}_history.json")
        rows.append(job_row(meta, json.load(open(hp)) if os.path.exists(hp) else None))
    return rows


def _mean(xs):
    xs = [x for x in xs if x != ""]
    return f"{sum(xs) / len(xs):.3f}" if xs else "–"


def sweep_summary(run_dir: str, run_id: str) -> str:
    """SUMMARY.md for a sweep folder (also writes sweep.csv there). Used as sync(summary_fn=...)."""
    rows = collect_rows(run_dir)
    with open(os.path.join(run_dir, "sweep.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    env = json.load(open(os.path.join(run_dir, "env.json"))) if os.path.exists(os.path.join(run_dir, "env.json")) else {}
    cells = list(dict.fromkeys(r["cell"] for r in rows))
    lines = [f"# iota seed sweep `{run_id}`", "",
             f"- **updated:** {env.get('updated', '?')} (last: `{env.get('stage', '?')}`)",
             f"- **code:** `{env.get('commit', '?')}` on `{env.get('branch', '?')}` · **device:** {env.get('device', '?')}",
             f"- **on track** = held-out recall (assoc, full difficulty) ≥ {THRESH} at the stop step. It matched the",
             "  15k outcome in 21 of 22 full runs (PROGRESS.md 2026-10-26). Jobs are the full schedule stopped early.",
             "", "## Per cell", "",
             "| cell | lr | gate bias | jobs done | failed | on track | mean @1.5k | mean @3k | mean @last |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    counts = {}
    for c in cells:
        rs = [r for r in rows if r["cell"] == c]
        done = [r for r in rs if r["status"] in DONE]
        on = sum(r["on_track"] for r in done)
        counts[c] = (on, len(done) - on)
        lines.append(f"| {c} | {rs[0]['lr']} | {rs[0]['decay_bias_init']} | {len(done)} | {len(rs) - len(done)} | "
                     f"{on}/{len(done)} | {_mean([r['assoc_1500'] for r in done])} | "
                     f"{_mean([r['assoc_3000'] for r in done])} | {_mean([r['assoc_last'] for r in done])} |")
    if len(cells) == 2:
        (a, b), (c_, d) = counts[cells[0]], counts[cells[1]]
        lines += ["", f"Fisher exact (two-sided), on track {cells[0]} vs {cells[1]}: "
                      f"**p = {fisher_exact(a, b, c_, d):.3f}** ({a}/{a + b} vs {c_}/{c_ + d})."]
    lines += ["", "## Jobs", "",
              "| job | status | last step | @1.5k | @3k | @4.5k | @last | on track | first step ≥ thresh | minutes | gpu |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in sorted(rows, key=lambda r: (r["seed"], r["cell"])):
        mins = f"{r['seconds'] / 60:.0f}" if r["seconds"] != "" else ""
        lines.append(f"| {r['job']} | {r['status']} | {r['last_step']} | {r['assoc_1500']} | {r['assoc_3000']} | "
                     f"{r['assoc_4500']} | {r['assoc_last']} | {r['on_track']} | {r['first_step_ge_thresh']} | "
                     f"{mins} | {r['gpu']} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# what is already published
# ---------------------------------------------------------------------------
def published_jobs(sweep_id: str, remote_url: Optional[str], token: Optional[str]) -> set:
    """Jobs with a finished meta file under runs/<sweep_id>/ on the results branch (empty if unreachable)."""
    from scripts.sync_results import DEFAULT_BRANCH, DEFAULT_REPO, _prepare_checkout
    if not remote_url and not token:
        return set()
    url = remote_url or f"https://x-access-token:{token}@github.com/{DEFAULT_REPO}.git"
    tmp = tempfile.mkdtemp(prefix="iota-sweep-")
    try:
        _prepare_checkout(url, DEFAULT_BRANCH, tmp, token)
        rows = collect_rows(os.path.join(tmp, "runs", sweep_id))
        return {r["job"] for r in rows if r["status"] in DONE}
    except Exception as e:
        print(f"  [sweep] could not read published jobs ({type(e).__name__}); assuming none", flush=True)
        return set()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------
def worker(job_path: str) -> int:
    import iota.train as tr
    job = json.load(open(job_path))
    tr.RESULTS_DIR = job["results_dir"]
    tr.train_sweep(job["cfg"], smoke=job["smoke"])
    return 0


# ---------------------------------------------------------------------------
# coordinator
# ---------------------------------------------------------------------------
def _slots(gpus: str, per_gpu: int) -> List[str]:
    ids = [g.strip() for g in gpus.split(",") if g.strip()]
    gpu_ids = [g for g in ids if g != "cpu"]
    if gpu_ids:
        import torch
        n = torch.cuda.device_count()
        missing = [g for g in gpu_ids if int(g) >= n]
        if missing:
            print(f"!! [sweep] GPU(s) {missing} not visible ({n} found) -- dropped. On Kaggle pick "
                  f"'GPU T4 x2' to run two jobs at once.", flush=True)
        ids = [g for g in ids if g == "cpu" or int(g) < n]
    if not ids:
        raise SystemExit("no usable device in --gpus")
    return [g for g in ids for _ in range(per_gpu)]


def _job_cfg(path: str, seed: int, stop_at: int, run_name: str, device: str) -> Dict:
    cfg = yaml.safe_load(open(path))
    cfg["train"] = dict(cfg.get("train", {}), seed=seed, stop_at=stop_at, run_name=run_name, device=device)
    return cfg


def coordinate(args) -> int:
    from scripts.sync_results import sync
    cells = parse_cells(args.cells)
    seeds = parse_seeds(args.seeds)
    root = os.path.abspath(os.path.join(args.out, args.sweep_id))
    pub = os.path.join(root, "pub")
    os.makedirs(os.path.join(pub, "logs"), exist_ok=True)
    token = os.environ.get("GH_TOKEN")
    remote = args.remote_url or os.environ.get("IOTA_GH_REMOTE_URL")
    publish = not args.no_gh and bool(remote or token)
    if not args.no_gh and not publish:
        print("NOTE: GH_TOKEN not set -> sweep results will NOT be published", flush=True)

    done = published_jobs(args.sweep_id, remote, token) if publish else set()
    done |= {r["job"] for r in collect_rows(pub) if r["status"] in DONE}
    queue = [(c, s) for s in seeds for c in cells if job_name(c, s) not in done]   # cells interleaved per seed
    slots = _slots(args.gpus, args.per_gpu)
    threads = max(1, (os.cpu_count() or 2) // len(slots))
    print(f"[sweep] {args.sweep_id}: {len(queue)} jobs to run ({len(done)} already done), "
          f"{len(slots)} at once on {slots}, stop at {args.stop_at}, {threads} CPU threads each", flush=True)

    t_start, durations = time.time(), []
    running: Dict[int, tuple] = {}                    # slot index -> (job, cell, seed, proc, t0, log fh, cfg)
    stopped_for_budget = False
    while queue or running:
        for i, dev in enumerate(slots):
            if i in running or not queue:
                continue
            est = (sum(durations) / len(durations) if durations else args.est_min * 60) * 1.1
            if args.budget_h and time.time() - t_start + est > args.budget_h * 3600:
                if not stopped_for_budget:
                    print(f"[sweep] budget: {len(queue)} job(s) not started (would end past "
                          f"{args.budget_h} h); re-run to continue", flush=True)
                stopped_for_budget = True
                queue = []
                break
            cell, seed = queue.pop(0)
            name = job_name(cell, seed)
            work = os.path.join(root, "work", name)
            os.makedirs(work, exist_ok=True)
            cfg = _job_cfg(cells[cell], seed, args.stop_at, name, "cpu" if dev == "cpu" else "cuda")
            jp = os.path.join(work, "job.json")
            json.dump({"cfg": cfg, "results_dir": work, "smoke": args.smoke}, open(jp, "w"))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="" if dev == "cpu" else dev,
                       OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")
            fh = open(os.path.join(pub, "logs", f"{name}.log"), "w")
            proc = subprocess.Popen([sys.executable, "-u", "-m", "scripts.seed_sweep", "--worker", jp],
                                    cwd=REPO_ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT)
            running[i] = (name, cell, seed, proc, time.time(), fh, cfg)
            print(f"[sweep] start {name} on {dev}", flush=True)
        finished = [i for i, r in running.items() if r[3].poll() is not None]
        if not finished:
            time.sleep(args.poll)
            continue
        for i in finished:
            name, cell, seed, proc, t0, fh, cfg = running.pop(i)
            fh.close()
            secs = round(time.time() - t0)
            work = os.path.join(root, "work", name)
            for f in (f"{name}_history.json", f"{name}.json"):
                if os.path.exists(os.path.join(work, f)):
                    shutil.copy2(os.path.join(work, f), os.path.join(pub, f))
            hp = os.path.join(pub, f"{name}_history.json")
            hist = json.load(open(hp)) if os.path.exists(hp) else {}
            status = hist.get("status", "failed") if proc.returncode == 0 else "failed"
            meta = {"job": name, "cell": cell, "seed": seed, "gpu": slots[i], "seconds": secs,
                    "returncode": proc.returncode, "status": status, "stop_at": args.stop_at,
                    "lr": cfg["train"].get("lr"), "decay_bias_init": cfg.get("decay_bias_init"),
                    "config": cells[cell]}
            json.dump(meta, open(os.path.join(pub, f"{name}_meta.json"), "w"), indent=1)
            if status in DONE:
                durations.append(secs)
            row = job_row(meta, hist)
            print(f"[sweep] done  {name}: {status}, held-out recall {row['assoc_last']} at step "
                  f"{row['last_step']} ({secs / 60:.0f} min)", flush=True)
            if proc.returncode != 0:
                tail = open(os.path.join(pub, "logs", f"{name}.log")).read().splitlines()[-15:]
                print("\n".join(f"    | {line}" for line in tail), flush=True)
            if publish:
                sync(pub, args.sweep_id, stage=f"sweep {name} {status}", remote_url=remote, token=token,
                     summary_fn=sweep_summary)
    print(f"[sweep] finished in {(time.time() - t_start) / 3600:.2f} h", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="many short GLA runs in parallel (gate-init success rates)")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--sweep-id", default="sw1", help="folder under runs/ on the kaggle-results branch")
    ap.add_argument("--cells", default="bias0=configs/zerow0/sweep_gated_linear.yaml,"
                                       "bias2=configs/bias2/sweep_gated_linear.yaml")
    ap.add_argument("--seeds", default="5-12")
    ap.add_argument("--stop-at", type=int, default=6000)
    ap.add_argument("--gpus", default="0,1", help="device ids, or 'cpu' (repeat for several CPU workers)")
    ap.add_argument("--per-gpu", type=int, default=1, help="concurrent jobs per device")
    ap.add_argument("--budget-h", type=float, default=0.0, help="don't start a job that would end after this")
    ap.add_argument("--est-min", type=float, default=75.0, help="job length guess before any job finishes")
    ap.add_argument("--out", default="experiments/sweep")
    ap.add_argument("--poll", type=float, default=15.0)
    ap.add_argument("--smoke", action="store_true", help="tiny jobs (60-step schedule), for tests")
    ap.add_argument("--no-gh", action="store_true")
    ap.add_argument("--remote-url", default=None, help="override the push URL (tests)")
    args = ap.parse_args()
    if args.worker:
        return worker(args.worker)
    return coordinate(args)


if __name__ == "__main__":
    raise SystemExit(main())
