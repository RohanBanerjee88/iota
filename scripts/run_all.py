"""End-to-end, RESUMABLE driver for the Phase-6 sweep (run this on Kaggle).

One command per stage; every stage is safe to re-run. Work that is already done is
skipped, so a Kaggle session timeout costs you only the in-progress model -- never
the whole run.

    python -m scripts.run_all --stage tune     --repo <user>/iota-sweep   # lr probe
    python -m scripts.run_all --stage train    --repo <user>/iota-sweep
    python -m scripts.run_all --stage eval     --passes 1,3
    python -m scripts.run_all --stage eval     --passes 2
    python -m scripts.run_all --stage profile
    python -m scripts.run_all --stage plot
    python -m scripts.run_all --stage all      --repo <user>/iota-sweep
    python -m scripts.run_all --stage all      --smoke     # whole pipeline in minutes

Ablations select another config folder with IOTA_CONFIG_DIR (default `configs`), e.g.
    IOTA_CONFIG_DIR=configs/fixinit python -m scripts.run_all --stage train \
        --repo <user>/iota-r07 --reuse transformer=<user>/iota-r04

Every stage tees its output to experiments/results/logs/<stage>.log and, when
GH_TOKEN is set, publishes the run's small artifacts (CSVs, run jsons, logs, the
figure, a SUMMARY.md) to the `kaggle-results` branch under runs/<run_id>/ -- that
is how results get from Kaggle back to the repo (see scripts/sync_results.py).

--smoke trains each model for 60 steps, evaluates every pass at n=8, profiles and
plots, all into experiments/smoke/ with HF traffic off. It exists to catch a
broken pipeline, GPU issue or bad GitHub token in ~5 minutes instead of 6 hours
in. Its numbers are meaningless.

Resume logic (the part that answers "do I have to run everything again?"):
a checkpoint counts as DONE only if all three hold --
  1. the weights exist locally (or can be pulled from the HF Hub),
  2. its SAVED curriculum matches the CURRENT config, and
  3. its history shows real training (not a 60-step --smoke run).
Condition 2 matters: the checkpoints already on the Hub were trained at the old
seq_len<=256 / n_bindings<=8 ceilings that this experiment deliberately replaced.
Silently reusing them would reproduce the broken sweep, so they are detected as
STALE and retrained.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import yaml

RESULTS_DIR = "experiments/results"
SMOKE_DIR = "experiments/smoke"
# short -> long: results bank early, and the slow learner (most timeout-prone)
# runs last, once the others are already safe on the Hub.
ARCH_ORDER = ["hybrid", "transformer", "gated_linear"]
MIN_REAL_STEPS = 1000  # below this it's a smoke run, not a trained model


# ---------------------------------------------------------------------------
# checkpoint bookkeeping
# ---------------------------------------------------------------------------
def _cfg_path(arch: str) -> str:
    from iota.util import sweep_config_path
    return sweep_config_path(arch)


def _archs() -> list:
    """Architectures that have a config in the active config folder ($IOTA_CONFIG_DIR).

    An ablation folder may hold a subset (e.g. only sweep_gated_linear.yaml for a
    GLA-only seed replicate); every stage then touches only those architectures."""
    return [a for a in ARCH_ORDER if os.path.exists(_cfg_path(a))]


def _run_name(cfg: dict, arch: str) -> str:
    return cfg.get("train", {}).get("run_name", f"{arch}_sweep")


def _fingerprint(cfg: dict) -> dict:
    """The parts of a config that define WHAT the model is and what it learned.

    Every model key (arch, sizes, short_conv, ...) + the curriculum. Deliberately
    excludes the `train` block (lr / max_steps / patience): re-tuning the optimizer
    shouldn't invalidate a finished checkpoint, but changing the architecture or the
    task distribution must. (vocab_size is filled in at train time, so it's ignored.)
    """
    fp = {k: v for k, v in cfg.items() if k not in ("train", "vocab_size")}
    # legacy_gate_init absent == True (every run before the key existed used the
    # overwritten gate init), so old checkpoints match configs that state it.
    fp.setdefault("legacy_gate_init", True)
    return fp


def checkpoint_status(arch: str) -> tuple:
    """-> (status, run_name, detail) where status is 'ok' | 'stale' | 'missing'."""
    cfg = yaml.safe_load(open(_cfg_path(arch)))
    run_name = _run_name(cfg, arch)
    wpath = os.path.join(RESULTS_DIR, f"{run_name}.safetensors")
    jpath = os.path.join(RESULTS_DIR, f"{run_name}.json")
    if not os.path.exists(wpath):
        return "missing", run_name, "no weights"
    if not os.path.exists(jpath):
        return "stale", run_name, "weights present but no run json to verify against"
    try:
        run = json.load(open(jpath))
    except Exception as e:
        return "stale", run_name, f"unreadable run json ({e})"
    if _fingerprint(run.get("config", {})) != _fingerprint(cfg):
        return "stale", run_name, "trained on a DIFFERENT curriculum than the current config"
    evals = run.get("history", {}).get("eval_acc", [])
    last_step = evals[-1].get("step", 0) if evals else 0
    if last_step < MIN_REAL_STEPS and RESULTS_DIR != SMOKE_DIR:
        return "stale", run_name, f"only {last_step} steps (smoke run)"
    return "ok", run_name, f"trained {last_step} steps"


def _by_mode(run_name: str) -> str:
    """Final per-mode accuracy from a run json, for the summary table."""
    jpath = os.path.join(RESULTS_DIR, f"{run_name}.json")
    if not os.path.exists(jpath):
        return "(no run json)"
    try:
        evals = json.load(open(jpath)).get("history", {}).get("eval_acc", [])
    except Exception:
        return "(unreadable)"
    if not evals:
        return "(no evals)"
    last = evals[-1]
    bm = last.get("by_mode", {})
    if not bm:
        return f"step {last.get('step')}: per_query={last.get('per_query')}"
    return (f"step {last.get('step')}: "
            + " ".join(f"{m}={v:.3f}" for m, v in sorted(bm.items()))
            + f"  (balanced {last.get('balanced')})")


# ---------------------------------------------------------------------------
# HF hub (backup + recovery)
# ---------------------------------------------------------------------------
def _hf_api(token):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("  [hf] huggingface_hub not installed -> no backup/recovery", flush=True)
        return None
    if not token:
        print("  [hf] no HF_TOKEN -> no backup/recovery", flush=True)
        return None
    return HfApi(token=token)


def hf_pull(run_name: str, repo: str, token) -> bool:
    """Try to fetch this run's artifacts from the Hub. True if weights landed."""
    api = _hf_api(token)
    if api is None:
        return False
    from huggingface_hub import hf_hub_download
    got = False
    for ext in ("safetensors", "json"):
        try:
            p = hf_hub_download(repo_id=repo, filename=f"{run_name}.{ext}",
                                repo_type="model", token=token,
                                local_dir=RESULTS_DIR)
            print(f"  [hf] pulled {os.path.basename(p)}", flush=True)
            got = got or ext == "safetensors"
        except Exception:
            pass  # absent on the Hub is a normal outcome, not an error
    return got


def hf_push(run_name: str, repo: str, token) -> None:
    api = _hf_api(token)
    if api is None:
        return
    from huggingface_hub import create_repo
    create_repo(repo, repo_type="model", exist_ok=True, token=token)
    for ext in ("safetensors", "json"):
        p = os.path.join(RESULTS_DIR, f"{run_name}.{ext}")
        if os.path.exists(p):
            api.upload_file(path_or_fileobj=p, path_in_repo=f"{run_name}.{ext}",
                            repo_id=repo, repo_type="model", token=token)
            print(f"  [hf] pushed {run_name}.{ext}", flush=True)


def hf_pull_results(repo: str, token) -> None:
    """Fetch result CSVs that exist on the Hub but not locally.

    Kaggle wipes the disk between sessions, so the plot stage in Session B would
    otherwise only see Session B's own CSVs and silently drop Session A's panels.
    Local files always win -- a CSV is only downloaded when it is absent here.
    """
    api = _hf_api(token)
    if api is None:
        return
    from huggingface_hub import hf_hub_download
    try:
        remote = api.list_repo_files(repo, repo_type="model")
    except Exception as e:
        print(f"  [hf] cannot list {repo} ({e})", flush=True)
        return
    for f in remote:
        if f.endswith(".csv") and "/" not in f and not os.path.exists(os.path.join(RESULTS_DIR, f)):
            try:
                hf_hub_download(repo_id=repo, filename=f, repo_type="model", token=token,
                                local_dir=RESULTS_DIR)
                print(f"  [hf] pulled {f}", flush=True)
            except Exception as e:
                print(f"  [hf] could not pull {f} ({e})", flush=True)


def hf_push_results(repo: str, token) -> None:
    """Push CSVs / figures produced by the eval, profile and plot stages."""
    api = _hf_api(token)
    if api is None:
        return
    from huggingface_hub import create_repo
    create_repo(repo, repo_type="model", exist_ok=True, token=token)
    for f in sorted(os.listdir(RESULTS_DIR)):
        if f.endswith((".csv", ".png", ".pdf")):
            api.upload_file(path_or_fileobj=os.path.join(RESULTS_DIR, f),
                            path_in_repo=f, repo_id=repo, repo_type="model", token=token)
            print(f"  [hf] pushed {f}", flush=True)


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------
def stage_train(args) -> None:
    from iota.train import train

    archs = [args.only] if args.only else _archs()
    print(f"\n### STAGE train  ({', '.join(archs)})\n", flush=True)
    reuse = dict(r.split("=", 1) for r in args.reuse.split(",") if r.strip()) if args.reuse else {}
    for arch in archs:
        status, run_name, detail = checkpoint_status(arch)
        if status == "missing" and not args.no_hf:
            print(f"[{arch}] no local checkpoint -> trying the Hub", flush=True)
            if hf_pull(run_name, args.repo, args.token):
                status, run_name, detail = checkpoint_status(arch)
        if status == "missing" and arch in reuse and not args.no_hf:
            # An ablation that leaves this arch untouched reuses an earlier run's
            # checkpoint -- but only if it is EXACTLY the current config (fingerprint).
            print(f"[{arch}] reusing the checkpoint from {reuse[arch]}", flush=True)
            if hf_pull(run_name, reuse[arch], args.token):
                status, run_name, detail = checkpoint_status(arch)
                if status == "ok":
                    hf_push(run_name, args.repo, args.token)  # later sessions pull it from --repo
                else:
                    print(f"!! [{arch}] {reuse[arch]} checkpoint is {status} ({detail}) -- "
                          f"NOT reusing it; training from scratch", flush=True)
                    for ext in ("safetensors", "json"):
                        p = os.path.join(RESULTS_DIR, f"{run_name}.{ext}")
                        if os.path.exists(p):
                            os.remove(p)
                    status, run_name, detail = checkpoint_status(arch)
        if status == "ok" and not args.force:
            print(f"[{arch}] SKIP -- {detail}", flush=True)
            continue
        if status == "stale":
            print(f"[{arch}] RETRAIN -- {detail}", flush=True)
        print(f"\n{'='*72}\n=== TRAIN {arch} -> {run_name}\n{'='*72}", flush=True)
        cfg = yaml.safe_load(open(_cfg_path(arch)))
        try:
            out = train(cfg, smoke=args.smoke)
        except Exception as e:
            print(f"!! {arch} FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        print(f"=== {arch} done, best balanced acc = {out.get('best_acc'):.4f}", flush=True)
        if not args.no_hf:
            try:
                hf_push(run_name, args.repo, args.token)
            except Exception as e:
                print(f"  [hf] push failed ({e}); weights still local", flush=True)
        _publish(args, f"train:{arch}")  # visible on GitHub even if a later model dies

    print(f"\n{'='*72}\n=== TRAIN SUMMARY\n{'='*72}", flush=True)
    for arch in archs:
        status, run_name, detail = checkpoint_status(arch)
        print(f"  {arch:14s} [{status}] {_by_mode(run_name)}", flush=True)
    print("\nHealthy = assoc high AND state well above chance (~0.01).\n"
          "If state is still ~0.01 the control never learned -- stop and fix that\n"
          "before trusting any figure.", flush=True)


TUNE_ORDER = ["transformer", "hybrid", "gated_linear"]  # the baseline that broke in r01 first
TUNE_FIELDS = ["arch", "lr", "grad_clip", "steps", "best_step", "best_balanced",
               "assoc", "state", "exact", "final_balanced", "seconds"]


def stage_tune(args) -> None:
    """Short, identical lr probe for every architecture (the fairness protocol).

    Each (arch, lr) trains on the sweep curriculum for a short fixed budget --
    same steps, schedule shape, grad_clip and eval for all -- and records its best
    per-mode score. Pick each arch's lr from tune.csv, then run the real training.
    Resumable: probes already in tune.csv are skipped.
    """
    import csv
    import time

    from iota.train import train

    lrs = [float(x) for x in args.tune_lrs.split(",") if x.strip()]
    if args.smoke:
        lrs = lrs[:1]
    path = os.path.join(RESULTS_DIR, "tune.csv")
    done = set()
    if os.path.exists(path) and not args.smoke:
        with open(path) as fh:
            done = {(r["arch"], float(r["lr"])) for r in csv.DictReader(fh)}
    archs = [args.only] if args.only else [a for a in TUNE_ORDER if a in _archs()]
    print(f"\n### STAGE tune  ({', '.join(archs)} x lr {lrs}, {args.tune_steps} steps each)\n",
          flush=True)
    if args.smoke and os.path.exists(path):
        os.remove(path)
    for arch in archs:
        for lr in lrs:
            if (arch, lr) in done:
                print(f"[tune] SKIP {arch} lr={lr} (already in tune.csv)", flush=True)
                continue
            cfg = yaml.safe_load(open(_cfg_path(arch)))
            steps = args.tune_steps
            cfg["train"].update({
                "lr": lr, "grad_clip": 1.0, "max_steps": steps,
                "easy_steps": steps // 4, "ramp_steps": steps // 2,
                "eval_every": max(1, steps // 8), "eval_n": 256, "patience": 10 ** 6,
                "run_name": f"tune_{arch}_lr{lr:g}",
            })
            print(f"\n=== TUNE {arch} lr={lr:g}", flush=True)
            t0 = time.time()
            try:
                out = train(cfg, smoke=args.smoke)
            except Exception as e:
                print(f"!! tune {arch} lr={lr:g} FAILED: {type(e).__name__}: {e}", flush=True)
                continue
            evals = out["history"]["eval_acc"]
            best = max(evals, key=lambda e: e.get("balanced", 0)) if evals else {}
            bm = best.get("by_mode", {})
            row = {"arch": arch, "lr": lr, "grad_clip": 1.0, "steps": steps,
                   "best_step": best.get("step"), "best_balanced": best.get("balanced"),
                   "assoc": bm.get("assoc_recall"), "state": bm.get("state_track"),
                   "exact": round(best.get("exact", 0), 4),
                   "final_balanced": evals[-1].get("balanced") if evals else None,
                   "seconds": round(time.time() - t0)}
            new = not os.path.exists(path)
            with open(path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=TUNE_FIELDS)
                if new:
                    w.writeheader()
                w.writerow(row)
            print(f"[tune] {arch} lr={lr:g}: best balanced {row['best_balanced']} "
                  f"(assoc {row['assoc']}, state {row['state']}) in {row['seconds']}s", flush=True)
            _publish(args, f"tune:{arch}:{lr:g}")

    if os.path.exists(path):
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        print(f"\n{'='*72}\n=== TUNE SUMMARY (best lr per arch by balanced score)\n{'='*72}",
              flush=True)
        for arch in archs:
            mine = [r for r in rows if r["arch"] == arch and r["best_balanced"] not in ("", "None")]
            if mine:
                b = max(mine, key=lambda r: float(r["best_balanced"]))
                print(f"  {arch:14s} lr={float(b['lr']):g}  balanced {b['best_balanced']}  "
                      f"assoc {b['assoc']}  state {b['state']}", flush=True)


def _ensure_checkpoints(args) -> None:
    """Pull any checkpoint that isn't usable locally from the Hub (Session B)."""
    for arch in _archs():
        status, run_name, _ = checkpoint_status(arch)
        if status != "ok" and not args.no_hf:
            hf_pull(run_name, args.repo, args.token)


def stage_sanity(args) -> None:
    """In-distribution, per-mode check: the gate before trusting ANY sweep number."""
    from scripts.sanity_indist import run_sanity

    print("\n### STAGE sanity (each model on its own training distribution)\n", flush=True)
    _ensure_checkpoints(args)
    archs = [a for a in _archs() if checkpoint_status(a)[0] == "ok"]
    run_sanity(n=8 if args.smoke else 500, archs=archs, minibatch=args.minibatch,
               device=args.device, results_dir=RESULTS_DIR,
               out_csv=os.path.join(RESULTS_DIR, "sanity_indist.csv"))


def stage_eval(args) -> None:
    import torch

    from iota.eval import load_checkpoint, run_pass

    device = args.device
    passes = [int(p) for p in args.passes.split(",") if p.strip()]
    print(f"\n### STAGE eval  (passes {passes}, device={device})\n", flush=True)

    models = {}
    for arch in _archs():
        status, run_name, detail = checkpoint_status(arch)
        if status != "ok":
            if not args.no_hf and hf_pull(run_name, args.repo, args.token):
                status, run_name, detail = checkpoint_status(arch)
        if status != "ok":
            print(f"!! {arch}: checkpoint {status} ({detail}) -- run --stage train first",
                  flush=True)
            continue
        models[run_name] = load_checkpoint(run_name, results_dir=RESULTS_DIR,
                                           device=device)[0]
        print(f"  loaded {run_name} ({detail})", flush=True)
    if not models:
        print("no usable checkpoints; nothing to evaluate", flush=True)
        return
    if len(models) < len(_archs()):
        print(f"WARNING: evaluating {len(models)}/{len(_archs())} models -- the "
              f"comparison is incomplete", flush=True)

    names = {1: "pass1_capacity", 2: "pass2_length", 3: "pass3_control", 4: "pass4_keylen",
             5: "pass5_grid", 6: "pass6_freerun", 7: "pass7_history", 8: "pass8_gates",
             9: "pass9_clamp"}
    for p in passes:
        out_csv = os.path.join(RESULTS_DIR, f"{names.get(p, f'pass{p}')}.csv")
        print(f"\n--- pass {p} -> {out_csv}", flush=True)
        if p == 4:  # diagnostic: recall split by key digit count (same prompts as pass 1)
            from iota.eval import run_keylen_pass
            run_keylen_pass(models, n=args.n, device=device, out_csv=out_csv,
                            minibatch=args.minibatch)
        elif p == 7:  # matched histories: does answering earlier questions change later recall?
            from iota.history import run_history_pass
            run_history_pass(models, n=args.n, device=device, out_csv=out_csv,
                             minibatch=args.minibatch)
        elif p == 9:  # gate-clamp intervention: which layers' forgetting recall needs
            from iota.clamp import run_clamp_pass
            run_clamp_pass(models, n=args.n, device=device, out_csv=out_csv,
                           minibatch=args.minibatch)
        elif p == 8:  # gate statistics: what the GLA decay gates do, per token role
            from iota.gates import run_gate_pass
            run_gate_pass(models, n=args.n, device=device, out_csv=out_csv,
                          minibatch=args.minibatch)
        elif p == 6:  # free-running vs teacher-forced on pass 1's prompts
            from iota.eval import run_freerun_pass
            run_freerun_pass(models, n=args.n, device=device, out_csv=out_csv,
                             minibatch=args.minibatch)
        else:
            run_pass(p, models, n=args.n, device=device, out_csv=out_csv,
                     minibatch=args.minibatch)
        if not args.no_hf:
            try:
                hf_push_results(args.repo, args.token)
            except Exception as e:
                print(f"  [hf] results push failed ({e})", flush=True)


def stage_profile(args) -> None:
    from iota.profile import run_decode_profile, run_profile

    print("\n### STAGE profile\n", flush=True)
    run_profile(out_csv=os.path.join(RESULTS_DIR, "cost_profile.csv"),
                device=args.device, archs=_archs())
    # decode: the cost axis that matters for linear attention (KV cache vs state)
    lens = (128, 512, 2048) if args.smoke else None
    kw = {"context_lens": lens} if lens else {}
    run_decode_profile(out_csv=os.path.join(RESULTS_DIR, "decode_profile.csv"),
                       device=args.device, archs=_archs(), **kw)
    if not args.no_hf:
        try:
            hf_push_results(args.repo, args.token)
        except Exception as e:
            print(f"  [hf] results push failed ({e})", flush=True)


def stage_plot(args) -> None:
    from iota.plot import make_figure

    print("\n### STAGE plot\n", flush=True)
    if not args.no_hf:
        hf_pull_results(args.repo, args.token)  # Session A's CSVs, if this is Session B
    make_figure(results_dir=RESULTS_DIR)
    if not args.no_hf:
        try:
            hf_push_results(args.repo, args.token)
        except Exception as e:
            print(f"  [hf] results push failed ({e})", flush=True)


# ---------------------------------------------------------------------------
# logging + publishing
# ---------------------------------------------------------------------------
class _Tee:
    """Write to the console AND a log file, so every stage leaves a full transcript."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)
        return len(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def __getattr__(self, name):  # isatty, encoding, ... -> the real console
        return getattr(self.stream, name)


def _run_logged(log_name: str, fn, args) -> None:
    os.makedirs(os.path.join(RESULTS_DIR, "logs"), exist_ok=True)
    path = os.path.join(RESULTS_DIR, "logs", f"{log_name}.log")
    with open(path, "a", buffering=1) as fh:
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        fh.write(f"\n##### {log_name} started {stamp} argv={' '.join(sys.argv[1:])}\n")
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(out, fh), _Tee(err, fh)
        try:
            fn(args)
        except Exception:
            import traceback
            traceback.print_exc()  # lands in the log too, then re-raise
            raise
        finally:
            sys.stdout, sys.stderr = out, err


def _publish(args, stage: str) -> bool:
    if args.no_gh:
        return False
    from scripts.sync_results import sync
    # IOTA_GH_REMOTE_URL redirects the push (e.g. to a local bare repo) for testing.
    return sync(RESULTS_DIR, args.run_id, stage=stage, repo=args.gh_repo,
                remote_url=os.environ.get("IOTA_GH_REMOTE_URL"))


def main() -> int:
    global RESULTS_DIR
    import torch

    ap = argparse.ArgumentParser(description="iota Phase-6 end-to-end driver")
    ap.add_argument("--stage", default="all",
                    choices=["all", "tune", "train", "sanity", "eval", "profile", "plot", "status"])
    ap.add_argument("--only", choices=ARCH_ORDER, help="train/tune just one architecture")
    ap.add_argument("--passes", default="1,3", help="eval passes, e.g. '1,3' or '2,4' (4 = key-length diagnostic, 5 = joint load x distance grid, 6 = free-running, 7 = matched histories, 8 = gate statistics, 9 = gate-clamp intervention)")
    ap.add_argument("--repo", default="BanerjeeRohan44/iota-sweep")
    ap.add_argument("--no-hf", action="store_true", help="skip all Hub traffic")
    ap.add_argument("--force", action="store_true", help="retrain even if a checkpoint is ok")
    ap.add_argument("--reuse", default="",
                    help="train: ARCH=HF_REPO[,...] -- take an unchanged arch's checkpoint from an "
                         "earlier run (must match the current config) instead of retraining it")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--n", type=int, default=1000, help="examples per eval cell")
    ap.add_argument("--minibatch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--smoke", action="store_true",
                    help="dry-run the whole pipeline in minutes (experiments/smoke/, no HF)")
    ap.add_argument("--run-id", default=os.environ.get("IOTA_RUN_ID", "adhoc"),
                    help="folder name under runs/ on the kaggle-results branch")
    ap.add_argument("--gh-repo", default="RohanBanerjee88/iota")
    ap.add_argument("--no-gh", action="store_true", help="don't publish to GitHub")
    ap.add_argument("--tune-lrs", default="0.00075,0.0015",
                    help="lr grid for --stage tune (same grid for every arch)")
    ap.add_argument("--tune-steps", type=int, default=4000)
    args = ap.parse_args()

    if args.smoke:
        import iota.train
        RESULTS_DIR = SMOKE_DIR
        iota.train.RESULTS_DIR = SMOKE_DIR  # smoke checkpoints never touch real ones
        args.no_hf, args.force = True, True
        args.n, args.minibatch = 8, 8
        args.run_id = f"{args.run_id}-smoke"
    if not args.no_gh and not os.environ.get("GH_TOKEN"):
        print("NOTE: GH_TOKEN not set -> results will NOT be published to GitHub", flush=True)
        args.no_gh = True

    print(f"iota run_all: stage={args.stage} device={args.device} "
          f"hf={'off' if args.no_hf else args.repo} "
          f"github={'off' if args.no_gh else f'{args.gh_repo}@kaggle-results/runs/{args.run_id}'}"
          f"{'  [SMOKE -> ' + SMOKE_DIR + ']' if args.smoke else ''}", flush=True)

    if args.stage == "status":
        for arch in _archs():
            status, run_name, detail = checkpoint_status(arch)
            print(f"  {arch:14s} [{status:7s}] {detail}", flush=True)
        return 0

    published = []

    def stage(log_name, fn, label):
        # Publish in `finally`: a stage that crashes still ships its log (with the
        # traceback) to GitHub, which is exactly when it is needed most.
        try:
            _run_logged(log_name, fn, args)
        finally:
            published.append(_publish(args, label))

    # tune is its own stage; the smoke run exercises it too so it can't rot.
    if args.stage == "tune" or (args.stage == "all" and args.smoke):
        stage("tune", stage_tune, "tune")
    if args.stage in ("all", "train"):
        stage("train", stage_train, "train")
    if args.stage in ("all", "sanity"):
        stage("sanity", stage_sanity, "sanity")
    if args.stage in ("all", "eval"):
        if args.stage == "all":
            args.passes = "1,3,4,2"  # cheap+decisive first, expensive last
        stage(f"eval_p{args.passes.replace(',', '')}", stage_eval, f"eval {args.passes}")
    if args.stage in ("all", "profile"):
        stage("profile", stage_profile, "profile")
    if args.stage in ("all", "plot"):
        stage("plot", stage_plot, "plot")

    if not args.no_gh:
        ok = all(published)
        print(f"\nGitHub link: {'OK' if ok else 'FAILED'} -> https://github.com/{args.gh_repo}"
              f"/tree/kaggle-results/runs/{args.run_id}", flush=True)
        if args.smoke and not ok:
            return 2  # a broken link must stop you before the long run, not after
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
