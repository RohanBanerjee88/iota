"""End-to-end, RESUMABLE driver for the Phase-6 sweep (run this on Kaggle).

One command per stage; every stage is safe to re-run. Work that is already done is
skipped, so a Kaggle session timeout costs you only the in-progress model -- never
the whole run.

    python -m scripts.run_all --stage train    --repo <user>/iota-sweep
    python -m scripts.run_all --stage eval     --passes 1,3
    python -m scripts.run_all --stage eval     --passes 2
    python -m scripts.run_all --stage profile
    python -m scripts.run_all --stage plot
    python -m scripts.run_all --stage all      --repo <user>/iota-sweep

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
import json
import os
import sys

import yaml

RESULTS_DIR = "experiments/results"
# short -> long: results bank early, and the slow learner (most timeout-prone)
# runs last, once the others are already safe on the Hub.
ARCH_ORDER = ["hybrid", "transformer", "gated_linear"]
MIN_REAL_STEPS = 1000  # below this it's a smoke run, not a trained model


# ---------------------------------------------------------------------------
# checkpoint bookkeeping
# ---------------------------------------------------------------------------
def _cfg_path(arch: str) -> str:
    return f"configs/sweep_{arch}.yaml"


def _run_name(cfg: dict, arch: str) -> str:
    return cfg.get("train", {}).get("run_name", f"{arch}_sweep")


def _fingerprint(cfg: dict) -> dict:
    """The parts of a config that define WHAT the model learned.

    Architecture + curriculum only. Deliberately excludes lr / max_steps / patience:
    re-tuning the optimizer shouldn't invalidate a finished checkpoint, but changing
    the task distribution (sequence length, binding/query ceilings, task mix) must.
    """
    return {"arch": cfg.get("arch"), "curriculum": cfg.get("curriculum")}


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
    if last_step < MIN_REAL_STEPS:
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

    archs = [args.only] if args.only else ARCH_ORDER
    print(f"\n### STAGE train  ({', '.join(archs)})\n", flush=True)
    for arch in archs:
        status, run_name, detail = checkpoint_status(arch)
        if status == "missing" and not args.no_hf:
            print(f"[{arch}] no local checkpoint -> trying the Hub", flush=True)
            if hf_pull(run_name, args.repo, args.token):
                status, run_name, detail = checkpoint_status(arch)
        if status == "ok" and not args.force:
            print(f"[{arch}] SKIP -- {detail}", flush=True)
            continue
        if status == "stale":
            print(f"[{arch}] RETRAIN -- {detail}", flush=True)
        print(f"\n{'='*72}\n=== TRAIN {arch} -> {run_name}\n{'='*72}", flush=True)
        cfg = yaml.safe_load(open(_cfg_path(arch)))
        try:
            out = train(cfg, smoke=False)
        except Exception as e:
            print(f"!! {arch} FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        print(f"=== {arch} done, best balanced acc = {out.get('best_acc'):.4f}", flush=True)
        if not args.no_hf:
            try:
                hf_push(run_name, args.repo, args.token)
            except Exception as e:
                print(f"  [hf] push failed ({e}); weights still local", flush=True)

    print(f"\n{'='*72}\n=== TRAIN SUMMARY\n{'='*72}", flush=True)
    for arch in archs:
        status, run_name, detail = checkpoint_status(arch)
        print(f"  {arch:14s} [{status}] {_by_mode(run_name)}", flush=True)
    print("\nHealthy = assoc high AND state well above chance (~0.01).\n"
          "If state is still ~0.01 the control never learned -- stop and fix that\n"
          "before trusting any figure.", flush=True)


def stage_eval(args) -> None:
    import torch

    from iota.eval import load_checkpoint, run_pass

    device = args.device
    passes = [int(p) for p in args.passes.split(",") if p.strip()]
    print(f"\n### STAGE eval  (passes {passes}, device={device})\n", flush=True)

    models = {}
    for arch in ARCH_ORDER:
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
    if len(models) < len(ARCH_ORDER):
        print(f"WARNING: evaluating {len(models)}/{len(ARCH_ORDER)} models -- the "
              f"comparison is incomplete", flush=True)

    names = {1: "pass1_capacity", 2: "pass2_length", 3: "pass3_control"}
    for p in passes:
        out_csv = os.path.join(RESULTS_DIR, f"{names.get(p, f'pass{p}')}.csv")
        print(f"\n--- pass {p} -> {out_csv}", flush=True)
        run_pass(p, models, n=args.n, device=device, out_csv=out_csv,
                 minibatch=args.minibatch)
        if not args.no_hf:
            try:
                hf_push_results(args.repo, args.token)
            except Exception as e:
                print(f"  [hf] results push failed ({e})", flush=True)


def stage_profile(args) -> None:
    from iota.profile import run_profile

    print("\n### STAGE profile\n", flush=True)
    run_profile(out_csv=os.path.join(RESULTS_DIR, "cost_profile.csv"),
                device=args.device)
    if not args.no_hf:
        try:
            hf_push_results(args.repo, args.token)
        except Exception as e:
            print(f"  [hf] results push failed ({e})", flush=True)


def stage_plot(args) -> None:
    from iota.plot import make_figure

    print("\n### STAGE plot\n", flush=True)
    make_figure(results_dir=RESULTS_DIR)
    if not args.no_hf:
        try:
            hf_push_results(args.repo, args.token)
        except Exception as e:
            print(f"  [hf] results push failed ({e})", flush=True)


def main() -> int:
    import torch

    ap = argparse.ArgumentParser(description="iota Phase-6 end-to-end driver")
    ap.add_argument("--stage", default="all",
                    choices=["all", "train", "eval", "profile", "plot", "status"])
    ap.add_argument("--only", choices=ARCH_ORDER, help="train just one architecture")
    ap.add_argument("--passes", default="1,3", help="eval passes, e.g. '1,3' or '2'")
    ap.add_argument("--repo", default="BanerjeeRohan44/iota-sweep")
    ap.add_argument("--no-hf", action="store_true", help="skip all Hub traffic")
    ap.add_argument("--force", action="store_true", help="retrain even if a checkpoint is ok")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--n", type=int, default=1000, help="examples per eval cell")
    ap.add_argument("--minibatch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"iota run_all: stage={args.stage} device={args.device} "
          f"hf={'off' if args.no_hf else args.repo}", flush=True)

    if args.stage == "status":
        for arch in ARCH_ORDER:
            status, run_name, detail = checkpoint_status(arch)
            print(f"  {arch:14s} [{status:7s}] {detail}", flush=True)
        return 0

    if args.stage in ("all", "train"):
        stage_train(args)
    if args.stage in ("all", "eval"):
        if args.stage == "all":
            args.passes = "1,3,2"  # cheap+decisive first, expensive last
        stage_eval(args)
    if args.stage in ("all", "profile"):
        stage_profile(args)
    if args.stage in ("all", "plot"):
        stage_plot(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
