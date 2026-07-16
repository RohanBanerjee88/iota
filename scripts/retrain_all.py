"""Long-context retrain driver for the Phase-6 sweep (run on the Kaggle GPU).

Trains each architecture on the long-context curriculum (seq_len<=640) so the
capacity sweep isn't confounded by length extrapolation, and backs each model up
to the HF Hub IMMEDIATELY after it finishes -- so a Kaggle session timeout only
costs the in-progress model, not the ones already done.

The trainer now logs per-mode accuracy (`[assoc .. state ..]`) every eval, so you
can watch BOTH the recall task and the state_track control learn in real time. A
healthy run ends with assoc high AND state well above chance (~0.01).

Usage (Kaggle):
    !HF_TOKEN=$HF_TOKEN python -m scripts.retrain_all --repo BanerjeeRohan44/iota-sweep
    # single model:
    !python -m scripts.retrain_all --only gated_linear --repo BanerjeeRohan44/iota-sweep
    # skip HF backup (local only):
    !python -m scripts.retrain_all --no-hf

Ordered shortest->longest (hybrid, transformer, gated_linear) so results
accumulate early and the slowest model (most timeout-prone) runs last, after the
others are already safe on HF.
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

RESULTS_DIR = "experiments/results"
ORDER = ["hybrid", "transformer", "gated_linear"]  # short -> long


def _push_to_hf(run_name: str, repo: str, token: str) -> None:
    """Upload just this run's artifacts to the HF Hub (best-effort)."""
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("  [hf] huggingface_hub not installed -> skipping backup", flush=True)
        return
    if not token:
        print("  [hf] no HF_TOKEN in env -> skipping backup", flush=True)
        return
    api = HfApi(token=token)
    create_repo(repo, repo_type="model", exist_ok=True, token=token)
    for ext in ("safetensors", "json"):
        f = f"{run_name}.{ext}"
        path = os.path.join(RESULTS_DIR, f)
        if os.path.exists(path):
            api.upload_file(path_or_fileobj=path, path_in_repo=f, repo_id=repo,
                            repo_type="model", token=token)
            print(f"  [hf] uploaded {f} -> {repo}", flush=True)


def _final_by_mode(history) -> str:
    """Pull the last eval's per-mode accuracy out of the run history for a summary."""
    evals = history.get("eval_acc", [])
    if not evals:
        return "(no eval recorded)"
    last = evals[-1]
    bm = last.get("by_mode", {})
    parts = [f"{m}={v:.3f}" for m, v in sorted(bm.items())]
    return f"step {last.get('step')}: " + " ".join(parts) + f"  (balanced {last.get('balanced')})"


def main() -> int:
    ap = argparse.ArgumentParser(description="long-context retrain driver")
    ap.add_argument("--only", choices=ORDER, help="train just one architecture")
    ap.add_argument("--repo", default="BanerjeeRohan44/iota-sweep")
    ap.add_argument("--no-hf", action="store_true", help="skip HF backup")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    # Import here so a missing torch/etc. surfaces with the training call, not at parse.
    from iota.train import train

    archs = [args.only] if args.only else ORDER
    summary = {}
    for arch in archs:
        cfg_path = f"configs/sweep_{arch}.yaml"
        cfg = yaml.safe_load(open(cfg_path))
        run_name = cfg.get("train", {}).get("run_name", f"{arch}_sweep")
        print(f"\n{'='*72}\n=== RETRAIN {arch}  ({cfg_path} -> {run_name})\n{'='*72}", flush=True)
        try:
            out = train(cfg, smoke=False)
        except Exception as e:  # keep going so one blow-up doesn't lose the others
            print(f"!! {arch} FAILED: {type(e).__name__}: {e}", flush=True)
            summary[arch] = f"FAILED: {e}"
            continue
        summary[arch] = _final_by_mode(out.get("history", {}))
        print(f"=== {arch} done. best balanced acc = {out.get('best_acc'):.4f}", flush=True)
        if not args.no_hf:
            try:
                _push_to_hf(run_name, args.repo, args.token)
            except Exception as e:
                print(f"  [hf] backup failed ({e}); weights are still local in "
                      f"{RESULTS_DIR}", flush=True)

    print(f"\n{'='*72}\n=== RETRAIN SUMMARY\n{'='*72}", flush=True)
    for arch in archs:
        print(f"  {arch:14s} {summary.get(arch, '(not run)')}", flush=True)
    print("\nHealthy = assoc high AND state well above chance (~0.01). If state is "
          "still ~0.01, the control didn't learn -> tell Claude and we adjust it.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
