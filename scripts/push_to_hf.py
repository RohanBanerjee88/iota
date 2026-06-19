"""Push run artifacts (checkpoints, run jsons, result CSVs) to the Hugging Face Hub.

Kaggle wipes non-output state at session end, so call this DURING the run.

Usage:
    python scripts/push_to_hf.py --repo <user>/<repo> [--type model|dataset] \
        [--results_dir experiments/results] [--include "*.safetensors,*.json,*.csv"]

Requires `huggingface_hub` and an HF token (Kaggle: log in once, or set HF_TOKEN).
The dependency is optional and isolated — this script is the only thing that uses it.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="upload iota artifacts to HF Hub")
    ap.add_argument("--repo", required=True, help="e.g. yourname/iota-sweep")
    ap.add_argument("--type", default="model", choices=["model", "dataset"])
    ap.add_argument("--results_dir", default="experiments/results")
    ap.add_argument("--include", default="*.safetensors,*.json,*.csv")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("huggingface_hub not installed: pip install huggingface_hub", file=sys.stderr)
        return 1

    patterns = [p.strip() for p in args.include.split(",") if p.strip()]
    files = [
        f for f in os.listdir(args.results_dir)
        if any(fnmatch.fnmatch(f, p) for p in patterns)
    ]
    if not files:
        print(f"no matching files in {args.results_dir}", file=sys.stderr)
        return 1

    api = HfApi(token=args.token)
    create_repo(args.repo, repo_type=args.type, exist_ok=True, token=args.token)
    for f in sorted(files):
        path = os.path.join(args.results_dir, f)
        api.upload_file(path_or_fileobj=path, path_in_repo=f, repo_id=args.repo,
                        repo_type=args.type, token=args.token)
        print(f"uploaded {f} -> {args.repo}")
    print(f"done: {len(files)} files -> https://huggingface.co/{args.repo}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
