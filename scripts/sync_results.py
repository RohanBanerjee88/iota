"""Publish a run's results to the `kaggle-results` branch on GitHub.

This is the link between a Kaggle run and whoever analyses it (you, or Claude in a
later session). Kaggle wipes its disk when a session ends and Claude's sandbox
cannot reach Kaggle or the HF Hub -- but both can reach GitHub. So after every
stage, the run's small artifacts are committed to a dedicated branch:

    kaggle-results
      README.md                 index of every run, newest first
      runs/<run_id>/
        SUMMARY.md              human + Claude readable digest (regenerated each sync)
        env.json                device, torch, code commit, last stage
        syncs.log               one line per sync (what stage, when, which commit)
        *_sweep.json            per-model training history (per-mode accuracy curve)
        pass*_*.csv, cost_profile.csv, money_figure*.png/pdf
        logs/<stage>.log        full stdout of each stage

Weights (*.safetensors) are NOT pushed here -- they live on the HF Hub.

Files already on the branch are kept (merge, not replace), so Session B's sync
adds to Session A's results instead of wiping them.

    GH_TOKEN=... python -m scripts.sync_results --run-id r01 --stage manual

A fine-grained GitHub token with "Contents: read and write" on this one repo is
enough. The token is never printed; any git error text has it scrubbed.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import glob
import json
import os
import shutil
import subprocess
import tempfile
import time
from typing import Dict, List, Optional

DEFAULT_REPO = "RohanBanerjee88/iota"
DEFAULT_BRANCH = "kaggle-results"
PUSH_EXTS = (".json", ".csv", ".png", ".pdf")
MAX_FILE_MB = 20
ARCHS = ["transformer", "gated_linear", "hybrid"]
CHANCE = 0.01


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _scrub(text: str, secret: Optional[str]) -> str:
    return text.replace(secret, "***") if secret else text


def _git(args: List[str], cwd: Optional[str] = None, secret: Optional[str] = None,
         check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    p = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, env=env)
    if check and p.returncode != 0:
        msg = _scrub(f"git {' '.join(args)} failed ({p.returncode}): {p.stderr.strip()}", secret)
        raise RuntimeError(msg)
    return p


def _code_commit() -> Dict[str, str]:
    """The commit of the CODE that produced the results (the Kaggle checkout)."""
    out = {}
    for key, args in (("commit", ["rev-parse", "--short", "HEAD"]),
                      ("branch", ["rev-parse", "--abbrev-ref", "HEAD"])):
        p = _git(args, check=False)
        out[key] = p.stdout.strip() if p.returncode == 0 else "?"
    return out


def _device_info() -> Dict[str, str]:
    info = {"kaggle_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "")}
    try:
        import torch

        info["torch"] = torch.__version__
        info["device"] = (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    except Exception:  # torch missing is fine for a manual sync
        info["device"] = "?"
    return info


# ---------------------------------------------------------------------------
# collecting the bundle
# ---------------------------------------------------------------------------
def collect(results_dir: str, dest: str) -> List[str]:
    """Copy the small artifacts from results_dir into dest. Returns copied paths."""
    os.makedirs(dest, exist_ok=True)
    copied = []
    for name in sorted(os.listdir(results_dir)) if os.path.isdir(results_dir) else []:
        src = os.path.join(results_dir, name)
        if not os.path.isfile(src) or not name.endswith(PUSH_EXTS):
            continue
        if os.path.getsize(src) > MAX_FILE_MB * 1024 * 1024:
            print(f"  [sync] skip {name} (> {MAX_FILE_MB} MB)", flush=True)
            continue
        shutil.copy2(src, os.path.join(dest, name))
        copied.append(name)
    logs = os.path.join(results_dir, "logs")
    if os.path.isdir(logs):
        os.makedirs(os.path.join(dest, "logs"), exist_ok=True)
        for src in sorted(glob.glob(os.path.join(logs, "*.log"))):
            shutil.copy2(src, os.path.join(dest, "logs", os.path.basename(src)))
            copied.append(f"logs/{os.path.basename(src)}")
    return copied


# ---------------------------------------------------------------------------
# SUMMARY.md -- built from whatever is in the run directory
# ---------------------------------------------------------------------------
def _arch_of(name: str) -> str:
    for suffix in ("_sweep", "_milestone"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _read_csv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _fmt(v, nd: int = 3) -> str:
    if v in (None, "", "None"):
        return "OOM"
    try:
        return f"{float(v):.{nd}f}"
    except ValueError:
        return str(v)


def _train_section(run_dir: str) -> List[str]:
    lines = ["## 1. Training (best checkpoint, full-difficulty held-out set)", ""]
    rows = []
    for arch in ARCHS:
        p = os.path.join(run_dir, f"{arch}_sweep.json")
        if not os.path.exists(p):
            rows.append((arch, None))
            continue
        try:
            run = json.load(open(p))
        except Exception:
            rows.append((arch, None))
            continue
        evals = run.get("history", {}).get("eval_acc", [])
        rows.append((arch, (run, evals)))
    if all(r[1] is None for r in rows):
        return lines + ["_No training runs synced yet._", ""]
    lines += ["| model | best step | balanced | assoc (per-query) | state (control) | exact | lr | verdict |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for arch, got in rows:
        if got is None:
            lines.append(f"| {arch} | – | – | – | – | – | – | not trained yet |")
            continue
        run, evals = got
        if not evals:
            lines.append(f"| {arch} | – | – | – | – | – | – | no evals recorded |")
            continue
        best = max(evals, key=lambda e: e.get("balanced", e.get("per_query", 0)) or 0)
        bm = best.get("by_mode", {})
        assoc, state = bm.get("assoc_recall"), bm.get("state_track")
        lr = run.get("config", {}).get("train", {}).get("lr", "?")
        if state is not None and state < 5 * CHANCE:
            verdict = "⚠ control at chance -- figure NOT trustworthy"
        elif assoc is not None and assoc < 5 * CHANCE:
            verdict = "⚠ recall at chance -- model never learned"
        else:
            verdict = "ok"
        lines.append(f"| {arch} | {best.get('step')} | {_fmt(best.get('balanced'))} | "
                     f"{_fmt(assoc)} | {_fmt(state)} | {_fmt(best.get('exact'))} | {lr} | {verdict} |")
    lines += ["", "Healthy = assoc high **and** state well above chance (~0.01). "
              "Full per-eval curves are in `logs/train.log` and the `*_sweep.json` files.", ""]
    return lines


def _tune_section(run_dir: str) -> List[str]:
    rows = _read_csv(os.path.join(run_dir, "tune.csv"))
    if not rows:
        return []
    lines = ["## 0. LR probe (identical short budget per arch; pick each arch's best)", "",
             "| arch | lr | best balanced | assoc | state | exact | best step | final balanced | time |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    best = {}
    for r in rows:
        try:
            if float(r["best_balanced"]) > float(best.get(r["arch"], {}).get("best_balanced", -1)):
                best[r["arch"]] = r
        except (TypeError, ValueError):
            pass
    for r in sorted(rows, key=lambda r: (ARCHS.index(r["arch"]) if r["arch"] in ARCHS else 9,
                                         float(r["lr"]))):
        star = " ★" if best.get(r["arch"]) is r else ""
        lines.append(f"| {r['arch']}{star} | {float(r['lr']):g} | {_fmt(r['best_balanced'])} | "
                     f"{_fmt(r['assoc'])} | {_fmt(r['state'])} | {_fmt(r['exact'])} | "
                     f"{r['best_step']} | {_fmt(r['final_balanced'])} | {r['seconds']}s |")
    return lines + ["", f"{rows[0]['steps']} steps each, grad_clip {rows[0]['grad_clip']}. "
                    "★ = best lr for that arch.", ""]


def _sanity_section(run_dir: str) -> List[str]:
    rows = _read_csv(os.path.join(run_dir, "sanity_indist.csv"))
    lines = ["## 1b. Sanity gate — each model on its own training distribution", ""]
    if not rows:
        return lines + ["_`sanity_indist.csv` not synced yet._", ""]
    lines += ["| model | assoc per-query | assoc exact | state per-query | state exact | gate |",
              "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        ok = all(float(r[k]) >= 5 * CHANCE for k in ("assoc_pq", "state_pq"))
        lines.append(f"| {r['model']} | {_fmt(r['assoc_pq'])} | {_fmt(r['assoc_exact'])} | "
                     f"{_fmt(r['state_pq'])} | {_fmt(r['state_exact'])} | "
                     f"{'pass' if ok else '⚠ FAIL — a task is at chance'} |")
    return lines + [f"", f"n={rows[0].get('n')} per mode. Both per-query columns must be well "
                    "above chance (~0.01) before any sweep number means anything.", ""]


def _pass_section(run_dir: str, fname: str, title: str, xkey: str, xlabel: str) -> List[str]:
    rows = _read_csv(os.path.join(run_dir, fname))
    lines = [f"## {title}", ""]
    if not rows:
        return lines + [f"_`{fname}` not synced yet._", ""]
    models = [a for a in ARCHS if any(_arch_of(r["model"]) == a for r in rows)]
    xs = sorted({int(float(r[xkey])) for r in rows})
    lines += [f"Per-query accuracy [95% CI], exact-all-queries in parentheses. "
              f"n={rows[0].get('n')} per cell, paired prompts.", "",
              f"| {xlabel} | true tokens | " + " | ".join(models) + " |",
              "|---:|---:|" + "---|" * len(models)]
    for x in xs:
        cells, tl = [], ""
        for m in models:
            hit = [r for r in rows if _arch_of(r["model"]) == m and int(float(r[xkey])) == x]
            if not hit:
                cells.append("–")
                continue
            r = hit[0]
            tl = tl or _fmt(r.get("seq_len_true_tokens"), 0)
            if r.get("accuracy_per_query") in (None, "", "None"):
                cells.append("OOM")
            else:
                cells.append(f"{_fmt(r['accuracy_per_query'])} "
                             f"[{_fmt(r.get('ci_low_pq'), 2)}–{_fmt(r.get('ci_high_pq'), 2)}] "
                             f"({_fmt(r.get('accuracy_exact'), 2)})")
        lines.append(f"| {x} | {tl} | " + " | ".join(cells) + " |")
    return lines + [""]


def _keylen_section(run_dir: str) -> List[str]:
    rows = _read_csv(os.path.join(run_dir, "pass4_keylen.csv"))
    if not rows:
        return []
    models = [a for a in ARCHS if any(_arch_of(r["model"]) == a for r in rows)]
    lines = ["## 4b. Diagnostic — recall by key length (digits), same prompts as Pass 1", "",
             "| n_bindings | key digits | " + " | ".join(models) + " |",
             "|---:|---:|" + "---:|" * len(models)]
    for nb in sorted({int(r["n_bindings"]) for r in rows}):
        for d in sorted({int(r["key_digits"]) for r in rows}):
            cells = []
            for m in models:
                hit = [r for r in rows if _arch_of(r["model"]) == m and int(r["n_bindings"]) == nb
                       and int(r["key_digits"]) == d]
                cells.append(_fmt(hit[0]["accuracy_per_query"]) if hit else "–")
            lines.append(f"| {nb} | {d} | " + " | ".join(cells) + " |")
    return lines + ["", "If one model's errors pile up on 3-digit keys, its drop with load is partly "
                    "key resolution, not memory capacity.", ""]


def _cost_section(run_dir: str) -> List[str]:
    rows = _read_csv(os.path.join(run_dir, "cost_profile.csv"))
    lines = ["## 5. Prefill cost (one forward pass, batch 1; mostly kernel quality)", ""]
    if not rows:
        return lines + ["_`cost_profile.csv` not synced yet._", ""]
    models = [a for a in ARCHS if any(r.get("arch") == a for r in rows)]
    lines += ["Peak VRAM MB / median latency ms.", "",
              "| seq_len | " + " | ".join(models) + " |", "|---:|" + "---|" * len(models)]
    for sl in sorted({int(r["seq_len"]) for r in rows}):
        cells = []
        for m in models:
            hit = [r for r in rows if r.get("arch") == m and int(r["seq_len"]) == sl]
            if not hit:
                cells.append("–")
            elif str(hit[0].get("oom")).lower() in ("true", "1"):
                cells.append("OOM")
            else:
                cells.append(f"{_fmt(hit[0].get('peak_vram_mb'), 1)} MB / "
                             f"{_fmt(hit[0].get('latency_ms'), 2)} ms")
        lines.append(f"| {sl} | " + " | ".join(cells) + " |")
    return lines + [""]


def _decode_section(run_dir: str) -> List[str]:
    rows = _read_csv(os.path.join(run_dir, "decode_profile.csv"))
    lines = ["## 5b. Decode cost (one token at a time, batch 1)", ""]
    if not rows:
        return lines + ["_`decode_profile.csv` not synced yet._", ""]
    models = [a for a in ARCHS if any(r.get("arch") == a for r in rows)]
    lines += ["Memory each model keeps per sequence (exact cache/state size) / median ms per token.", "",
              "| context | " + " | ".join(models) + " |", "|---:|" + "---|" * len(models)]
    for L in sorted({int(r["context_len"]) for r in rows}):
        cells = []
        for m in models:
            hit = [r for r in rows if r.get("arch") == m and int(r["context_len"]) == L]
            if not hit:
                cells.append("–")
            elif str(hit[0].get("oom")).lower() in ("true", "1"):
                cells.append("OOM")
            else:
                cells.append(f"{_fmt(hit[0].get('cache_mb'), 2)} MB / "
                             f"{_fmt(hit[0].get('decode_ms_per_tok'), 2)} ms")
        lines.append(f"| {L} | " + " | ".join(cells) + " |")
    return lines + [""]


def build_summary(run_dir: str, run_id: str) -> str:
    env = {}
    if os.path.exists(os.path.join(run_dir, "env.json")):
        env = json.load(open(os.path.join(run_dir, "env.json")))
    head = [
        f"# iota run `{run_id}`", "",
        f"- **updated:** {env.get('updated', '?')} (last stage: `{env.get('stage', '?')}`)",
        f"- **code:** `{env.get('commit', '?')}` on `{env.get('branch', '?')}`",
        f"- **device:** {env.get('device', '?')} · torch {env.get('torch', '?')}"
        + (f" · Kaggle {env['kaggle_run_type']}" if env.get("kaggle_run_type") else ""),
        "",
    ]
    fig = [f for f in ("money_figure.png",) if os.path.exists(os.path.join(run_dir, f))]
    body = (
        _tune_section(run_dir)
        + _train_section(run_dir)
        + _sanity_section(run_dir)
        + _pass_section(run_dir, "pass1_capacity.csv", "2. Pass 1 — capacity (the headline)",
                        "n_bindings", "n_bindings")
        + _pass_section(run_dir, "pass2_length.csv", "3. Pass 2 — length generalisation",
                        "seq_len_nominal", "seq_len")
        + _pass_section(run_dir, "pass3_control.csv", "4. Pass 3 — state_track control",
                        "seq_len_nominal", "seq_len")
        + _keylen_section(run_dir)
        + _cost_section(run_dir)
        + _decode_section(run_dir)
    )
    tail = ["## 6. Figure", ""]
    tail += [f"![money figure]({fig[0]})", ""] if fig else ["_Not plotted yet._", ""]
    return "\n".join(head + body + tail)


def _write_index(root: str) -> None:
    runs = []
    for d in sorted(glob.glob(os.path.join(root, "runs", "*"))):
        envp = os.path.join(d, "env.json")
        env = json.load(open(envp)) if os.path.exists(envp) else {}
        runs.append((env.get("updated", ""), os.path.basename(d), env))
    runs.sort(reverse=True)
    lines = ["# iota — Kaggle results", "",
             "Written automatically by `scripts/sync_results.py`. Never merge this branch "
             "into `main`; it only holds run outputs.", "",
             "| run | updated | last stage | device | code |", "|---|---|---|---|---|"]
    for updated, rid, env in runs:
        lines.append(f"| [{rid}](runs/{rid}/SUMMARY.md) | {updated} | {env.get('stage', '?')} | "
                     f"{env.get('device', '?')} | `{env.get('commit', '?')}` |")
    with open(os.path.join(root, "README.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# push
# ---------------------------------------------------------------------------
def _prepare_checkout(url: str, branch: str, tmp: str, secret: Optional[str]) -> None:
    probe = _git(["ls-remote", "--heads", url, branch], secret=secret, check=False)
    if probe.returncode != 0:
        raise RuntimeError(_scrub(
            f"cannot reach the repo (bad token or no internet?): {probe.stderr.strip()}", secret))
    if probe.stdout.strip():
        _git(["clone", "--quiet", "--depth", "1", "--branch", branch, url, tmp], secret=secret)
    else:  # first ever sync: start an orphan branch that holds only results
        _git(["init", "--quiet", tmp])
        _git(["checkout", "--quiet", "-b", branch], cwd=tmp)
        _git(["remote", "add", "origin", url], cwd=tmp, secret=secret)


def sync(results_dir: str, run_id: str, stage: str = "manual", repo: str = DEFAULT_REPO,
         branch: str = DEFAULT_BRANCH, token: Optional[str] = None,
         remote_url: Optional[str] = None, retries: int = 4) -> bool:
    """Commit this run's artifacts to `branch`. Returns True on success, never raises."""
    token = token or os.environ.get("GH_TOKEN")
    if not remote_url and not token:
        print("  [sync] no GH_TOKEN -> results NOT published to GitHub", flush=True)
        return False
    url = remote_url or f"https://x-access-token:{token}@github.com/{repo}.git"
    code = _code_commit()
    delay = 2
    for attempt in range(1, retries + 1):
        tmp = tempfile.mkdtemp(prefix="iota-sync-")
        try:
            _prepare_checkout(url, branch, tmp, token)
            run_dir = os.path.join(tmp, "runs", run_id)
            copied = collect(results_dir, run_dir)
            env = {"run_id": run_id, "stage": stage, "updated": _now(), **code, **_device_info()}
            with open(os.path.join(run_dir, "env.json"), "w") as fh:
                json.dump(env, fh, indent=2)
            with open(os.path.join(run_dir, "syncs.log"), "a") as fh:
                fh.write(f"{env['updated']}  stage={stage}  code={code['commit']}  "
                         f"files={len(copied)}\n")
            with open(os.path.join(run_dir, "SUMMARY.md"), "w") as fh:
                fh.write(build_summary(run_dir, run_id))
            _write_index(tmp)
            _git(["add", "-A"], cwd=tmp)
            _git(["-c", "user.name=iota-kaggle", "-c", "user.email=iota-kaggle@users.noreply.github.com",
                  "commit", "--quiet", "-m", f"results: {run_id} after {stage}"], cwd=tmp)
            _git(["push", "--quiet", "origin", f"HEAD:{branch}"], cwd=tmp, secret=token)
            where = (f"https://github.com/{repo}/tree/{branch}/runs/{run_id}"
                     if not remote_url else f"{remote_url} ({branch})")
            print(f"  [sync] published {len(copied)} files -> {where}", flush=True)
            return True
        except Exception as e:
            print(f"  [sync] attempt {attempt}/{retries} failed: {_scrub(str(e), token)}", flush=True)
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    print("  [sync] !! GitHub publish FAILED -- results are only on this machine / the HF Hub",
          flush=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="publish run results to the kaggle-results branch")
    ap.add_argument("--run-id", default=os.environ.get("IOTA_RUN_ID", "adhoc"))
    ap.add_argument("--results-dir", default="experiments/results")
    ap.add_argument("--stage", default="manual")
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--branch", default=DEFAULT_BRANCH)
    ap.add_argument("--remote-url", default=None, help="override the push URL (tests)")
    ap.add_argument("--summary-only", action="store_true",
                    help="print the SUMMARY.md this results dir would produce; no push")
    args = ap.parse_args()
    if args.summary_only:
        tmp = tempfile.mkdtemp(prefix="iota-summary-")
        try:
            collect(args.results_dir, tmp)
            print(build_summary(tmp, args.run_id))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return 0
    ok = sync(args.results_dir, args.run_id, stage=args.stage, repo=args.repo,
              branch=args.branch, remote_url=args.remote_url)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
