"""Combine seed-replicate runs into one figure: mean across seeds, band = seed range.

One seed is not a result: r04 (seed 0) and r05 (seed 1) disagreed by up to 0.26 on
the same prompts at high recall load. This script reads several runs from the
`kaggle-results` branch (or local run directories), averages every eval cell
across seeds, and draws the money figure with the band showing the min-max
spread ACROSS SEEDS rather than one run's sampling CI. Every per-seed value is
kept in the output CSVs and in SEEDS.md, so the mean never hides a disagreement.

    python -m scripts.aggregate_seeds --runs r04 r05 r06            # from origin/kaggle-results
    python -m scripts.aggregate_seeds --dirs path/r04 path/r05     # local run directories

Decode/prefill cost depend only on the architecture, so they are taken from the
first run that has them, not averaged.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import subprocess
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional

PASS_FILES = ["pass1_capacity.csv", "pass2_length.csv", "pass3_control.csv"]
KEYLEN_FILE = "pass4_keylen.csv"
GRID_FILE = "pass5_grid.csv"
FREERUN_FILE = "pass6_freerun.csv"
COST_FILES = ["decode_profile.csv", "cost_profile.csv"]
ALL_FILES = PASS_FILES + [KEYLEN_FILE, GRID_FILE, FREERUN_FILE] + COST_FILES
ARCHS = ["transformer", "gated_linear", "hybrid"]


def _arch(model: str) -> str:
    return model[: -len("_sweep")] if model.endswith("_sweep") else model


def _read(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def fetch_runs(runs: List[str], ref: str, dest: str) -> Dict[str, str]:
    """Copy each run's CSVs out of a git ref (e.g. origin/kaggle-results)."""
    out = {}
    for run in runs:
        d = os.path.join(dest, run)
        os.makedirs(d, exist_ok=True)
        for f in ALL_FILES:
            p = subprocess.run(["git", "show", f"{ref}:runs/{run}/{f}"], capture_output=True, text=True)
            if p.returncode == 0:
                with open(os.path.join(d, f), "w") as fh:
                    fh.write(p.stdout)
        out[run] = d
    return out


def _f(v) -> Optional[float]:
    try:
        return None if v in (None, "", "None") else float(v)
    except ValueError:
        return None


def aggregate_pass(rows_by_run: Dict[str, List[Dict]]) -> List[Dict]:
    """Mean per cell across runs; ci_*_pq / ci_* become the across-seed min/max."""
    keyf = ("model", "mode", "pass", "seq_len_nominal", "n_bindings")
    cells = defaultdict(list)
    for run, rows in rows_by_run.items():
        for r in rows:
            cells[tuple(r[k] for k in keyf)].append((run, r))
    out = []
    for key, items in cells.items():
        pq = [(run, _f(r["accuracy_per_query"])) for run, r in items]
        ex = [_f(r["accuracy_exact"]) for _, r in items]
        pq_v = [v for _, v in pq if v is not None]
        ex_v = [v for v in ex if v is not None]
        first = items[0][1]
        row = dict(first)
        if pq_v:
            row.update({
                "accuracy_per_query": round(statistics.fmean(pq_v), 4),
                "ci_low_pq": round(min(pq_v), 4), "ci_high_pq": round(max(pq_v), 4),
                "accuracy_exact": round(statistics.fmean(ex_v), 4) if ex_v else None,
                "ci_low": round(min(ex_v), 4) if ex_v else None,
                "ci_high": round(max(ex_v), 4) if ex_v else None,
            })
        else:  # every seed OOM'd on this cell
            row.update({k: None for k in ("accuracy_per_query", "ci_low_pq", "ci_high_pq",
                                          "accuracy_exact", "ci_low", "ci_high")})
        row["seq_len_true_tokens"] = round(statistics.fmean(
            [_f(r["seq_len_true_tokens"]) for _, r in items if _f(r["seq_len_true_tokens"]) is not None] or [0]), 1)
        row["seed"] = "+".join(run for run, _ in items)
        row["n_seeds"] = len(pq_v)
        row["per_seed"] = " ".join(f"{run}={'OOM' if v is None else f'{v:.3f}'}" for run, v in pq)
        out.append(row)
    return out


def aggregate_keylen(rows_by_run: Dict[str, List[Dict]]) -> List[Dict]:
    cells = defaultdict(list)
    for run, rows in rows_by_run.items():
        for r in rows:
            cells[(r["model"], r["n_bindings"], r["key_digits"])].append((run, _f(r["accuracy_per_query"])))
    out = []
    for (model, nb, d), items in cells.items():
        vals = [v for _, v in items if v is not None]
        out.append({"model": model, "n_bindings": nb, "key_digits": d,
                    "accuracy_per_query": round(statistics.fmean(vals), 4) if vals else None,
                    "ci_low": round(min(vals), 4) if vals else None,
                    "ci_high": round(max(vals), 4) if vals else None,
                    "n_queries": "", "seed": "+".join(r for r, _ in items), "n_seeds": len(vals),
                    "per_seed": " ".join(f"{r}={v:.3f}" for r, v in items if v is not None)})
    return out


def _write(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _md_table(rows: List[Dict], xkey: str, xlabel: str) -> List[str]:
    models = [a for a in ARCHS if any(_arch(r["model"]) == a for r in rows)]
    lines = [f"| {xlabel} | " + " | ".join(models) + " |", "|---:|" + "---|" * len(models)]
    for x in sorted({int(float(r[xkey])) for r in rows}):
        cells = []
        for m in models:
            hit = [r for r in rows if _arch(r["model"]) == m and int(float(r[xkey])) == x]
            if not hit or hit[0]["accuracy_per_query"] is None:
                cells.append("–" if not hit else "OOM")
                continue
            r = hit[0]
            cells.append(f"**{r['accuracy_per_query']:.3f}** [{r['ci_low_pq']:.2f}–{r['ci_high_pq']:.2f}] "
                         f"<sub>{r['per_seed']}</sub>")
        lines.append(f"| {x} | " + " | ".join(cells) + " |")
    return lines + [""]


def aggregate(run_dirs: Dict[str, str], out_dir: str, make_plot: bool = True) -> Dict[str, List[Dict]]:
    os.makedirs(out_dir, exist_ok=True)
    result = {}
    titles = {"pass1_capacity.csv": ("Capacity (Pass 1)", "n_bindings", "n_bindings"),
              "pass2_length.csv": ("Length generalisation (Pass 2)", "seq_len_nominal", "seq_len"),
              "pass3_control.csv": ("Control vs length (Pass 3)", "seq_len_nominal", "seq_len")}
    md = ["# iota — results across seeds", "",
          f"Runs: {', '.join(run_dirs)}. Per-query recall: **mean** [min–max across seeds], then each seed's value.",
          "One seed is not a result: r04 and r05 disagreed by up to 0.26 at high load on identical prompts.", ""]
    for f in PASS_FILES:
        by_run = {run: _read(os.path.join(d, f)) for run, d in run_dirs.items()}
        by_run = {r: rows for r, rows in by_run.items() if rows}
        if not by_run:
            continue
        rows = aggregate_pass(by_run)
        _write(os.path.join(out_dir, f), rows)
        result[f] = rows
        title, xkey, xlabel = titles[f]
        md += [f"## {title} — {len(by_run)} seed(s): {', '.join(by_run)}", ""] + _md_table(rows, xkey, xlabel)
    by_run = {run: _read(os.path.join(d, KEYLEN_FILE)) for run, d in run_dirs.items()}
    by_run = {r: rows for r, rows in by_run.items() if rows}
    if by_run:
        rows = aggregate_keylen(by_run)
        _write(os.path.join(out_dir, KEYLEN_FILE), rows)
        result[KEYLEN_FILE] = rows
    by_run = {run: _read(os.path.join(d, GRID_FILE)) for run, d in run_dirs.items()}
    by_run = {r: rows for r, rows in by_run.items() if rows}
    if by_run:  # same key fields as passes 1-3, so the same aggregation applies
        rows = aggregate_pass(by_run)
        _write(os.path.join(out_dir, GRID_FILE), rows)
        result[GRID_FILE] = rows
        md += [f"## Joint load × distance grid (Pass 5) — {len(by_run)} seed(s): {', '.join(by_run)}", ""]
        lens = sorted({int(float(r["seq_len_nominal"])) for r in rows})
        for m in [a for a in ARCHS if any(_arch(r["model"]) == a for r in rows)]:
            md += [f"**{m}**", "", "| bindings \\ tokens | " + " | ".join(map(str, lens)) + " |",
                   "|---:|" + "---|" * len(lens)]
            for nb in sorted({int(r["n_bindings"]) for r in rows}):
                cells = []
                for L in lens:
                    h = [r for r in rows if _arch(r["model"]) == m and int(r["n_bindings"]) == nb
                         and int(float(r["seq_len_nominal"])) == L]
                    cells.append("–" if not h or h[0]["accuracy_per_query"] is None else
                                 f"{h[0]['accuracy_per_query']:.3f} [{h[0]['ci_low_pq']:.2f}–{h[0]['ci_high_pq']:.2f}]")
                md.append(f"| {nb} | " + " | ".join(cells) + " |")
            md.append("")
    by_run = {run: _read(os.path.join(d, FREERUN_FILE)) for run, d in run_dirs.items()}
    by_run = {r: rows for r, rows in by_run.items() if rows}
    if by_run:
        cells = defaultdict(list)
        for run, rows in by_run.items():
            for r in rows:
                cells[(r["model"], int(r["n_bindings"]))].append(
                    (run, float(r["per_query_teacher_forced"]), float(r["per_query_free_running"])))
        agg = []
        for (model, nb), items in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            agg.append({"model": model, "n_bindings": nb,
                        "teacher_forced_mean": round(statistics.fmean(t for _, t, _ in items), 4),
                        "free_running_mean": round(statistics.fmean(f for _, _, f in items), 4),
                        "max_abs_gap": round(max(abs(t - f) for _, t, f in items), 4),
                        "n_seeds": len(items),
                        "per_seed": " ".join(f"{r}={t:.3f}/{f:.3f}" for r, t, f in items)})
        _write(os.path.join(out_dir, FREERUN_FILE), agg)
        result[FREERUN_FILE] = agg
        md += [f"## Free-running vs teacher-forced (Pass 6) — {len(by_run)} seed(s)", "",
               "| model | n_bindings | teacher-forced | free-running | max per-seed gap |",
               "|---|---:|---:|---:|---:|"]
        md += [f"| {_arch(r['model'])} | {r['n_bindings']} | {r['teacher_forced_mean']:.3f} | "
               f"{r['free_running_mean']:.3f} | {r['max_abs_gap']:.3f} |" for r in agg]
        md.append("")
    for f in COST_FILES:  # architecture-only: copy from the first run that has it
        for run, d in run_dirs.items():
            rows = _read(os.path.join(d, f))
            if rows:
                _write(os.path.join(out_dir, f), rows)
                result[f] = rows
                md += [f"_{f}: from {run} (cost depends on architecture only, not the seed)._", ""]
                break
    with open(os.path.join(out_dir, "SEEDS.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    if make_plot and any(f in result for f in PASS_FILES):
        from iota.plot import make_figure
        # Say how many seeds each panel actually has -- passes can be at different
        # stages (e.g. pass 2 still waiting on the newest run's plan B).
        counts = {f: max((r.get("n_seeds", 0) for r in result[f]), default=0)
                  for f in PASS_FILES if f in result}
        short = {"pass1_capacity.csv": "capacity", "pass2_length.csv": "length",
                 "pass3_control.csv": "control"}
        per = ", ".join(f"{short[f]} {n}" for f, n in counts.items())
        make_figure(results_dir=out_dir, out_png=os.path.join(out_dir, "money_figure_seeds.png"),
                    note=f"mean across seeds ({per}; runs {', '.join(run_dirs)}) · "
                         f"bands = min–max across seeds · decode cost is seed-independent")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description="aggregate seed-replicate runs into one figure")
    ap.add_argument("--runs", nargs="*", default=[], help="run ids on the kaggle-results branch")
    ap.add_argument("--ref", default="origin/kaggle-results")
    ap.add_argument("--dirs", nargs="*", default=[], help="local run directories instead")
    ap.add_argument("--out", default="figures/seeds")
    args = ap.parse_args()
    if args.dirs:
        run_dirs = {os.path.basename(os.path.normpath(d)): d for d in args.dirs}
    else:
        if not args.runs:
            ap.error("give --runs or --dirs")
        subprocess.run(["git", "fetch", "-q", "origin", args.ref.split("/", 1)[-1]], check=False)
        run_dirs = fetch_runs(args.runs, args.ref, tempfile.mkdtemp(prefix="iota-seeds-"))
    aggregate(run_dirs, args.out)
    print(f"wrote {args.out}/ (SEEDS.md, aggregated CSVs, money_figure_seeds.png)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
