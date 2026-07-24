"""The money figure (Phase 8): accuracy vs recall load and length, plus cost.

Three panels, each answering one clause of the thesis:

  A  capacity (headline) -- per-query recall accuracy vs n_bindings. The crossover:
     does the fixed-state model fall away as recall load grows while attention holds?
  B  length-generalisation, with the state_track CONTROL overlaid dashed. The control
     is what makes panel A mean something: if linear matches dense on the control but
     not on recall, the gap is specifically ASSOCIATIVE RECALL and not a general
     inability to handle long inputs.
  C  cost -- peak VRAM vs sequence length, with OOM marked. A hybrid that recovers
     dense accuracy only matters if it is cheaper.

Design rules (from the dataviz method):
  * ONE axis per panel -- cost gets its own panel, never a second y-scale on accuracy.
  * Categorical colour assigned by identity in fixed order, never cycled; palette
    validated for colour-vision deficiency (all-pairs) before use.
  * Distinct markers per series as secondary encoding, so identity never rests on
    colour alone; legend plus direct labels at the line ends.
  * Recessive grid and axes; thin marks.

    python -m iota.plot [--results_dir experiments/results] [--dark]
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots 1-3 of the validated palette (the first three clear the
# all-pairs CVD and normal-vision floors). Assigned by identity, never by rank.
SERIES = {
    "transformer":  {"label": "Transformer (dense)",   "light": "#2a78d6", "dark": "#3987e5", "marker": "o"},
    "gated_linear": {"label": "Gated linear (fixed state)", "light": "#eb6834", "dark": "#d95926", "marker": "s"},
    "hybrid":       {"label": "Hybrid",                "light": "#1baf7a", "dark": "#199e70", "marker": "^"},
}
ORDER = ["transformer", "gated_linear", "hybrid"]

THEME = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "grid": "#d9d8d4"},
    "dark":  {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "grid": "#3a3a38"},
}

# A 2-digit answer drawn at random is right ~1/100 of the time. Anything hugging
# this line has learned nothing, which is exactly how the dead control read.
CHANCE = 0.01


def _arch_of(model_name: str) -> str:
    """'gated_linear_sweep' -> 'gated_linear' (checkpoint names carry a suffix)."""
    for suffix in ("_sweep", "_milestone"):
        if model_name.endswith(suffix):
            model_name = model_name[: -len(suffix)]
    return model_name


def _f(v) -> Optional[float]:
    """CSV cell -> float, tolerating '' and 'None' (OOM rows)."""
    if v is None or v == "" or v == "None":
        return None
    try:
        return float(v)
    except ValueError:
        return None


def load_csv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _series(rows: List[Dict], xkey: str, ykey: str, lokey: str, hikey: str):
    """-> {arch: (xs, ys, los, his)} sorted by x, dropping rows with no y (OOM)."""
    out = defaultdict(list)
    for r in rows:
        x, y = _f(r.get(xkey)), _f(r.get(ykey))
        if x is None or y is None:
            continue
        out[_arch_of(r["model"])].append((x, y, _f(r.get(lokey)), _f(r.get(hikey))))
    packed = {}
    for arch, pts in out.items():
        pts.sort(key=lambda p: p[0])
        packed[arch] = ([p[0] for p in pts], [p[1] for p in pts],
                        [p[2] for p in pts], [p[3] for p in pts])
    return packed


def _style_axes(ax, th, xlabel, ylabel, title, subtitle=None):
    ax.set_facecolor(th["surface"])
    ax.set_title(title, color=th["ink"], fontsize=11, fontweight="bold", loc="left", pad=14)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=th["ink2"],
                fontsize=8.5, va="bottom")
    ax.set_xlabel(xlabel, color=th["ink2"], fontsize=9)
    ax.set_ylabel(ylabel, color=th["ink2"], fontsize=9)
    ax.tick_params(colors=th["ink2"], labelsize=8.5, length=3)
    ax.grid(True, color=th["grid"], linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(th["grid"])


def _plot_accuracy(ax, data, th, dashed=False, label_lines=True, alpha_band=0.15):
    """Draw one accuracy series-set; returns handles for the legend."""
    handles = []
    for arch in ORDER:
        if arch not in data:
            continue
        xs, ys, los, his = data[arch]
        spec = SERIES[arch]
        color = spec[th["_mode"]]
        (line,) = ax.plot(xs, ys, color=color, linewidth=2.0,
                          marker=spec["marker"], markersize=5.5,
                          linestyle="--" if dashed else "-",
                          markeredgecolor=th["surface"], markeredgewidth=1.0,
                          label=spec["label"], zorder=3)
        if not dashed:
            handles.append(line)
        if all(l is not None and h is not None for l, h in zip(los, his)):
            ax.fill_between(xs, los, his, color=color, alpha=alpha_band,
                            linewidth=0, zorder=2)
        # Direct label at the line end -- required relief for the low-contrast slot,
        # and it keeps identity off colour alone.
        if label_lines and xs:
            ax.annotate(spec["label"].split(" (")[0], (xs[-1], ys[-1]),
                        textcoords="offset points", xytext=(6, 0), va="center",
                        fontsize=8, color=th["ink2"], zorder=4)
    return handles


def _chance_line(ax, th, xs):
    if not xs:
        return
    ax.axhline(CHANCE, color=th["ink2"], linewidth=1.0, linestyle=":", alpha=0.7, zorder=1)
    ax.annotate("chance (~1%)", (xs[0], CHANCE), textcoords="offset points",
                xytext=(2, 4), fontsize=7.5, color=th["ink2"])


def make_figure(
    results_dir: str = "experiments/results",
    out_png: Optional[str] = None,
    dark: bool = False,
) -> Optional[str]:
    """Build the figure from whatever passes exist; skip panels with no data."""
    mode = "dark" if dark else "light"
    th = dict(THEME[mode])
    th["_mode"] = mode

    p1 = load_csv(os.path.join(results_dir, "pass1_capacity.csv"))
    p2 = load_csv(os.path.join(results_dir, "pass2_length.csv"))
    p3 = load_csv(os.path.join(results_dir, "pass3_control.csv"))
    cost = load_csv(os.path.join(results_dir, "cost_profile.csv"))

    panels = []
    if p1:
        panels.append("capacity")
    if p2 or p3:
        panels.append("length")
    if cost:
        panels.append("cost")
    if not panels:
        print(f"no result CSVs in {results_dir}/ -- run the eval and profile stages first")
        return None
    missing = [n for n, d in (("pass1_capacity", p1), ("pass2_length", p2),
                              ("pass3_control", p3), ("cost_profile", cost)) if not d]
    if missing:
        print(f"[plot] note: no data for {', '.join(missing)} -- those panels are omitted")

    fig, axes = plt.subplots(1, len(panels), figsize=(5.4 * len(panels), 4.4))
    if len(panels) == 1:
        axes = [axes]
    fig.patch.set_facecolor(th["surface"])
    ax_of = dict(zip(panels, axes))

    # --- A: capacity (headline) -------------------------------------------------
    if "capacity" in panels:
        ax = ax_of["capacity"]
        data = _series(p1, "n_bindings", "accuracy_per_query", "ci_low_pq", "ci_high_pq")
        _plot_accuracy(ax, data, th)
        xs = sorted({_f(r["n_bindings"]) for r in p1 if _f(r["n_bindings"]) is not None})
        _chance_line(ax, th, xs)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{int(x)}" for x in xs])
        ax.set_ylim(-0.03, 1.03)
        _style_axes(ax, th, "recall load  (n_bindings, log scale)",
                    "per-query accuracy",
                    "A · Capacity",
                    "multi-query recall, sequence length held in-distribution")

    # --- B: length-generalisation + control -------------------------------------
    if "length" in panels:
        ax = ax_of["length"]
        xs_all = []
        if p2:
            d2 = _series(p2, "seq_len_true_tokens", "accuracy_per_query",
                         "ci_low_pq", "ci_high_pq")
            # No direct labels here: the curves converge at the right edge, so
            # end-of-line labels pile into an unreadable smear. The legend carries
            # identity, and markers keep it off colour alone.
            _plot_accuracy(ax, d2, th, label_lines=False)
            xs_all += [x for v in d2.values() for x in v[0]]
        if p3:
            d3 = _series(p3, "seq_len_true_tokens", "accuracy_per_query",
                         "ci_low_pq", "ci_high_pq")
            # control drawn dashed and unlabelled -- it is the reference, not a 4th series
            _plot_accuracy(ax, d3, th, dashed=True, label_lines=False, alpha_band=0.08)
            xs_all += [x for v in d3.values() for x in v[0]]
        _chance_line(ax, th, sorted(xs_all))
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.03, 1.03)
        sub = "solid: associative recall" + ("   ·   dashed: state-track control" if p3 else "")
        _style_axes(ax, th, "sequence length  (true tokens, log scale)",
                    "per-query accuracy", "B · Length generalisation", sub)

    # --- C: cost ----------------------------------------------------------------
    if "cost" in panels:
        ax = ax_of["cost"]
        has_vram = any(_f(r.get("peak_vram_mb")) is not None for r in cost)
        ykey = "peak_vram_mb" if has_vram else "latency_ms"
        ylabel = "peak VRAM (MB)" if has_vram else "forward latency (ms)"
        data = _series(cost, "seq_len", ykey, "_none", "_none")
        for arch in ORDER:
            if arch not in data:
                continue
            xs, ys, _, _ = data[arch]
            spec = SERIES[arch]
            ax.plot(xs, ys, color=spec[mode], linewidth=2.0, marker=spec["marker"],
                    markersize=5.5, markeredgecolor=th["surface"], markeredgewidth=1.0,
                    label=spec["label"], zorder=3)
        # OOM is a result: mark the FIRST length at which each model stops being
        # runnable at all (one marker per architecture, not just the first found).
        first_oom = {}
        for r in cost:
            if str(r.get("oom", "")).lower() in ("true", "1"):
                arch, x = _arch_of(r["model"]), _f(r.get("seq_len"))
                if arch in SERIES and x is not None and arch not in first_oom:
                    first_oom[arch] = x
        for arch, x in first_oom.items():
            ax.axvline(x, color=SERIES[arch][mode], linewidth=1.2,
                       linestyle=":", alpha=0.9, zorder=1)
            # Blended transform: x in data coords (pinned to the line), y in axes
            # fraction. Placed mid-height and to the LEFT of the rule, so it stays
            # inside the panel even when the OOM lands on the last x-tick.
            ax.text(x, 0.5, f"{SERIES[arch]['label'].split(' (')[0]} OOM",
                    transform=ax.get_xaxis_transform(), rotation=90,
                    fontsize=7.5, color=SERIES[arch][mode], va="center", ha="right",
                    bbox=dict(facecolor=th["surface"], edgecolor="none",
                              pad=1.5, alpha=0.85), zorder=5)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        _style_axes(ax, th, "sequence length (tokens, log scale)", ylabel,
                    "C · Cost", "forward pass, batch 1")

    # Legend: always present for >=2 series, so identity never rests on colour.
    handles = [plt.Line2D([], [], color=SERIES[a][mode], marker=SERIES[a]["marker"],
                          linewidth=2.0, markersize=5.5, label=SERIES[a]["label"])
               for a in ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=len(ORDER), frameon=False,
               fontsize=9, labelcolor=th["ink2"], bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))

    out_png = out_png or os.path.join(results_dir, f"money_figure{'_dark' if dark else ''}.png")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=200, facecolor=th["surface"], bbox_inches="tight")
    pdf = os.path.splitext(out_png)[0] + ".pdf"
    fig.savefig(pdf, facecolor=th["surface"], bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_png}\nwrote {pdf}")
    return out_png


def main() -> int:
    ap = argparse.ArgumentParser(description="iota money figure")
    ap.add_argument("--results_dir", default="experiments/results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dark", action="store_true", help="also/instead render dark mode")
    args = ap.parse_args()
    make_figure(results_dir=args.results_dir, out_png=args.out, dark=args.dark)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
