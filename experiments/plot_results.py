"""Generate the figures for EXPERIMENTAL_STUDY.md from the JSON summary.

Reads:
  experiments/results/summary.json
  experiments/results/runs.jsonl   (for the per-row plots that need raw data)

Writes (PDF + PNG side by side; PDF for paper, PNG for the markdown):
  experiments/figures/01_halfwidth_vs_budget.{pdf,png}
  experiments/figures/02_sample_efficiency.{pdf,png}
  experiments/figures/03_coverage.{pdf,png}
  experiments/figures/04_wallclock.{pdf,png}
  experiments/figures/05_error_vs_truth.{pdf,png}
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "experiments" / "results"
FIG_DIR = ROOT / "experiments" / "figures"


METHOD_ORDER = [
    "plain_mc",
    "plain_mc_hoeffding",
    "plain_mc_eb",
    "plain_mc_betting",
    "quasi_mc_sobol",
    "stratified_random",
    "adaptive_stratified",
    "dise_wilson",
    "dise_anytime",
    "dise_betting",
]

METHOD_COLOURS = {
    "plain_mc":            "#9c9c9c",  # grey
    "plain_mc_hoeffding":  "#aec7e8",  # light blue
    "plain_mc_eb":         "#ffbb78",  # light orange
    "plain_mc_betting":    "#98df8a",  # light green
    "quasi_mc_sobol":      "#c5b0d5",  # light purple
    "stratified_random":   "#bcbd22",  # mustard
    "adaptive_stratified": "#8c564b",  # brown
    "dise_wilson":         "#1f77b4",  # blue
    "dise_anytime":        "#2ca02c",  # green
    "dise_betting":        "#d62728",  # red
}

METHOD_MARKERS = {
    "plain_mc":            "o",
    "plain_mc_hoeffding":  "o",
    "plain_mc_eb":         "o",
    "plain_mc_betting":    "o",
    "quasi_mc_sobol":      "*",
    "stratified_random":   "s",
    "adaptive_stratified": "P",
    "dise_wilson":         "^",
    "dise_anytime":        "v",
    "dise_betting":        "D",
}


def _short(name: str) -> str:
    """Compact benchmark label for plot titles / x-axes."""
    s = name.replace("_BG(", "/BG(").replace("_U(", "/U(")
    if len(s) > 26:
        s = s[:23] + "..."
    return s


def _load_summary() -> dict[str, Any]:
    return json.loads((RESULTS_DIR / "summary.json").read_text())


def _load_runs() -> list[dict[str, Any]]:
    runs = []
    with open(RESULTS_DIR / "runs.jsonl") as fh:
        for line in fh:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIG_DIR / f"{name}.pdf"
    png_path = FIG_DIR / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"  wrote {pdf_path.name} and {png_path.name}")


# ---------------------------------------------------------------------------
# 1. Half-width vs budget (one panel per benchmark)
# ---------------------------------------------------------------------------


def plot_halfwidth_vs_budget(summary: dict[str, Any]) -> None:
    rows = summary["summary_rows"]
    by_bench: dict[str, list] = defaultdict(list)
    for r in rows:
        by_bench[r["benchmark"]].append(r)

    benchmarks = sorted(by_bench.keys())
    n = len(benchmarks)
    cols = 3 if n >= 3 else n
    rows_n = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows_n, cols, figsize=(cols * 4.5, rows_n * 3.2),
                            squeeze=False)

    for i, bench in enumerate(benchmarks):
        ax = axs[i // cols][i % cols]
        rs = by_bench[bench]
        methods = sorted({r["method"] for r in rs},
                         key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
        for method in methods:
            mrs = sorted([r for r in rs if r["method"] == method],
                         key=lambda r: r["budget"])
            xs = [r["budget"] for r in mrs]
            ys = [r["median_half_width"] for r in mrs]
            ax.plot(xs, ys,
                    color=METHOD_COLOURS.get(method, "black"),
                    marker=METHOD_MARKERS.get(method, "o"),
                    label=method, linewidth=1.5, markersize=5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(_short(bench), fontsize=9)
        ax.set_xlabel("budget (samples)")
        ax.set_ylabel("median half-width")
        ax.grid(True, alpha=0.3, which="both")

    # Hide unused panels
    for j in range(n, rows_n * cols):
        axs[j // cols][j % cols].axis("off")

    handles, labels = axs[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               bbox_to_anchor=(0.5, -0.02), fontsize=9, frameon=False)
    fig.suptitle("Certified half-width vs.\\ sample budget", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    _save(fig, "01_halfwidth_vs_budget")


# ---------------------------------------------------------------------------
# 2. Sample efficiency: half-width at max budget, per method × benchmark
# ---------------------------------------------------------------------------


def plot_sample_efficiency(summary: dict[str, Any]) -> None:
    rows = summary["summary_rows"]
    max_budget = max(r["budget"] for r in rows)
    rows_max = [r for r in rows if r["budget"] == max_budget]

    benchmarks = sorted({r["benchmark"] for r in rows_max})
    methods = sorted({r["method"] for r in rows_max},
                     key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)

    by_bm = {(r["benchmark"], r["method"]): r["median_half_width"] for r in rows_max}

    fig, ax = plt.subplots(figsize=(max(6.0, len(benchmarks) * 0.8), 4.5))
    width = 0.85 / len(methods)
    x = list(range(len(benchmarks)))
    for j, method in enumerate(methods):
        ys = [by_bm.get((b, method), float("nan")) for b in benchmarks]
        offsets = [xi - 0.425 + (j + 0.5) * width for xi in x]
        ax.bar(offsets, ys, width=width,
               color=METHOD_COLOURS.get(method, "black"),
               label=method, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([_short(b) for b in benchmarks], rotation=30, ha="right",
                       fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("median half-width  (log scale)")
    ax.set_title(f"Half-width at budget = {max_budget} samples")
    ax.grid(True, alpha=0.3, which="both", axis="y")
    ax.legend(fontsize=9, ncol=len(methods), loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    _save(fig, "02_sample_efficiency")


# ---------------------------------------------------------------------------
# 3. Coverage: empirical (1-delta) coverage rate per method per benchmark
# ---------------------------------------------------------------------------


def plot_coverage(summary: dict[str, Any]) -> None:
    rows = summary["summary_rows"]
    max_budget = max(r["budget"] for r in rows)
    rows_max = [r for r in rows if r["budget"] == max_budget]
    delta = summary.get("metadata", {}).get("delta", 0.05)

    benchmarks = sorted({r["benchmark"] for r in rows_max})
    methods = sorted({r["method"] for r in rows_max},
                     key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    by_bm = {(r["benchmark"], r["method"]): r["coverage"] for r in rows_max}

    fig, ax = plt.subplots(figsize=(max(6.0, len(benchmarks) * 0.8), 4.5))
    width = 0.85 / len(methods)
    x = list(range(len(benchmarks)))
    for j, method in enumerate(methods):
        ys = [(by_bm.get((b, method)) or 0.0) * 100 for b in benchmarks]
        offsets = [xi - 0.425 + (j + 0.5) * width for xi in x]
        ax.bar(offsets, ys, width=width,
               color=METHOD_COLOURS.get(method, "black"),
               label=method, edgecolor="white", linewidth=0.3)
    target_label = "target  $1-\\delta$ = {:.0%}".format(1 - delta)
    ax.axhline(100 * (1 - delta), color="black", linestyle=":", lw=1,
               label=target_label)
    ax.set_xticks(x)
    ax.set_xticklabels([_short(b) for b in benchmarks], rotation=30, ha="right",
                       fontsize=8)
    ax.set_ylabel("empirical coverage (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Coverage at budget = {max_budget}")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9, ncol=len(methods) + 1, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    _save(fig, "03_coverage")


# ---------------------------------------------------------------------------
# 4. Wall-clock per method per benchmark (at the largest budget)
# ---------------------------------------------------------------------------


def plot_wallclock(summary: dict[str, Any]) -> None:
    rows = summary["summary_rows"]
    max_budget = max(r["budget"] for r in rows)
    rows_max = [r for r in rows if r["budget"] == max_budget]

    benchmarks = sorted({r["benchmark"] for r in rows_max})
    methods = sorted({r["method"] for r in rows_max},
                     key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    by_bm = {(r["benchmark"], r["method"]): r["median_wall_clock_s"] for r in rows_max}

    fig, ax = plt.subplots(figsize=(max(6.0, len(benchmarks) * 0.8), 4.5))
    width = 0.85 / len(methods)
    x = list(range(len(benchmarks)))
    for j, method in enumerate(methods):
        ys = [max(by_bm.get((b, method), 1e-6), 1e-6) for b in benchmarks]
        offsets = [xi - 0.425 + (j + 0.5) * width for xi in x]
        ax.bar(offsets, ys, width=width,
               color=METHOD_COLOURS.get(method, "black"),
               label=method, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([_short(b) for b in benchmarks], rotation=30, ha="right",
                       fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("median wall-clock  (s, log scale)")
    ax.set_title(f"Wall-clock at budget = {max_budget}")
    ax.grid(True, alpha=0.3, which="both", axis="y")
    ax.legend(fontsize=9, ncol=len(methods), loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    _save(fig, "04_wallclock")


# ---------------------------------------------------------------------------
# 5. Point-estimate error: |mu_hat - mu_truth| per method × benchmark
# ---------------------------------------------------------------------------


def plot_error(summary: dict[str, Any]) -> None:
    rows = summary["summary_rows"]
    max_budget = max(r["budget"] for r in rows)
    rows_max = [r for r in rows if r["budget"] == max_budget]

    benchmarks = sorted({r["benchmark"] for r in rows_max})
    methods = sorted({r["method"] for r in rows_max},
                     key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    by_bm = {(r["benchmark"], r["method"]): r.get("median_error_vs_truth") for r in rows_max}

    fig, ax = plt.subplots(figsize=(max(6.0, len(benchmarks) * 0.8), 4.5))
    width = 0.85 / len(methods)
    x = list(range(len(benchmarks)))
    for j, method in enumerate(methods):
        ys = [max(by_bm.get((b, method)) or 1e-6, 1e-6) for b in benchmarks]
        offsets = [xi - 0.425 + (j + 0.5) * width for xi in x]
        ax.bar(offsets, ys, width=width,
               color=METHOD_COLOURS.get(method, "black"),
               label=method, edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([_short(b) for b in benchmarks], rotation=30, ha="right",
                       fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("median $|\\hat\\mu - \\mu|$  (log scale)")
    ax.set_title(f"Point-estimate error at budget = {max_budget}")
    ax.grid(True, alpha=0.3, which="both", axis="y")
    ax.legend(fontsize=9, ncol=len(methods), loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=False)
    fig.tight_layout()
    _save(fig, "05_error_vs_truth")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def plot_rare_event_scaling() -> None:
    """The headline log-log: samples-to-epsilon vs true rare-event mass.

    Theory predicts plain MC has slope -1 (samples ~ Theta(1/mu)) and
    DiSE has slope 0 (samples ~ Theta(1)).  The figure makes that
    contrast visible at a glance.
    """
    summary_path = RESULTS_DIR / "rare_event.json"
    if not summary_path.exists():
        print(f"  rare_event.json not found; skipping headline figure")
        return
    summary = json.loads(summary_path.read_text())
    rows = summary["rows"]
    methods = sorted({r["method"] for r in rows},
                     key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    eps = summary["metadata"]["epsilon"]
    max_budget = summary["metadata"]["max_budget"]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for method in methods:
        mrs = sorted([r for r in rows if r["method"] == method],
                     key=lambda r: r["mu_truth"])
        if not mrs:
            continue
        xs = [r["mu_truth"] for r in mrs]
        ys = [r["median_samples"] for r in mrs]
        hit_budget = [r["any_hit_budget"] for r in mrs]
        ax.plot(xs, ys,
                color=METHOD_COLOURS.get(method, "black"),
                marker=METHOD_MARKERS.get(method, "o"),
                label=method, linewidth=2.0, markersize=7)
        # Mark budget-capped points with an open ring overlay.
        for x, y, hit in zip(xs, ys, hit_budget):
            if hit:
                ax.scatter([x], [y], s=120, facecolor="none",
                           edgecolor="black", linewidth=1.2, zorder=10)

    # Reference line: samples ~ log(2/delta) / (2 * mu * eps^2) is the
    # Hoeffding-budget envelope for plain MC at rare events.  Plot it
    # as a guide showing the slope -1 that MC must trace.
    import numpy as _np
    mu_grid = _np.geomspace(min(r["mu_truth"] for r in rows),
                             max(r["mu_truth"] for r in rows), 50)
    log_term = math.log(2.0 / summary["metadata"]["delta"])
    hoeffding_curve = log_term / (2.0 * mu_grid * eps ** 2)
    ax.plot(mu_grid, hoeffding_curve, ":", color="#444", lw=1.2,
            label="Hoeffding $1 / (2\\mu\\varepsilon^2)$")

    ax.axhline(max_budget, color="black", linestyle="--", lw=0.8, alpha=0.5,
               label=f"max budget = {max_budget:,}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"true rare-event mass  $\mu$")
    ax.set_ylabel(r"samples to certify hw $\leq$ "
                  + f"{eps}" + r" at $\delta=$" + f"{summary['metadata']['delta']}")
    ax.set_title(r"Rare-event scaling: samples-to-$\varepsilon$ vs.\ $\mu$")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, ncol=2, loc="best", frameon=True)
    fig.tight_layout()
    _save(fig, "00_rare_event_scaling")


def main() -> int:
    global RESULTS_DIR, FIG_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out-dir", default=str(FIG_DIR))
    parser.add_argument("--rare-event-only", action="store_true",
                        help="only regenerate the rare-event scaling figure")
    args = parser.parse_args()
    RESULTS_DIR = Path(args.results_dir)
    FIG_DIR = Path(args.out_dir)

    if args.rare_event_only:
        plot_rare_event_scaling()
        return 0

    summary = _load_summary()
    print(f"# {len(summary['summary_rows'])} summary rows; generating figures")

    plot_rare_event_scaling()
    plot_halfwidth_vs_budget(summary)
    plot_sample_efficiency(summary)
    plot_coverage(summary)
    plot_wallclock(summary)
    plot_error(summary)
    print(f"# figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
