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
import statistics
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


def plot_decision_curve() -> None:
    """Samples-to-decision vs threshold-distance.

    The operational headline: for any contractual threshold $\\tau$
    in the neighbourhood of the truth $\\mu$, how many samples does
    each method need until the certified interval lies entirely on
    one side of $\\tau$?

    Wilson's MC interval gives $n \\sim \\mu(1-\\mu) / (\\tau-\\mu)^2$
    — diverges as $\\tau \\to \\mu$.  \\toolname{} on an axis-aligned
    benchmark gives an exact certificate; samples-to-decision is
    constant in $|\\tau - \\mu|$, equal to the bootstrap-plus-
    refinement cost.
    """
    summary_path = RESULTS_DIR / "decision.json"
    if not summary_path.exists():
        print("  decision.json not found; skipping decision-curve plot")
        return
    summary = json.loads(summary_path.read_text())
    rows = summary["rows"]
    meta = summary["metadata"]
    mu = meta["mu_truth"]
    max_budget = meta["max_budget"]
    delta = meta["delta"]

    methods = sorted({r["method"] for r in rows},
                     key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for method in methods:
        mrs = sorted([r for r in rows if r["method"] == method],
                     key=lambda r: abs(r["tau_minus_mu"]))
        if not mrs:
            continue
        # Symmetrise: |tau - mu| on x-axis, median samples on y; combine
        # both sides of mu by absolute distance.
        from collections import defaultdict
        bydist: dict[float, list[int]] = defaultdict(list)
        capped: dict[float, bool] = defaultdict(bool)
        for r in mrs:
            d = abs(r["tau_minus_mu"])
            bydist[d].append(r["median_samples"])
            if not r.get("all_decided", True):
                capped[d] = True
        xs = sorted(bydist.keys())
        ys = [statistics.median(bydist[d]) for d in xs]
        ax.plot(xs, ys,
                color=METHOD_COLOURS.get(method, "black"),
                marker=METHOD_MARKERS.get(method, "o"),
                label=method, linewidth=1.8, markersize=6)
        for x, y, _ in zip(xs, ys, xs):
            if capped[x]:
                ax.scatter([x], [y], s=140, facecolor="none",
                           edgecolor="black", linewidth=1.2, zorder=10)

    # Theoretical envelope: Wilson MC needs n ~ z^2 * mu(1-mu) / (tau-mu)^2.
    import numpy as _np
    z = 1.96
    dist_grid = _np.geomspace(min(abs(r["tau_minus_mu"]) for r in rows),
                              max(abs(r["tau_minus_mu"]) for r in rows), 50)
    env = z * z * mu * (1 - mu) / dist_grid ** 2
    ax.plot(dist_grid, env, ":", color="#444", lw=1.2,
            label=r"Wilson envelope $z^2\mu(1{-}\mu)/(\tau{-}\mu)^2$")
    ax.axhline(max_budget, color="black", linestyle="--", lw=0.8, alpha=0.5,
               label=f"max budget = {max_budget:,}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"$|\tau - \mu|$  (decision precision; smaller is harder)")
    ax.set_ylabel(r"samples to a $(1-\delta)$-certified decision")
    ax.set_title(
        r"Samples-to-decision on the SLA benchmark "
        rf"($\mu = {mu:.4f}$, $\delta = {delta}$)"
    )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left", frameon=True)
    fig.tight_layout()
    _save(fig, "07_decision_curve")


def plot_convergence_curve() -> None:
    """Half-width vs sample budget on a single fixed-mu benchmark.

    The visual story: \\toolname{} achieves half-width exactly zero
    at the budget threshold where refinement closes the partition,
    and stays at zero from then onward.  Plain MC's certified half-
    width decays as $\\Theta(1/\\sqrt{n})$ and never reaches zero.

    For the log y-axis, exact-zero half-widths are replotted at a
    small floor (1e-5) and overlaid with a marker indicating "exact".
    """
    summary_path = RESULTS_DIR / "convergence.json"
    if not summary_path.exists():
        print("  convergence.json not found; skipping")
        return
    summary = json.loads(summary_path.read_text())
    rows = summary["rows"]
    meta = summary["metadata"]

    methods = sorted({r["method"] for r in rows},
                     key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99)
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    floor = 1e-5  # plot floor for hw = 0

    for method in methods:
        mrs = sorted([r for r in rows if r["method"] == method],
                     key=lambda r: r["budget"])
        if not mrs:
            continue
        xs = [r["budget"] for r in mrs]
        ys = [max(r["median_half_width"], floor) for r in mrs]
        exact = [r["median_half_width"] == 0.0 for r in mrs]
        ax.plot(xs, ys,
                color=METHOD_COLOURS.get(method, "black"),
                marker=METHOD_MARKERS.get(method, "o"),
                label=method, linewidth=1.8, markersize=6)
        # Mark "exact zero" points with a special star overlay so the
        # reader sees that the y-coordinate is a plotting artifact.
        for x, y, e in zip(xs, ys, exact):
            if e:
                ax.scatter([x], [y], s=180, marker="*",
                           facecolor=METHOD_COLOURS.get(method, "black"),
                           edgecolor="black", linewidth=1.0, zorder=10)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(floor, color="black", linestyle=":", lw=0.7, alpha=0.6)
    ax.set_xlabel(r"sample budget  $n$")
    ax.set_ylabel(r"median certified half-width  (log scale)")
    ax.set_title(
        r"Convergence on the threshold benchmark "
        rf"($\mu = {meta['mu_truth']:.4f}$, $\delta = {meta['delta']}$)"
    )
    ax.grid(True, alpha=0.3, which="both")
    # Star marker = exact-zero certificate (the y-coordinate is plot floor).
    from matplotlib.lines import Line2D
    star_handle = Line2D([0], [0], marker="*", linestyle="",
                          markerfacecolor=METHOD_COLOURS.get("dise_betting", "red"),
                          markeredgecolor="black", markersize=10,
                          label="hw $=0$ (plot floor)")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [star_handle], labels + [star_handle.get_label()],
              fontsize=9, loc="lower left", frameon=True)
    fig.tight_layout()
    _save(fig, "06_convergence_curve")


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
        plot_decision_curve()
        plot_rare_event_scaling()
        plot_convergence_curve()
        return 0

    summary = _load_summary()
    print(f"# {len(summary['summary_rows'])} summary rows; generating figures")

    plot_decision_curve()
    plot_rare_event_scaling()
    plot_convergence_curve()
    plot_halfwidth_vs_budget(summary)
    plot_sample_efficiency(summary)
    plot_coverage(summary)
    plot_wallclock(summary)
    plot_error(summary)
    print(f"# figures in {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
