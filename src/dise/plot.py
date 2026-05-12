"""Plot helpers — used by ``dise plot``.

Requires ``matplotlib``. We keep this module a thin layer over
:func:`run` so the CLI doesn't import matplotlib unless needed.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def _load(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _plot_compare(report: dict[str, Any], out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({a["method"] for a in report["aggregates"]})
    half_widths = [
        next(a["median_half_width"] for a in report["aggregates"] if a["method"] == m)
        for m in methods
    ]
    coverages = [
        (
            next(a["coverage"] for a in report["aggregates"] if a["method"] == m)
            or 0.0
        )
        * 100.0
        for m in methods
    ]
    samples = [
        next(a["median_samples"] for a in report["aggregates"] if a["method"] == m)
        for m in methods
    ]

    fig, axs = plt.subplots(1, 3, figsize=(15, 4))

    axs[0].bar(methods, half_widths, color="steelblue")
    axs[0].set_title(f"Median half-width\n{report['benchmark']}")
    axs[0].set_ylabel("half-width")
    axs[0].tick_params(axis="x", rotation=20)

    axs[1].bar(methods, samples, color="seagreen")
    axs[1].set_title("Median samples used")
    axs[1].set_ylabel("samples")
    axs[1].tick_params(axis="x", rotation=20)

    axs[2].bar(methods, coverages, color="indianred")
    axs[2].axhline(95.0, color="black", linestyle=":", lw=1, label="1 - delta = 95%")
    axs[2].set_title("Coverage (interval contains MC truth)")
    axs[2].set_ylabel("coverage (%)")
    axs[2].set_ylim(0, 105)
    axs[2].legend(fontsize=8)
    axs[2].tick_params(axis="x", rotation=20)

    fig.tight_layout()
    fig.savefig(out, dpi=150)


def _plot_convergence(report: dict[str, Any], out: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    methods = sorted({r["method"] for r in report["runs"]})
    for m in methods:
        runs = [r for r in report["runs"] if r["method"] == m]
        runs.sort(key=lambda r: r["samples_used"])
        xs = [r["samples_used"] for r in runs]
        ys = [r["half_width"] for r in runs]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=m)
    ax.set_xlabel("samples used")
    ax.set_ylabel("half-width")
    ax.set_title(f"Per-seed convergence — {report['benchmark']}")
    ax.legend()
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def run(args: argparse.Namespace) -> int:
    report = _load(args.report)
    if args.kind == "compare":
        _plot_compare(report, args.out)
    elif args.kind == "convergence":
        _plot_convergence(report, args.out)
    else:
        raise ValueError(args.kind)
    print(f"wrote {args.out}")
    return 0
