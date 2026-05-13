"""A/B comparison between two SoTA-comparison runs.

Reads two ``summary.json`` files (e.g. ``results/sota_anytime/summary.json``
and ``results/sota_betting/summary.json``) and emits a single
Markdown table comparing DiSE's half-width / samples / wall / coverage
across the two configurations.

Usage::

    uv run python scripts/sota_ab.py \\
        --a results/sota_anytime/summary.json --a-label anytime \\
        --b results/sota_betting/summary.json --b-label betting \\
        --out results/sota_betting/ab_dise.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> list[dict[str, Any]]:
    with open(path) as f:
        return json.load(f)["benchmarks"]


def pivot_dise(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["benchmark"]: r for r in rows if r["method"] == "dise"}


def fmt_delta(a: float | None, b: float | None, fmt: str = "{:+.4f}") -> str:
    if a is None or b is None:
        return "—"
    return fmt.format(b - a)


def fmt_ratio(a: float | None, b: float | None) -> str:
    if a is None or b is None or a == 0:
        return "—"
    return f"{b/a:.2f}×"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--a-label", required=True)
    p.add_argument("--b-label", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    a = pivot_dise(load(args.a))
    b = pivot_dise(load(args.b))
    benches = sorted(set(a) & set(b))

    out = [
        f"# DiSE A/B: `method={args.a_label}` vs `method={args.b_label}`",
        "",
        f"| benchmark | half(`{args.a_label}`) | half(`{args.b_label}`) | Δhalf | b/a | "
        f"n(`{args.a_label}`) | n(`{args.b_label}`) | cov(`{args.a_label}`) | cov(`{args.b_label}`) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    def fmt_cov(c: float | None) -> str:
        return "—" if c is None else f"{c:.2f}"

    n_tighter = 0
    n_looser = 0
    n_same = 0
    for bn in benches:
        ra = a[bn]
        rb = b[bn]
        ha = ra["median_half_width"]
        hb = rb["median_half_width"]
        na = ra["median_samples"]
        nb = rb["median_samples"]
        ca = ra.get("coverage")
        cb = rb.get("coverage")
        if hb < ha - 1e-9:
            n_tighter += 1
            marker = "↓"
        elif hb > ha + 1e-9:
            n_looser += 1
            marker = "↑"
        else:
            n_same += 1
            marker = "="
        out.append(
            f"| `{bn}` | {ha:.4f} | {hb:.4f} {marker} | {fmt_delta(ha, hb)} | "
            f"{fmt_ratio(ha, hb)} | {na} | {nb} | "
            f"{fmt_cov(ca)} | {fmt_cov(cb)} |"
        )
    out += [
        "",
        f"**Summary**: ↓ tighter on **{n_tighter}**/{len(benches)} benchmarks · "
        f"↑ looser on **{n_looser}** · = unchanged on **{n_same}**.",
        "",
    ]
    Path(args.out).write_text("\n".join(out) + "\n")
    print(f"Wrote {args.out}")
    print(f"DiSE `{args.b_label}` is tighter on {n_tighter}/{len(benches)}, "
          f"looser on {n_looser}, unchanged on {n_same}.")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
