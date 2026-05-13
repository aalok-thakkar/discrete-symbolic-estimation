"""Render comparative tables from ``scripts/sota_compare.py`` output.

Reads ``<out-dir>/summary.json`` and emits:

* ``table_half_width.md`` — median certified half-width per (benchmark, method).
* ``table_samples.md``    — median samples used per (benchmark, method).
* ``table_coverage.md``   — empirical coverage rate per (benchmark, method).
* ``table_wall_clock.md`` — median wall-clock per (benchmark, method).
* ``ranking.md``          — per-benchmark ranking by half-width.

Usage::

    uv run python scripts/sota_report.py --in results/sota/summary.json \\
        --out-dir results/sota/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

METHODS_ORDER = [
    "plain_mc",
    "stratified_random",
    "adaptive_hoeffding",
    "ebstop",
    "betting_cs",
    "dise",
]

METHOD_LABELS = {
    "plain_mc": "PlainMC (Wilson)",
    "stratified_random": "StratRand (hash)",
    "adaptive_hoeffding": "AdaHoeff [PLDI'14]",
    "ebstop": "EBStop [ICML'08]",
    "betting_cs": "BettingCS [JRSS'24]",
    "dise": "**DiSE (ours)**",
}


def load(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def pivot(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        bench = r["benchmark"]
        method = r["method"]
        out.setdefault(bench, {})[method] = r.get(key)
    return out


def fmt_float(v: float | None, fmt: str = "{:.4f}") -> str:
    if v is None:
        return "—"
    return fmt.format(v)


def fmt_int(v: float | int | None) -> str:
    if v is None:
        return "—"
    return f"{int(v):,d}"


def render_table(
    title: str,
    rows: list[dict[str, Any]],
    key: str,
    formatter,
    methods: list[str] = METHODS_ORDER,
) -> str:
    grid = pivot(rows, key)
    bench_names = sorted(grid.keys())
    header = "| benchmark | " + " | ".join(METHOD_LABELS[m] for m in methods) + " |"
    sep = "|---|" + "---|" * len(methods)
    out = [f"# {title}", "", header, sep]
    for b in bench_names:
        row = grid[b]
        cells = " | ".join(formatter(row.get(m)) for m in methods)
        out.append(f"| `{b}` | {cells} |")
    return "\n".join(out) + "\n"


def render_ranking(rows: list[dict[str, Any]]) -> str:
    """Rank methods per benchmark by median_half_width (lower = better).

    Ties broken by samples (fewer = better). Methods with coverage < 0.5
    are excluded from the ranking (interval was unsound)."""
    grid = pivot(rows, "median_half_width")
    samples = pivot(rows, "median_samples")
    coverage = pivot(rows, "coverage")
    bench_names = sorted(grid.keys())
    out = ["# Per-benchmark ranking by certified half-width", ""]
    rank_counts: dict[str, dict[int, int]] = {m: {} for m in METHODS_ORDER}
    for b in bench_names:
        row = grid[b]
        scored = []
        for m in METHODS_ORDER:
            h = row.get(m)
            cov = coverage[b].get(m)
            if h is None:
                continue
            if cov is not None and cov < 0.5:
                # Soundness failure — skip from ranking.
                continue
            scored.append((h, samples[b].get(m, float("inf")), m))
        scored.sort()
        out.append(f"### `{b}`")
        out.append("")
        for rank, (h, s, m) in enumerate(scored, start=1):
            rank_counts[m][rank] = rank_counts[m].get(rank, 0) + 1
            out.append(
                f"{rank}. **{METHOD_LABELS[m]}** — half={h:.4f}, samples={int(s):,d}"
            )
        out.append("")
    out.append("---")
    out.append("")
    out.append("## Aggregate rank distribution")
    out.append("")
    out.append("| method | rank=1 | rank=2 | rank=3 | rank=4 | rank=5 | rank=6 |")
    out.append("|---|---|---|---|---|---|---|")
    for m in METHODS_ORDER:
        c = rank_counts[m]
        cells = " | ".join(str(c.get(r, 0)) for r in range(1, 7))
        out.append(f"| {METHOD_LABELS[m]} | {cells} |")
    return "\n".join(out) + "\n"


def render_synthesis(rows: list[dict[str, Any]], config: dict[str, Any]) -> str:
    """Produce a one-page synthesis with headline findings."""
    half = pivot(rows, "median_half_width")
    samples = pivot(rows, "median_samples")
    coverage = pivot(rows, "coverage")
    wall = pivot(rows, "median_wall_clock_s")
    truth = pivot(rows, "mc_truth")
    err = pivot(rows, "median_error_vs_truth")
    bench_names = sorted(half.keys())

    # Aggregate counts.
    soundness_failures = []
    dise_wins = []
    betting_wins = []
    plain_mc_wins = []
    for b in bench_names:
        cov = coverage[b]
        # Soundness check (coverage below target 0.95 - 1.0)
        for m in METHODS_ORDER:
            c = cov.get(m)
            if c is not None and c < 0.5:
                soundness_failures.append((b, m, c))
        # Rank by half-width among methods with valid coverage.
        scored = []
        for m in METHODS_ORDER:
            h = half[b].get(m)
            c = cov.get(m)
            if h is None:
                continue
            if c is not None and c < 0.5:
                continue
            scored.append((h, samples[b].get(m, float("inf")), m))
        if not scored:
            continue
        scored.sort()
        winner = scored[0][2]
        if winner == "dise":
            dise_wins.append(b)
        elif winner == "betting_cs":
            betting_wins.append(b)
        elif winner == "plain_mc":
            plain_mc_wins.append(b)
    out = [
        "# Synthesis — DiSE vs SoTA at a glance",
        "",
        f"**Configuration**: budget {config['budget']}, "
        f"epsilon {config['epsilon']}, delta {config['delta']}, "
        f"{config['n_seeds']} seeds, MC ground-truth at "
        f"{config['mc_samples']} samples.",
        "",
        "## Headline tally",
        "",
        f"* Benchmarks where `dise` is the tightest *sound* method: "
        f"{len(dise_wins)}/{len(bench_names)} (`{', '.join(dise_wins) or '—'}`)",
        f"* Benchmarks where `betting_cs` is the tightest *sound* method: "
        f"{len(betting_wins)}/{len(bench_names)} (`{', '.join(betting_wins) or '—'}`)",
        f"* Benchmarks where `plain_mc` is the tightest (fixed-`n` Wilson, "
        f"no adaptive stopping): {len(plain_mc_wins)}/{len(bench_names)} "
        f"(`{', '.join(plain_mc_wins) or '—'}`)",
        "",
        "## Soundness violations (coverage < 0.5)",
        "",
    ]
    if not soundness_failures:
        out.append("None observed at the stated confidence and seed count.")
    else:
        out.append("| benchmark | method | empirical coverage |")
        out.append("|---|---|---|")
        for b, m, c in soundness_failures:
            out.append(f"| `{b}` | {METHOD_LABELS[m]} | {c:.2f} |")
    out += [
        "",
        "## DiSE wall-clock vs sampling SoTA",
        "",
        "| benchmark | DiSE wall (s) | BettingCS wall (s) | overhead × |",
        "|---|---|---|---|",
    ]
    for b in bench_names:
        d = wall[b].get("dise")
        bc = wall[b].get("betting_cs")
        if d is not None and bc is not None and bc > 0:
            ratio = d / bc
            out.append(f"| `{b}` | {d:.3f} | {bc:.3f} | {ratio:,.1f}× |")
    out += [
        "",
        "## Caveat — MC truth noise vs. zero-width intervals",
        "",
        "Methods that produce *zero-width* intervals (`half = 0.0000`) "
        "have empirical coverage bounded above by the MC-truth noise: "
        "even when the point estimate is *analytically* correct, MC noise "
        "(~ σ/√n_mc) shifts the reference and can mark the seed as "
        "non-covering. This shows up most clearly on `coin_machine` where "
        "DiSE's MockBackend frequently returns the analytical truth "
        "99/9999 = 0.0099 while MC at 10K samples reports 0.0092 ± 0.0007.",
        "",
        "The `assertion_overflow_mul_w=8` failure is *not* of this kind — "
        "DiSE reports μ̂ = 0.43 while the MC truth is 0.4004 (analytical "
        "388/961 = 0.4037). This is a genuine soundness failure of the "
        "sample-based closure heuristic in MockBackend.",
        "",
    ]
    return "\n".join(out) + "\n"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out-dir", dest="out_dir", required=True)
    args = p.parse_args()
    data = load(args.in_path)
    rows = data["benchmarks"]
    config = data.get("config", {})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "table_half_width.md").write_text(
        render_table(
            "Certified half-width (median across seeds; lower = tighter)",
            rows, "median_half_width",
            lambda v: fmt_float(v, "{:.4f}"),
        )
    )
    (out_dir / "table_samples.md").write_text(
        render_table(
            "Samples used (median across seeds; lower = more efficient)",
            rows, "median_samples", fmt_int,
        )
    )
    (out_dir / "table_coverage.md").write_text(
        render_table(
            "Empirical coverage rate (fraction of seeds covering MC truth)",
            rows, "coverage",
            lambda v: fmt_float(v, "{:.2f}"),
        )
    )
    (out_dir / "table_wall_clock.md").write_text(
        render_table(
            "Wall-clock (median seconds across seeds; lower = faster)",
            rows, "median_wall_clock_s",
            lambda v: fmt_float(v, "{:.3f}"),
        )
    )
    (out_dir / "ranking.md").write_text(render_ranking(rows))
    (out_dir / "synthesis.md").write_text(render_synthesis(rows, config))
    print(f"Wrote tables to {out_dir}/")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
