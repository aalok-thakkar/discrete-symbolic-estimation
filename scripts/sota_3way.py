"""Three-way A/B between SoTA sweeps with three DiSE configurations.

Reads three ``summary.json`` files (the same baselines but DiSE
configured differently) and emits a single Markdown table comparing
DiSE half-width / samples / coverage across the three configurations,
alongside the strongest sampling-SoTA reference (PlainMC and BettingCS).

Usage::

    uv run python scripts/sota_3way.py \\
        --a results/sota_anytime/summary.json --a-label "DiSE(old)" \\
        --b results/sota_sound_e02/summary.json --b-label "DiSE(ε=.02)" \\
        --c results/sota_sound_e05/summary.json --c-label "DiSE(ε=.05)" \\
        --out results/sota_sound_e02/three_way.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path):
    with open(path) as f:
        return json.load(f)["benchmarks"]


def pivot(rows, method_name):
    return {r["benchmark"]: r for r in rows if r["method"] == method_name}


def fmt_cov(c):
    return "—" if c is None else f"{c:.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", required=True); p.add_argument("--a-label", required=True)
    p.add_argument("--b", required=True); p.add_argument("--b-label", required=True)
    p.add_argument("--c", required=True); p.add_argument("--c-label", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    A = pivot(load(args.a), "dise")
    B = pivot(load(args.b), "dise")
    C = pivot(load(args.c), "dise")
    plain = pivot(load(args.a), "plain_mc")
    betting = pivot(load(args.a), "betting_cs")
    benches = sorted(set(A) & set(B) & set(C))

    out_lines = [
        f"# DiSE A/B: closure-rule variants",
        "",
        f"`{args.a_label}` vs `{args.b_label}` vs `{args.c_label}`, "
        f"with PlainMC and BettingCS shown for reference.",
        "",
        f"| benchmark | MC truth | `plain_mc` half (cov) | `betting_cs` half (cov) "
        f"| {args.a_label} half (cov) | {args.b_label} half (cov) | {args.c_label} half (cov) |",
        "|---|---|---|---|---|---|---|",
    ]

    sound_e02_wins = 0
    sound_e02_ties = 0
    sound_e02_losses = 0
    sound_e05_wins = 0
    sound_e05_ties = 0
    sound_e05_losses = 0
    sound_e02_unsound = 0
    sound_e05_unsound = 0
    old_unsound = 0
    for bn in benches:
        a = A[bn]; b = B[bn]; c = C[bn]
        pmc = plain[bn]; bts = betting[bn]
        mc = a["mc_truth"]
        # The relevant comparison: does sound-DiSE beat PlainMC on half_width
        # AND maintain coverage >= 1-delta = 0.95?
        for r_old, r_new in [(a, b), (a, c)]:
            pass
        if a["coverage"] is not None and a["coverage"] < 0.95: old_unsound += 1
        if b["coverage"] is not None and b["coverage"] < 0.95: sound_e02_unsound += 1
        if c["coverage"] is not None and c["coverage"] < 0.95: sound_e05_unsound += 1
        # Compare b vs a (sound ε=0.02 vs old heuristic)
        if b["median_half_width"] < a["median_half_width"] - 1e-9: sound_e02_wins += 1
        elif b["median_half_width"] > a["median_half_width"] + 1e-9: sound_e02_losses += 1
        else: sound_e02_ties += 1
        if c["median_half_width"] < a["median_half_width"] - 1e-9: sound_e05_wins += 1
        elif c["median_half_width"] > a["median_half_width"] + 1e-9: sound_e05_losses += 1
        else: sound_e05_ties += 1
        out_lines.append(
            f"| `{bn}` | {mc:.4f} | "
            f"{pmc['median_half_width']:.4f} ({fmt_cov(pmc['coverage'])}) | "
            f"{bts['median_half_width']:.4f} ({fmt_cov(bts['coverage'])}) | "
            f"{a['median_half_width']:.4f} ({fmt_cov(a['coverage'])}) | "
            f"{b['median_half_width']:.4f} ({fmt_cov(b['coverage'])}) | "
            f"{c['median_half_width']:.4f} ({fmt_cov(c['coverage'])}) |"
        )
    out_lines += [
        "",
        f"### Summary vs old heuristic",
        "",
        f"* `{args.b_label}`: tighter on {sound_e02_wins}/{len(benches)}, "
        f"looser on {sound_e02_losses}, tied on {sound_e02_ties}. "
        f"Unsound (cov < 0.95) on {sound_e02_unsound}/{len(benches)}.",
        f"* `{args.c_label}`: tighter on {sound_e05_wins}/{len(benches)}, "
        f"looser on {sound_e05_losses}, tied on {sound_e05_ties}. "
        f"Unsound (cov < 0.95) on {sound_e05_unsound}/{len(benches)}.",
        f"* `{args.a_label}` (old heuristic): unsound (cov < 0.95) on "
        f"{old_unsound}/{len(benches)}.",
        "",
    ]
    # Samples comparison.
    out_lines += ["### Samples used (median)", "",
                  f"| benchmark | `plain_mc` | `betting_cs` "
                  f"| {args.a_label} | {args.b_label} | {args.c_label} |",
                  "|---|---|---|---|---|---|"]
    for bn in benches:
        out_lines.append(
            f"| `{bn}` | {plain[bn]['median_samples']} | "
            f"{betting[bn]['median_samples']} | "
            f"{A[bn]['median_samples']} | "
            f"{B[bn]['median_samples']} | "
            f"{C[bn]['median_samples']} |"
        )
    Path(args.out).write_text("\n".join(out_lines) + "\n")
    print(f"Wrote {args.out}")
    print(f"DiSE sound ε=0.02: {sound_e02_wins} tighter, {sound_e02_losses} looser, "
          f"{sound_e02_ties} tied; unsound on {sound_e02_unsound}/{len(benches)}")
    print(f"DiSE sound ε=0.05: {sound_e05_wins} tighter, {sound_e05_losses} looser, "
          f"{sound_e05_ties} tied; unsound on {sound_e05_unsound}/{len(benches)}")
    print(f"DiSE old heuristic: unsound on {old_unsound}/{len(benches)}")


if __name__ == "__main__":
    import sys
    sys.exit(main())
