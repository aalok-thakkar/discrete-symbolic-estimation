"""Run DiSE and SoTA comparators on every registered benchmark.

Methods compared
================
* ``plain_mc``           — fixed-n Wilson interval (DiSE's existing baseline).
* ``stratified_random``  — 16-bucket hash-stratified MC (DiSE's existing baseline).
* ``adaptive_hoeffding`` — Sequential Hoeffding, Sampson et al. (PLDI 2014).
* ``ebstop``             — Empirical-Bernstein stopping, Mnih–Szepesvári–Audibert (ICML 2008).
* ``betting_cs``         — Hedged-capital betting CS, Waudby-Smith & Ramdas (JRSS-B 2024).
* ``dise``               — DiSE (ASIP), this work.

Usage
-----
    uv run python scripts/sota_compare.py --budget 2000 --n-seeds 3 \
        --epsilon 0.05 --delta 0.05 \
        --out-dir results/sota/

Writes per-benchmark JSON reports plus a top-level ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dise.baselines import (
    AdaptiveHoeffding,
    BettingConfidenceSequence,
    DiSEBaseline,
    EmpiricalBernsteinStopping,
    PlainMonteCarlo,
    StratifiedRandomMC,
)
from dise.benchmarks import get_benchmark, list_benchmarks
from dise.experiment import run_experiment, save_report
from dise.smt import MockBackend


def build_methods(
    epsilon: float,
    batch_size: int = 50,
    bootstrap: int = 200,
    dise_budget_seconds: float = 30.0,
    dise_max_concolic_branches: int = 500,
    dise_method: str = "anytime",
    dise_closure_epsilon: float = 0.025,
    dise_delta_close: float = 0.005,
    dise_n_mass_samples: int = 10_000,
) -> list:
    # MockBackend = DiSE's documented fast configuration (sample-based
    # closure, no Z3 calls). See docs/evaluation.md §6. The Z3 backend
    # is for soundness validation; it does not change ASIP's adaptive
    # stratification, only how aggressively leaves can be proved
    # path-deterministic. Using MockBackend here keeps the comparison
    # within a tractable wall-clock budget; we report wall-clock in the
    # tables so the reader can see DiSE's symbolic overhead.
    #
    # `dise_budget_seconds` (default 30s) caps each DiSE run because
    # benchmarks with deep concolic traces (e.g. Collatz, Miller-Rabin)
    # can otherwise consume tens of minutes per seed. The cap is in
    # addition to the sample budget and is reported by DiSE as
    # `terminated_reason='time_exhausted'`.
    return [
        PlainMonteCarlo(),
        StratifiedRandomMC(n_strata=16),
        AdaptiveHoeffding(epsilon=epsilon, batch_size=batch_size),
        EmpiricalBernsteinStopping(epsilon=epsilon, batch_size=batch_size),
        BettingConfidenceSequence(epsilon=epsilon, batch_size=batch_size, grid_size=1024),
        DiSEBaseline(
            epsilon=epsilon,
            bootstrap=bootstrap,
            batch_size=batch_size,
            method=dise_method,
            backend=MockBackend(),
            budget_seconds=dise_budget_seconds,
            max_concolic_branches=dise_max_concolic_branches,
            closure_epsilon=dise_closure_epsilon,
            delta_close=dise_delta_close,
            n_mass_samples=dise_n_mass_samples,
        ),
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=2000)
    p.add_argument("--n-seeds", type=int, default=3)
    p.add_argument("--epsilon", type=float, default=0.05)
    p.add_argument("--delta", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--mc-samples", type=int, default=10_000)
    p.add_argument("--dise-budget-seconds", type=float, default=30.0)
    p.add_argument("--dise-max-concolic-branches", type=int, default=500)
    p.add_argument("--dise-method", type=str, default="anytime",
                   choices=["wilson", "anytime"],
                   help="DiSE per-leaf half-width method.")
    p.add_argument("--dise-closure-epsilon", type=float, default=0.025,
                   help="DiSE sample-closure disagreement budget (sound mode). "
                        "Pass 1.0 to recover the previous unsound heuristic.")
    p.add_argument("--dise-delta-close", type=float, default=0.005,
                   help="DiSE per-leaf closure-failure budget (sound mode).")
    p.add_argument("--dise-n-mass-samples", type=int, default=10_000,
                   help="MC samples used for general-region mass estimation.")
    p.add_argument(
        "--out-dir", type=str, default="results/sota",
        help="Directory for JSON outputs.",
    )
    p.add_argument(
        "--benchmarks", nargs="*", default=None,
        help="Restrict to a subset (space-separated). Default: all registered.",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = args.benchmarks or list_benchmarks()
    summary = {
        "config": {
            "budget": args.budget,
            "n_seeds": args.n_seeds,
            "epsilon": args.epsilon,
            "delta": args.delta,
            "batch_size": args.batch_size,
            "mc_samples": args.mc_samples,
        },
        "benchmarks": [],
    }
    t0 = time.perf_counter()
    for bname in benchmarks:
        bench = get_benchmark(bname)
        print(f"\n=== {bench.name} ===", flush=True)
        print(f"    {bench.description}", flush=True)
        methods = build_methods(
            epsilon=args.epsilon,
            batch_size=args.batch_size,
            bootstrap=args.bootstrap,
            dise_budget_seconds=args.dise_budget_seconds,
            dise_max_concolic_branches=args.dise_max_concolic_branches,
            dise_method=args.dise_method,
            dise_closure_epsilon=args.dise_closure_epsilon,
            dise_delta_close=args.dise_delta_close,
            dise_n_mass_samples=args.dise_n_mass_samples,
        )
        t_bench = time.perf_counter()
        report = run_experiment(
            benchmark_name=bench.name,
            description=bench.description,
            program=bench.program,
            distribution=bench.distribution,
            property_fn=bench.property_fn,
            methods=methods,
            budget=args.budget,
            delta=args.delta,
            seeds=range(args.n_seeds),
            mc_samples=args.mc_samples,
        )
        dt = time.perf_counter() - t_bench
        # Print the aggregates table.
        for a in report.aggregates:
            cov = f"{a.coverage:.2f}" if a.coverage is not None else "n/a"
            print(
                f"  {a.method:25s} "
                f"mu={a.median_mu_hat:.4f}  "
                f"half={a.median_half_width:.4f}  "
                f"samples={a.median_samples:5d}  "
                f"wall={a.median_wall_clock_s*1000:7.1f}ms  "
                f"coverage={cov}",
                flush=True,
            )
        # Persist per-benchmark JSON.
        safe = bname.replace("/", "_").replace(" ", "_")
        out_path = out_dir / f"{safe}.json"
        save_report(report, str(out_path))
        # Append a flat record to summary.
        for a in report.aggregates:
            summary["benchmarks"].append({
                "benchmark": bname,
                "method": a.method,
                "n_seeds": a.n_seeds,
                "median_mu_hat": a.median_mu_hat,
                "median_half_width": a.median_half_width,
                "median_samples": a.median_samples,
                "median_wall_clock_s": a.median_wall_clock_s,
                "iqr_half_width": a.iqr_half_width,
                "iqr_samples": a.iqr_samples,
                "coverage": a.coverage,
                "mc_truth": report.mc_truth,
                "median_error_vs_truth": a.median_error_vs_truth,
            })
        print(f"    [wrote {out_path}, {dt:.1f}s]", flush=True)

    wall = time.perf_counter() - t0
    summary["wall_clock_s"] = wall
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nTotal wall-clock: {wall:.1f}s", flush=True)
    print(f"Summary: {out_dir/'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
