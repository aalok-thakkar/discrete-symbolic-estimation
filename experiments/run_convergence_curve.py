"""Convergence-curve experiment: certified half-width as a function of
sample budget on a fixed-$\\mu$ benchmark.

This is the experiment that visualises \\toolname{}'s structural
advantage at any budget where the partition closes.  Plain Monte
Carlo's half-width decays as $\\Theta(1/\\sqrt{n})$ at best and never
reaches zero (an iid Bernoulli estimator's standard error has no
finite-sample floor of zero).  \\toolname{}, on an axis-aligned
partition whose leaves all admit closed-form mass, reports half-width
exactly zero from the budget threshold at which all leaves close,
onward.

The headline narrative is: at any reasonable target $\\varepsilon$ on
this benchmark, plain MC eventually catches up; below $\\varepsilon$
small enough that the Bernoulli SE is forced into the same regime as
\\toolname{}'s closed-form mass, plain MC cannot match \\toolname{}
at any budget.

Outputs:
  experiments/results/convergence.jsonl
  experiments/results/convergence.json
  experiments/figures/06_convergence_curve.{pdf,png}

Usage:
  uv run python experiments/run_convergence_curve.py
  QUICK=1 uv run python experiments/run_convergence_curve.py
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dise.baselines import (
    DiSEBaseline,
    PlainMonteCarlo,
    PlainMonteCarloBetting,
    PlainMonteCarloHoeffding,
)
from dise.distributions import Uniform


RESULTS_DIR = ROOT / "experiments" / "results"
FIG_DIR = ROOT / "experiments" / "figures"


# ---------------------------------------------------------------------------
# Benchmark: a single-conditional threshold program.
#
#   def threshold(x):
#       return 1 if x < THRESHOLD else 0
#
# Under x ~ Uniform(1, N), mu = (THRESHOLD - 1) / N exactly.  The program
# has a single conditional branch, so DiSE refines the root on
# ``x < THRESHOLD`` and produces two axis-aligned leaves whose mass is
# computed in closed form with zero variance.  This is the cleanest
# regime for DiSE: refinement is one step, closure is two SMT calls,
# and the certified half-width is exactly zero.
#
# Plain Monte Carlo cannot match: its certified half-width is bounded
# below by the Bernoulli standard error sqrt(mu(1-mu)/n) (times the
# z-quantile), which decays as Theta(1/sqrt(n)) and never reaches
# zero at any finite budget.
# ---------------------------------------------------------------------------


N = 9999
THRESHOLD = 50              # mu = 49 / 9999 ~= 0.0049
MU_TRUTH = (THRESHOLD - 1) / N


def threshold(x: int) -> int:
    return 1 if x < THRESHOLD else 0


DISTRIBUTION = {"x": Uniform(1, N)}
PROPERTY = lambda y: y == 1


# Methods compared on the convergence curve.  We focus on four:
# - plain_mc            (Wilson, Bernoulli-tight)
# - plain_mc_hoeffding  (Hoeffding, distribution-free)
# - plain_mc_betting    (WSR PrPl-EB, anytime-valid)
# - dise_betting        (DiSE with the same WSR bound + symbolic strata)


METHOD_FACTORIES = {
    "plain_mc":            lambda: PlainMonteCarlo(),
    "plain_mc_hoeffding":  lambda: PlainMonteCarloHoeffding(),
    "plain_mc_betting":    lambda: PlainMonteCarloBetting(),
    "dise_betting":        lambda: _dise(),
}


def _dise():
    b = DiSEBaseline(
        epsilon=1e-6,            # never auto-terminate; let budget cap drive
        method="betting",
        bootstrap=500,
        batch_size=100,
        budget_seconds=30.0,     # generous; we want full convergence
    )
    b.name = "dise_betting"
    return b


# Budgets to sweep — log-spaced from 200 to 100,000.
BUDGETS_FULL = [200, 500, 1000, 2000, 5000, 10_000, 20_000, 50_000, 100_000]
BUDGETS_QUICK = [500, 2000, 10_000]
SEEDS_FULL = [0, 1, 2]
SEEDS_QUICK = [0]
DELTA = 0.05


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        default=bool(os.environ.get("QUICK")))
    args = parser.parse_args()

    budgets = BUDGETS_QUICK if args.quick else BUDGETS_FULL
    seeds = SEEDS_QUICK if args.quick else SEEDS_FULL

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = RESULTS_DIR / "convergence.jsonl"
    summary_path = RESULTS_DIR / "convergence.json"
    rows_path.write_text("")

    n_total = len(budgets) * len(METHOD_FACTORIES) * len(seeds)
    print(f"# convergence-curve sweep")
    print(f"# benchmark: threshold (x < {THRESHOLD}; mu = {MU_TRUTH:.6f})")
    print(f"# {len(budgets)} budgets x {len(METHOD_FACTORIES)} methods "
          f"x {len(seeds)} seeds = {n_total} cells")
    print(f"# delta={DELTA}, budgets={budgets}")
    print()

    t_start = time.perf_counter()
    n_done = 0
    with open(rows_path, "a") as fh:
        for budget in budgets:
            for method_name, factory in METHOD_FACTORIES.items():
                for seed in seeds:
                    method = factory()
                    t0 = time.perf_counter()
                    result = method.run(
                        program=threshold,
                        distribution=DISTRIBUTION,
                        property_fn=PROPERTY,
                        budget=budget,
                        delta=DELTA,
                        seed=seed,
                    )
                    wall = time.perf_counter() - t0
                    lo, hi = result.interval
                    row = dict(
                        method=method_name,
                        budget=budget,
                        seed=seed,
                        mu_truth=MU_TRUTH,
                        mu_hat=result.mu_hat,
                        half_width=(hi - lo) / 2.0,
                        samples_used=result.samples_used,
                        wall_s=wall,
                        delta=DELTA,
                        interval=[lo, hi],
                        extras=result.extras,
                    )
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    n_done += 1
                    print(
                        f"  [{n_done:3d}/{n_total}] {method_name:20s} "
                        f"budget={budget:6d} seed={seed} mu={result.mu_hat:.4f} "
                        f"hw={(hi - lo) / 2.0:.5f} n_used={result.samples_used:6d} "
                        f"t={wall:5.2f}s",
                        flush=True,
                    )

    wall_total = time.perf_counter() - t_start
    print(f"\n# total wall-clock: {wall_total:.1f}s")

    # Aggregate.
    rows = [json.loads(l) for l in open(rows_path) if l.strip()]
    agg: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        agg.setdefault((r["budget"], r["method"]), []).append(r)

    summary_rows = []
    for (b, m), runs in sorted(agg.items()):
        hws = [r["half_width"] for r in runs]
        mus = [r["mu_hat"] for r in runs]
        ns = [r["samples_used"] for r in runs]
        ws = [r["wall_s"] for r in runs]
        summary_rows.append({
            "budget": b,
            "method": m,
            "n_seeds": len(runs),
            "median_mu_hat": statistics.median(mus),
            "median_half_width": statistics.median(hws),
            "median_samples_used": int(statistics.median(ns)),
            "median_wall_s": statistics.median(ws),
            "min_half_width": min(hws),
            "max_half_width": max(hws),
        })

    summary = {
        "metadata": {
            "benchmark": "threshold",
            "N": N,
            "threshold": THRESHOLD,
            "mu_truth": MU_TRUTH,
            "delta": DELTA,
            "methods": list(METHOD_FACTORIES.keys()),
            "budgets": budgets,
            "seeds": seeds,
            "total_wall_s": wall_total,
        },
        "rows": summary_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"# wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
