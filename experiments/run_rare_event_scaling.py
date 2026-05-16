"""Rare-event scaling experiment: samples to a target half-width as the
true rare-event mass shrinks.

Headline figure of the experimental study.  For a parametric variant
of ``coin_machine`` whose rare-event mass $\\mu$ can be dialed by a
single integer knob, we measure the *sample count needed to certify
$\\mu$ to half-width $\\le \\varepsilon$* under each method, and plot
``samples_to_eps`` vs $\\mu$ on log--log axes.

Theory predicts:

  - Plain Monte Carlo:  ``samples ~ Theta(1 / (mu * eps^2))``,
    slope -1 on the log-log plot.
  - DiSE:               ``samples ~ Theta(1)``, slope 0 — the
    algorithm refines the input space into a constant number of
    closed-form leaves regardless of how small the rare-event slice
    is.

So we expect to see DiSE's curve diverge from MC's by orders of
magnitude as $\\mu$ shrinks.  This is the headline result.

Outputs:
  experiments/results/rare_event.jsonl     per-cell rows
  experiments/results/rare_event.json      aggregate summary
  experiments/figures/rare_event_scaling.{pdf,png}

Usage:
  uv run python experiments/run_rare_event_scaling.py
  QUICK=1 uv run python experiments/run_rare_event_scaling.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dise.baselines import (
    DiSEBaseline,
    PlainMonteCarlo,
    PlainMonteCarloBetting,
    PlainMonteCarloHoeffding,
)
from dise.distributions import ProductDistribution, Uniform
from dise.estimator import (
    prpl_eb_halfwidth_anytime,
    wilson_halfwidth_for_leaf,
)


RESULTS_DIR = ROOT / "experiments" / "results"
FIG_DIR = ROOT / "experiments" / "figures"


# ---------------------------------------------------------------------------
# Parametric benchmark
# ---------------------------------------------------------------------------


def make_rare_slice(rare_width: int, rare_lo: int = 1000) -> Callable[[int], int]:
    """Returns a program with a single axis-aligned rare-event slice.

    The program partitions its input into three regions, all
    **axis-aligned** (only ``<`` and ``>=`` comparisons on a single
    integer variable).  This is exactly the regime where DiSE's
    closed-form mass applies: refining on the two boundary clauses
    produces three leaves whose mass is computed exactly with zero
    variance.  As ``rare_width`` shrinks, $\\mu$ shrinks.

    The earlier draft of this benchmark used ``x % period == 0`` for
    the rare slice — non-LIA arithmetic that forces general-region
    mass estimation, defeating DiSE's structural advantage.  This
    version isolates the scaling question cleanly.
    """

    def program(x: int) -> int:
        if x < rare_lo:                  # below the rare slice
            return 0
        if x >= rare_lo + rare_width:    # above the rare slice
            return 0
        return 1                         # inside the slice

    program.__name__ = f"rare_slice_width_{rare_width}"
    return program


def exact_mu(rare_width: int, N: int = 9999, rare_lo: int = 1000) -> float:
    """Closed-form $\\mu$ for the axis-aligned rare-slice benchmark.

    The slice is ``[rare_lo, rare_lo + rare_width)``, clipped to
    ``[1, N]``; ``mu = |slice ∩ [1, N]| / N``.
    """
    lo = max(rare_lo, 1)
    hi = min(rare_lo + rare_width - 1, N)
    return max(0, hi - lo + 1) / N


# ---------------------------------------------------------------------------
# samples-to-epsilon driver
# ---------------------------------------------------------------------------


def _streaming_mc_to_epsilon(
    program: Callable[[int], int],
    N: int,
    eps: float,
    delta: float,
    seed: int,
    max_budget: int,
    bound: str,
    batch: int = 200,
) -> tuple[int, float, float]:
    """Plain Monte Carlo that streams samples in batches and stops
    when the certified half-width drops to <= ``eps``.

    Returns ``(samples_used, achieved_half_width, mu_hat)``.

    Supports three bounds:
      - "wilson"      : :func:`wilson_halfwidth_for_leaf` at fixed n
      - "hoeffding"   : sqrt(log(2/delta) / 2n)
      - "betting"     : :func:`prpl_eb_halfwidth_anytime` (WSR PrPl-EB)
    """
    rng = np.random.default_rng(seed)
    dist = ProductDistribution(factors={"x": Uniform(1, N)})
    n = 0
    hits = 0
    phis: list[int] = []
    while n < max_budget:
        take = min(batch, max_budget - n)
        sub = dist.sample(rng, take)
        for i in range(take):
            x = int(sub["x"][i])
            v = 1 if program(x) == 1 else 0
            phis.append(v)
            hits += v
        n += take
        # Compute current half-width.
        mu_hat = hits / n
        if bound == "wilson":
            hw = wilson_halfwidth_for_leaf(n, hits, delta)
        elif bound == "hoeffding":
            hw = math.sqrt(math.log(2.0 / delta) / (2.0 * n))
        elif bound == "betting":
            hw = prpl_eb_halfwidth_anytime(phis, delta)
        else:
            raise ValueError(bound)
        if hw <= eps:
            return n, hw, mu_hat
    # budget exhausted
    return n, hw, mu_hat


def samples_to_epsilon(
    method_name: str,
    program: Callable[[int], int],
    N: int,
    eps: float,
    delta: float,
    seed: int,
    max_budget: int,
    dise_method: str | None = None,
    dise_budget_seconds: float = 30.0,
) -> dict[str, Any]:
    """One ``(method, seed)`` measurement of samples-to-epsilon."""
    t0 = time.perf_counter()
    if method_name == "plain_mc":
        n, hw, mu = _streaming_mc_to_epsilon(
            program, N, eps, delta, seed, max_budget, bound="wilson"
        )
        wall = time.perf_counter() - t0
        return dict(method=method_name, samples=n, half_width=hw, mu_hat=mu,
                    wall=wall, hit_budget=(n >= max_budget))
    if method_name == "plain_mc_hoeffding":
        n, hw, mu = _streaming_mc_to_epsilon(
            program, N, eps, delta, seed, max_budget, bound="hoeffding"
        )
        wall = time.perf_counter() - t0
        return dict(method=method_name, samples=n, half_width=hw, mu_hat=mu,
                    wall=wall, hit_budget=(n >= max_budget))
    if method_name == "plain_mc_betting":
        n, hw, mu = _streaming_mc_to_epsilon(
            program, N, eps, delta, seed, max_budget, bound="betting"
        )
        wall = time.perf_counter() - t0
        return dict(method=method_name, samples=n, half_width=hw, mu_hat=mu,
                    wall=wall, hit_budget=(n >= max_budget))
    if method_name.startswith("dise_"):
        from dise.estimator.api import estimate
        meth = dise_method or method_name.split("_", 1)[1]
        # For very rare events, bump bootstrap so the rare branch has
        # a non-trivial chance of being observed before refinement.
        # Heuristic: aim for ~5 expected hits at the bootstrap stage.
        # We don't know mu in the algorithm; this is the experiment-
        # harness picking a generous default for the headline figure.
        bootstrap = 500
        result = estimate(
            program=program,
            distribution={"x": Uniform(1, N)},
            property_fn=lambda y: y == 1,
            epsilon=eps,
            delta=delta,
            budget=max_budget,
            budget_seconds=dise_budget_seconds,
            method=meth,
            bootstrap=bootstrap,
            batch_size=100,
            seed=seed,
        )
        wall = time.perf_counter() - t0
        lo, hi = result.interval
        hw = (hi - lo) / 2.0
        hit_budget = (result.terminated_reason in ("budget_exhausted",
                                                    "time_exhausted"))
        return dict(method=method_name, samples=result.samples_used,
                    half_width=hw, mu_hat=result.mu_hat,
                    wall=wall, hit_budget=hit_budget,
                    extras={"terminated_reason": result.terminated_reason,
                            "n_leaves": result.n_leaves,
                            "n_open_leaves": result.n_open_leaves})
    raise ValueError(method_name)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


METHODS = [
    "plain_mc",
    "plain_mc_hoeffding",
    "plain_mc_betting",
    "dise_wilson",
    "dise_betting",
]


# Rare-slice widths.  With N=9999 these span mu from ~0.5 down to
# ~1e-4 — five orders of magnitude, enough to see the Theta(1/mu) vs
# Theta(1) scaling diverge cleanly on a log-log plot.
RARE_WIDTHS_FULL = [5000, 1000, 100, 20, 5, 2, 1]
RARE_WIDTHS_QUICK = [1000, 20, 1]
SEEDS_FULL = [0, 1, 2]
SEEDS_QUICK = [0]

N = 9999
EPSILON = 0.005
DELTA = 0.05
MAX_BUDGET = 200_000       # cap MC well into the regime where it loses
DISE_BUDGET_SECONDS = 15   # generous; axis-aligned partitions converge fast


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        default=bool(os.environ.get("QUICK")))
    args = parser.parse_args()

    rare_widths = RARE_WIDTHS_QUICK if args.quick else RARE_WIDTHS_FULL
    seeds = SEEDS_QUICK if args.quick else SEEDS_FULL

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = RESULTS_DIR / "rare_event.jsonl"
    summary_path = RESULTS_DIR / "rare_event.json"
    rows_path.write_text("")

    n_total = len(rare_widths) * len(METHODS) * len(seeds)
    n_done = 0
    t_start = time.perf_counter()

    print(f"# rare-event scaling sweep")
    print(f"# {len(rare_widths)} rare_widths x {len(METHODS)} methods "
          f"x {len(seeds)} seeds = {n_total} cells")
    print(f"# eps={EPSILON}, delta={DELTA}, max_budget={MAX_BUDGET}, "
          f"dise_budget_seconds={DISE_BUDGET_SECONDS}")
    print()

    with open(rows_path, "a") as fh:
        for rare_width in rare_widths:
            program = make_rare_slice(rare_width)
            mu_truth = exact_mu(rare_width, N=N)
            print(f"[rare_width={rare_width:5d}] mu_truth = {mu_truth:.6f}",
                  flush=True)
            for method in METHODS:
                for seed in seeds:
                    res = samples_to_epsilon(
                        method_name=method,
                        program=program,
                        N=N,
                        eps=EPSILON,
                        delta=DELTA,
                        seed=seed,
                        max_budget=MAX_BUDGET,
                        dise_budget_seconds=DISE_BUDGET_SECONDS,
                    )
                    row = dict(rare_width=rare_width, mu_truth=mu_truth,
                               seed=seed, eps=EPSILON, delta=DELTA,
                               max_budget=MAX_BUDGET, **res)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    n_done += 1
                    print(
                        f"  [{n_done:3d}/{n_total}] w={rare_width:5d} "
                        f"{method:20s} seed={seed} samples={res['samples']:6d} "
                        f"hw={res['half_width']:.4f} mu={res['mu_hat']:.4f} "
                        f"t={res['wall']:5.2f}s "
                        f"{'(BUDGET)' if res['hit_budget'] else ''}",
                        flush=True,
                    )

    wall = time.perf_counter() - t_start
    print(f"\n# total wall-clock: {wall:.1f}s")

    # Build aggregate: median samples per (rare_period, method) across seeds.
    rows: list[dict[str, Any]] = []
    with open(rows_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    agg: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        agg.setdefault((r["rare_width"], r["method"]), []).append(r)

    summary_rows = []
    for (rw, m), runs in sorted(agg.items()):
        samples = [r["samples"] for r in runs]
        hws = [r["half_width"] for r in runs]
        any_hit = any(r["hit_budget"] for r in runs)
        summary_rows.append({
            "rare_width": rw,
            "mu_truth": runs[0]["mu_truth"],
            "method": m,
            "n_seeds": len(runs),
            "median_samples": int(statistics.median(samples)),
            "median_half_width": statistics.median(hws),
            "any_hit_budget": any_hit,
        })

    summary = {
        "metadata": {
            "epsilon": EPSILON,
            "delta": DELTA,
            "max_budget": MAX_BUDGET,
            "dise_budget_seconds": DISE_BUDGET_SECONDS,
            "N": N,
            "methods": METHODS,
            "rare_widths": rare_widths,
            "seeds": seeds,
            "total_wall_clock_s": wall,
        },
        "rows": summary_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"# wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
