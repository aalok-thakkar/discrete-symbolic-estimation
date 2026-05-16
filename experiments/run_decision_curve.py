"""Samples-to-decision experiment: the operational headline.

The question a reliability practitioner actually asks is *not* "what
is mu?" but "is mu below the contractual threshold tau?"  This is a
binary decision; the relevant cost is the sample count required to
make it.

For a sample-based method, the certified interval $[L, U]$ supports
a decision at confidence $1 - \\delta$ when it lies entirely on one
side of $\\tau$:
  - if $U < \\tau$: certify $\\mu < \\tau$ (decision: SLA met)
  - if $L > \\tau$: certify $\\mu > \\tau$ (decision: SLA violated)
  - otherwise: undecided, draw more samples.

For Wilson's interval at empirical $\\hat\\mu$, the half-width is
$\\Theta(\\sqrt{\\mu(1-\\mu)/n})$, so the sample count required to
support the decision scales as $n \\sim \\mu(1-\\mu)/(\\tau - \\mu)^2$.
**MC's decision time diverges as the threshold approaches the truth.**

DiSE, on a benchmark whose partition closes to closed-form mass,
delivers an exact certificate.  Its decision time is the sample count
required for the partition to close — *constant* in the
threshold-distance, regardless of how tight the SLA is.

This is the figure that should be the paper's headline.

Benchmark:
  An SLA-style classifier under a heavy-tailed BoundedGeometric load
  distribution.  ``load >= 5000`` flags an SLA violation; under
  ``BG(p=0.001, N=10000)`` the violation rate is
  $\\mu \\approx 0.0067$.

Threshold sweep:
  ``tau`` taken from a log-spaced grid above and below ``mu``;
  ``|tau - mu|`` ranges from $5 \\times 10^{-3}$ down to $5 \\times 10^{-5}$.

Methods: ``plain_mc``, ``plain_mc_betting``, ``dise_betting``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dise.distributions import BoundedGeometric, ProductDistribution
from dise.estimator import (
    prpl_eb_halfwidth_anytime,
    wilson_halfwidth_for_leaf,
)


RESULTS_DIR = ROOT / "experiments" / "results"
FIG_DIR = ROOT / "experiments" / "figures"


# ---------------------------------------------------------------------------
# Benchmark: SLA load classifier under BoundedGeometric(0.001, 10000).
# Closed-form truth: mu = P[load >= 5000] under BG.
# ---------------------------------------------------------------------------


N = 10_000
P_BG = 0.001
LOAD_THRESHOLD = 5000


def sla(load: int) -> int:
    """1 = SLA violation (load >= LOAD_THRESHOLD), 0 = OK."""
    if load < LOAD_THRESHOLD:
        return 0
    return 1


DISTRIBUTION = {"load": BoundedGeometric(p=P_BG, N=N)}
PROPERTY = lambda y: y == 1

# Closed-form truth.
_DIST_INSTANCE = BoundedGeometric(p=P_BG, N=N)
MU_TRUTH = 1.0 - _DIST_INSTANCE.mass(1, LOAD_THRESHOLD - 1)


# ---------------------------------------------------------------------------
# samples-to-decision driver
# ---------------------------------------------------------------------------


def _streaming_mc_decision(
    program: Callable[[int], int],
    distribution,
    property_fn: Callable[[int], bool],
    tau: float,
    delta: float,
    seed: int,
    max_budget: int,
    bound: str,
    batch: int = 200,
) -> tuple[int, float, float, str]:
    """Stream MC; stop when the certified interval lies entirely on
    one side of ``tau``.  Returns ``(n, hw, mu_hat, decision)``.
    ``decision`` in {``"below"``, ``"above"``, ``"undecided"``}.
    """
    rng = np.random.default_rng(seed)
    dist = ProductDistribution(factors=dict(distribution))
    keys = list(distribution.keys())
    n = 0
    hits = 0
    phis: list[int] = []
    while n < max_budget:
        take = min(batch, max_budget - n)
        sub = dist.sample(rng, take)
        for i in range(take):
            x = {k: int(sub[k][i]) for k in keys}
            v = 1 if bool(property_fn(program(**x))) else 0
            phis.append(v)
            hits += v
        n += take
        mu_hat = hits / n
        if bound == "wilson":
            hw = wilson_halfwidth_for_leaf(n, hits, delta)
        elif bound == "betting":
            hw = prpl_eb_halfwidth_anytime(phis, delta)
        else:
            raise ValueError(bound)
        lo = max(0.0, mu_hat - hw)
        hi = min(1.0, mu_hat + hw)
        if hi < tau:
            return n, hw, mu_hat, "below"
        if lo > tau:
            return n, hw, mu_hat, "above"
    return n, hw, mu_hat, "undecided"


def _dise_decision(
    tau: float, delta: float, seed: int, max_budget: int,
    dise_budget_seconds: float = 30.0,
) -> tuple[int, float, float, str]:
    """DiSE decision: run with a tight target epsilon, check the
    resulting certified interval against ``tau``."""
    from dise.estimator.api import estimate
    result = estimate(
        program=sla,
        distribution=DISTRIBUTION,
        property_fn=PROPERTY,
        epsilon=1e-6,
        delta=delta,
        budget=max_budget,
        budget_seconds=dise_budget_seconds,
        method="betting",
        bootstrap=500,
        batch_size=100,
        seed=seed,
    )
    lo, hi = result.interval
    hw = (hi - lo) / 2.0
    if hi < tau:
        decision = "below"
    elif lo > tau:
        decision = "above"
    else:
        decision = "undecided"
    return result.samples_used, hw, result.mu_hat, decision


def samples_to_decision(
    method: str, tau: float, delta: float, seed: int,
    max_budget: int, dise_budget_seconds: float = 30.0,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    if method == "plain_mc":
        n, hw, mu, dec = _streaming_mc_decision(
            sla, DISTRIBUTION, PROPERTY, tau, delta, seed, max_budget, "wilson")
    elif method == "plain_mc_betting":
        n, hw, mu, dec = _streaming_mc_decision(
            sla, DISTRIBUTION, PROPERTY, tau, delta, seed, max_budget, "betting")
    elif method == "dise_betting":
        n, hw, mu, dec = _dise_decision(
            tau, delta, seed, max_budget, dise_budget_seconds)
    else:
        raise ValueError(method)
    wall = time.perf_counter() - t0
    return dict(method=method, tau=tau, samples=n, half_width=hw,
                mu_hat=mu, decision=dec, wall=wall)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


METHODS = ["plain_mc", "plain_mc_betting", "dise_betting"]

# Threshold sweep — pick tau on a log-spaced grid below and above mu.
# Smaller |tau - mu| means MC needs more samples to decide.
def _build_taus() -> list[float]:
    deltas = [5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5]
    taus = []
    for d in deltas:
        taus.append(MU_TRUTH - d)
        taus.append(MU_TRUTH + d)
    return sorted(taus)


TAUS_FULL = _build_taus()
TAUS_QUICK = [MU_TRUTH - 1e-3, MU_TRUTH + 1e-3, MU_TRUTH - 1e-4, MU_TRUTH + 1e-4]
SEEDS_FULL = [0, 1, 2]
SEEDS_QUICK = [0]
DELTA = 0.05
MAX_BUDGET = 200_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        default=bool(os.environ.get("QUICK")))
    args = parser.parse_args()

    taus = TAUS_QUICK if args.quick else TAUS_FULL
    seeds = SEEDS_QUICK if args.quick else SEEDS_FULL

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows_path = RESULTS_DIR / "decision.jsonl"
    summary_path = RESULTS_DIR / "decision.json"
    rows_path.write_text("")

    print(f"# samples-to-decision sweep")
    print(f"# mu_truth = {MU_TRUTH:.6f} (BG p={P_BG}, N={N}, threshold {LOAD_THRESHOLD})")
    print(f"# {len(taus)} taus x {len(METHODS)} methods x {len(seeds)} seeds")
    print(f"# max_budget = {MAX_BUDGET:,}")
    print()
    n_total = len(taus) * len(METHODS) * len(seeds)
    n_done = 0
    t_start = time.perf_counter()
    with open(rows_path, "a") as fh:
        for tau in taus:
            for method in METHODS:
                for seed in seeds:
                    res = samples_to_decision(method, tau, DELTA, seed,
                                              MAX_BUDGET)
                    # ``res`` already has ``tau`` and ``method``; just enrich.
                    row = dict(res,
                               mu_truth=MU_TRUTH,
                               tau_minus_mu=tau - MU_TRUTH,
                               seed=seed, delta=DELTA,
                               max_budget=MAX_BUDGET)
                    fh.write(json.dumps(row) + "\n")
                    fh.flush()
                    n_done += 1
                    print(
                        f"  [{n_done:3d}/{n_total}] tau={tau:.6f} "
                        f"|tau-mu|={abs(tau-MU_TRUTH):.6f} "
                        f"{method:18s} seed={seed} n={res['samples']:6d} "
                        f"hw={res['half_width']:.6f} mu={res['mu_hat']:.6f} "
                        f"dec={res['decision']:9s} t={res['wall']:5.2f}s",
                        flush=True,
                    )

    wall = time.perf_counter() - t_start
    print(f"\n# total wall-clock: {wall:.1f}s")

    # Aggregate per (tau, method): median samples.
    rows = [json.loads(l) for l in open(rows_path) if l.strip()]
    agg: dict[tuple, list] = {}
    for r in rows:
        agg.setdefault((r["tau"], r["method"]), []).append(r)
    summary_rows = []
    for (tau, m), runs in sorted(agg.items()):
        samples = [r["samples"] for r in runs]
        hws = [r["half_width"] for r in runs]
        decs = [r["decision"] for r in runs]
        summary_rows.append({
            "tau": tau,
            "tau_minus_mu": tau - MU_TRUTH,
            "method": m,
            "n_seeds": len(runs),
            "median_samples": int(statistics.median(samples)),
            "median_half_width": statistics.median(hws),
            "all_decided": all(d in ("below", "above") for d in decs),
            "decisions": decs,
        })

    summary = {
        "metadata": {
            "benchmark": "sla_load_classifier",
            "N": N, "P_BG": P_BG, "load_threshold": LOAD_THRESHOLD,
            "mu_truth": MU_TRUTH, "delta": DELTA,
            "methods": METHODS, "taus": taus, "seeds": seeds,
            "max_budget": MAX_BUDGET, "total_wall_s": wall,
        },
        "rows": summary_rows,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"# wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
