"""Shared utilities for the benchmark scripts.

Every benchmark module in :mod:`dise.benchmarks` follows the same
shape: it builds a :class:`~dise.benchmarks.Benchmark` instance via a
``_build(...)`` factory, registers a canonical configuration, and
exposes a ``main()`` entry-point that runs the benchmark from the
command line. This module provides the three helpers each ``main()``
uses:

* :func:`pick_backend` — turns the ``--backend`` and ``--cache-smt``
  CLI flags into a concrete :class:`~dise.smt.SMTBackend` instance,
  optionally wrapped in a :class:`~dise.smt.CachedBackend`.
* :func:`common_argparser` — the shared ``argparse`` template; every
  benchmark inherits the standard knobs (``--epsilon``, ``--delta``,
  ``--budget``, ``--no-budget``, ``--budget-seconds``,
  ``--min-gain-per-cost``, ``--bootstrap``, ``--batch-size``,
  ``--seed``, ``--backend``, ``--cache-smt``, ``--mc-samples``,
  ``--skip-mc``, ``--json-out``).
* :func:`ground_truth_mc` — a plain-Monte-Carlo reference estimate
  used as a regression check (re-exported by :mod:`dise.experiment`
  for symmetry).
* :func:`run_and_print` — the canonical wiring: MC ground truth (on
  stderr) → :func:`dise.estimate` → JSON one-line summary (on stdout).

The split between **stderr** (human-readable progress) and **stdout**
(machine-readable JSON one-liner) is deliberate: piping
``dise run … | jq …`` works while a human running the same command
still sees the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Literal

from dise import EstimationResult, default_backend, estimate
from dise.experiment import ground_truth_mc
from dise.smt import CachedBackend, MockBackend, SMTBackend
from dise.smt import Z3Backend as _Z3Backend

from ._base import Benchmark


def pick_backend(
    name: Literal["auto", "z3", "mock"] | None,
    cache: bool = False,
) -> SMTBackend:
    """Construct an :class:`SMTBackend` from a CLI-style backend name.

    Parameters
    ----------
    name :
        ``"auto"`` or ``None`` →
        :func:`~dise.smt.default_backend` (Z3 if installed, else
        :class:`~dise.smt.MockBackend`). ``"z3"`` requires the
        ``z3-solver`` package. ``"mock"`` returns
        :class:`~dise.smt.MockBackend`.
    cache : bool, default False
        If ``True``, wrap the chosen backend in
        :class:`~dise.smt.CachedBackend` (memoizes ``is_satisfiable``,
        ``evaluate``, etc.; typical hit-rates of 40–60 % on the
        headline benchmark).

    Returns
    -------
    SMTBackend
    """
    if name in (None, "auto"):
        backend: SMTBackend = default_backend()
    elif name == "z3":
        if _Z3Backend is None:
            raise RuntimeError("z3-solver not installed but --backend z3 was requested")
        backend = _Z3Backend()
    elif name == "mock":
        backend = MockBackend()
    else:
        raise ValueError(f"unknown backend: {name!r}")
    if cache:
        backend = CachedBackend(backend)
    return backend


def common_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--epsilon", type=float, default=0.05, help="target half-width")
    p.add_argument("--delta", type=float, default=0.05, help="confidence parameter")
    p.add_argument(
        "--budget",
        type=int,
        default=5000,
        help="max concolic samples; pair with --no-budget to disable",
    )
    p.add_argument(
        "--no-budget",
        action="store_true",
        help="disable the sample cap; rely on epsilon (and --budget-seconds, if any)",
    )
    p.add_argument(
        "--budget-seconds",
        type=float,
        default=None,
        help="optional wall-clock cap in seconds",
    )
    p.add_argument(
        "--min-gain-per-cost",
        type=float,
        default=0.0,
        help="diminishing-returns floor for action selection (default 0)",
    )
    p.add_argument(
        "--method",
        type=str,
        default="wilson",
        choices=["wilson", "anytime", "bernstein", "empirical-bernstein"],
        help="certified half-width construction (default 'wilson'; "
             "'anytime' for ATVA-grade soundness under adaptive stopping)",
    )
    p.add_argument("--bootstrap", type=int, default=200, help="bootstrap samples")
    p.add_argument("--batch-size", type=int, default=50, help="samples per batch")
    p.add_argument("--seed", type=int, default=0, help="rng seed")
    p.add_argument(
        "--backend",
        choices=["auto", "z3", "mock"],
        default="auto",
        help="SMT backend (default: auto = z3 if installed, else mock)",
    )
    p.add_argument(
        "--cache-smt",
        action="store_true",
        help="wrap the backend with a memoizing CachedBackend",
    )
    p.add_argument(
        "--mc-samples", type=int, default=10_000, help="MC ground-truth samples"
    )
    p.add_argument(
        "--skip-mc",
        action="store_true",
        help="skip the MC ground-truth comparison",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="if set, write results to this JSON file in addition to stdout",
    )
    return p


def run_and_print(bench: Benchmark, args: argparse.Namespace) -> EstimationResult:
    backend = pick_backend(args.backend, cache=getattr(args, "cache_smt", False))
    print(f"=== Benchmark: {bench.name} ===", file=sys.stderr)
    print(f"  {bench.description}", file=sys.stderr)
    if bench.notes:
        print(f"  notes: {bench.notes}", file=sys.stderr)
    print(f"Backend: {type(backend).__name__}", file=sys.stderr)
    mc_mu: float | None = None
    mc_se: float | None = None
    if not args.skip_mc:
        mc_mu, mc_se = ground_truth_mc(
            bench.program, bench.distribution, bench.property_fn, args.mc_samples
        )
        ci_lo = max(0.0, mc_mu - 1.96 * mc_se)
        ci_hi = min(1.0, mc_mu + 1.96 * mc_se)
        print(
            f"MC ground truth (n={args.mc_samples}): "
            f"mu_MC = {mc_mu:.4f} ± {1.96 * mc_se:.4f}  "
            f"CI95 ≈ [{ci_lo:.4f}, {ci_hi:.4f}]",
            file=sys.stderr,
        )
    effective_budget: int | None = (
        None if getattr(args, "no_budget", False) else args.budget
    )
    budget_seconds = getattr(args, "budget_seconds", None)
    min_gain_per_cost = getattr(args, "min_gain_per_cost", 0.0)
    method: Literal["wilson", "anytime", "bernstein", "empirical-bernstein"] = (
        getattr(args, "method", "wilson")
    )
    print(
        f"Running DiSE: epsilon={args.epsilon}, delta={args.delta}, "
        f"budget={effective_budget!r}, budget_seconds={budget_seconds!r}, "
        f"method={method!r}, "
        f"bootstrap={args.bootstrap}, batch_size={args.batch_size} ...",
        file=sys.stderr,
    )
    result = estimate(
        program=bench.program,
        distribution=bench.distribution,
        property_fn=bench.property_fn,
        epsilon=args.epsilon,
        delta=args.delta,
        budget=effective_budget,
        budget_seconds=budget_seconds,
        min_gain_per_cost=min_gain_per_cost,
        method=method,
        bootstrap=args.bootstrap,
        batch_size=args.batch_size,
        seed=args.seed,
        backend=backend,
    )
    print(result, file=sys.stderr)
    print(f"  half-width = {result.half_width:.4g}", file=sys.stderr)
    print(
        f"  leaves={result.n_leaves} "
        f"(open={result.n_open_leaves}, closed={result.n_closed_leaves})",
        file=sys.stderr,
    )

    if getattr(args, "json_out", None):
        out = {
            "benchmark": bench.name,
            "description": bench.description,
            "backend": type(backend).__name__,
            "args": {
                "epsilon": args.epsilon,
                "delta": args.delta,
                "budget": effective_budget,
                "budget_seconds": budget_seconds,
                "min_gain_per_cost": min_gain_per_cost,
                "bootstrap": args.bootstrap,
                "batch_size": args.batch_size,
                "seed": args.seed,
            },
            "mc_ground_truth": {
                "mu": mc_mu,
                "se": mc_se,
                "n": args.mc_samples,
            } if not args.skip_mc else None,
            "dise_result": {
                "mu_hat": result.mu_hat,
                "interval": list(result.interval),
                "half_width": result.half_width,
                "eps_stat": result.eps_stat,
                "W_open": result.W_open,
                "delta": result.delta,
                "samples_used": result.samples_used,
                "refinements_done": result.refinements_done,
                "n_leaves": result.n_leaves,
                "n_open_leaves": result.n_open_leaves,
                "n_closed_leaves": result.n_closed_leaves,
                "terminated_reason": result.terminated_reason,
            },
        }
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.json_out}", file=sys.stderr)
    # also stdout: print JSON-like one-liner so | tee / grep is easy
    print(json.dumps({"mu_hat": result.mu_hat, "interval": list(result.interval),
                      "samples": result.samples_used,
                      "refinements": result.refinements_done,
                      "terminated": result.terminated_reason}))
    return result


__all__ = [
    "common_argparser",
    "ground_truth_mc",  # re-exported from dise.experiment
    "pick_backend",
    "run_and_print",
]
