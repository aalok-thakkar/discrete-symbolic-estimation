"""Full experimental study for the DiSE paper.

Sweeps the Cartesian product (benchmark x method x budget x seed),
runs each cell, and records:

  - per-row JSONL  ->  experiments/results/runs.jsonl
  - per-(benchmark, method, budget) summary  ->  experiments/results/summary.json

The benchmarks come from established sources:

  - Hacker's Delight 2e (Warren, 2012): popcount_w6, parity_w6, log2_w6
  - CLRS / Knuth TAoCP: gcd_steps_le_5_BG, modpow_fits_in_4b, integer_sqrt,
    sieve_primality, miller_rabin_w=2
  - CERT C Secure Coding (INT32-C): assertion_overflow_mul_w=8
  - Recreational math (Lagarias 2010): collatz_le_30
  - Pedagogical: coin_machine_U(1,9999), sparse_trie_depth

Methods compared:

  - plain_mc          : plain Monte Carlo with Wilson interval (Sampson-style)
  - stratified_random : 16 random hash buckets with per-bucket Wilson +
                        Bonferroni (a sampling-only stratifier)
  - dise_wilson       : DiSE with method="wilson"   (tight fixed-n bound)
  - dise_anytime      : DiSE with method="anytime"  (union-bound-in-time Wilson)
  - dise_betting      : DiSE with method="betting"  (WSR 2024 PrPl-EB)

Usage:
  uv run python experiments/run_full_study.py
  QUICK=1 uv run python experiments/run_full_study.py    # smaller sweep
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Ensure the local src/ tree wins over any installed copy.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dise.baselines import (
    AdaptiveStratifiedMC,
    DiSEBaseline,
    PlainMonteCarlo,
    PlainMonteCarloBetting,
    PlainMonteCarloEmpiricalBernstein,
    PlainMonteCarloHoeffding,
    QuasiMonteCarloSobol,
    StratifiedRandomMC,
)
from dise.benchmarks import get_benchmark, list_benchmarks
from dise.experiment import RunResult, ground_truth_mc, run_method


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


BENCHMARKS = [
    # name on registry, citation tag for the report.
    ("coin_machine_U(1,9999)",              "pedagogical"),
    ("popcount_w6",                          "hackers_delight"),
    ("parity_w6",                            "hackers_delight"),
    ("log2_w6",                              "hackers_delight"),
    ("integer_sqrt_correct_U(1,1023)",       "clrs_taocp"),
    ("sieve_primality_U(2,200)",             "clrs_taocp"),
    ("modpow_fits_in_4b_m=37",               "clrs_taocp"),
    ("gcd_steps_le_5_BG(p=0.1,N=100)",       "clrs_taocp"),
    ("miller_rabin_w=2_BG(p=0.05,N=200)",    "clrs_taocp"),
    ("collatz_le_30_BG(p=0.05,N=200)",       "lagarias_2010"),
    ("sparse_trie_depth_le_3_U(0,63)",       "pedagogical"),
    ("assertion_overflow_mul_w=8_U(1,31)",   "cert_c_int32"),
]


# Budget points (number of concolic samples allowed).
BUDGETS_FULL = [500, 2000]
BUDGETS_QUICK = [2000]


# Seeds for variance estimation.
SEEDS_FULL = [0, 1, 2]
SEEDS_QUICK = [0, 1]


DELTA = 0.05
MC_TRUTH_SAMPLES = 5_000
MC_TRUTH_SEED = 12_345


# Hard wall-clock cap per cell, in seconds. Without this the DiSE
# variants can spin for minutes when the strict closure rule can't
# certify a region — popcount_w6 is the canonical offender. We pass
# this down to DiSE as ``budget_seconds`` so the algorithm terminates
# cleanly via ``time_exhausted``. The outer SIGALRM is a safety net.
PER_CELL_TIMEOUT_S = 8.0


def _make_methods(
    epsilon: float = 0.05, budget_seconds: float = PER_CELL_TIMEOUT_S
) -> list:
    """The ten comparators evaluated in this study.

    Sampling-only baselines (no symbolic reasoning):
      - ``plain_mc``                — Wilson interval (Bernoulli-tight)
      - ``plain_mc_hoeffding``      — textbook SMC bound
      - ``plain_mc_eb``             — Maurer-Pontil empirical Bernstein
      - ``plain_mc_betting``        — WSR PrPl-EB (same bound as DiSE,
                                       no stratification)  [ablation]
      - ``quasi_mc_sobol``          — Sobol low-discrepancy points + Wilson
      - ``stratified_random``       — 16 random hash strata
      - ``adaptive_stratified``     — 2-pass Neyman-allocation stratification
                                       (Carpentier-Munos 2011 spirit)

    DiSE variants (share the scheduler, differ in the per-leaf bound):
      - ``dise_wilson``  — fixed-n Wilson
      - ``dise_anytime`` — union-bound-in-time Wilson
      - ``dise_betting`` — WSR PrPl-EB  [recommended]

    All DiSE variants get a ``budget_seconds`` cap so they terminate
    cleanly when the strict closure rule can't make progress.
    """
    methods = [
        PlainMonteCarlo(),
        PlainMonteCarloHoeffding(),
        PlainMonteCarloEmpiricalBernstein(),
        PlainMonteCarloBetting(),
        QuasiMonteCarloSobol(),
        StratifiedRandomMC(n_strata=16),
        AdaptiveStratifiedMC(n_strata=16, pilot_frac=0.3),
    ]
    for dise_method in ("wilson", "anytime", "betting"):
        b = DiSEBaseline(
            epsilon=epsilon,
            method=dise_method,
            bootstrap=200,
            batch_size=50,
            budget_seconds=budget_seconds,
        )
        # Disambiguate the row label in the report.
        b.name = f"dise_{dise_method}"
        methods.append(b)
    return methods


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _benchmark_with_truth(name: str) -> tuple[Any, float, float]:
    """Resolve a registered benchmark and compute its MC ground truth."""
    bench = get_benchmark(name)
    if bench.closed_form_mu is not None:
        mu = float(bench.closed_form_mu)
        se = 0.0
    else:
        mu, se = ground_truth_mc(
            bench.program,
            bench.distribution,
            bench.property_fn,
            n_samples=MC_TRUTH_SAMPLES,
            seed=MC_TRUTH_SEED,
        )
    return bench, mu, se


class _CellTimeout(Exception):
    """Raised by the SIGALRM handler when a cell exceeds its budget."""


def _alarm_handler(signum, frame):  # pragma: no cover - signal dispatch
    raise _CellTimeout()


def run_cell(
    bench_name: str,
    bench: Any,
    method: Any,
    budget: int,
    seed: int,
    mc_truth: float,
    timeout_s: float = PER_CELL_TIMEOUT_S,
) -> RunResult:
    """One (benchmark, method, budget, seed) point.

    Two-layer timeout:
      1. DiSE methods receive ``budget_seconds=timeout_s`` and stop
         themselves via ``time_exhausted``.
      2. As a safety net, the whole cell is wrapped in a SIGALRM with
         a slightly longer deadline; if it fires, we record a sentinel
         row and continue. SIGALRM is POSIX-only (works on macOS/Linux).
    """
    import signal

    t0 = time.perf_counter()
    # Give the cell a generous outer cap: timeout_s + 5s grace.
    outer_cap = max(int(timeout_s + 5), 2)
    prev_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(outer_cap)
    try:
        try:
            row = run_method(
                method=method,
                program=bench.program,
                distribution=bench.distribution,
                property_fn=bench.property_fn,
                budget=budget,
                delta=DELTA,
                seed=seed,
                benchmark_name=bench_name,
                mc_truth=mc_truth,
            )
        except _CellTimeout:
            elapsed = time.perf_counter() - t0
            row = RunResult(
                benchmark=bench_name,
                method=getattr(method, "name", str(method)),
                seed=seed,
                budget=budget,
                delta=DELTA,
                mu_hat=float("nan"),
                interval=(0.0, 1.0),
                half_width=0.5,
                samples_used=0,
                wall_clock_s=elapsed,
                mc_truth=mc_truth,
                interval_contains_truth=False,
                error_vs_truth=None,
                extras={"error": f"SIGALRM after {outer_cap}s"},
            )
        except Exception as exc:  # pragma: no cover - defensive
            elapsed = time.perf_counter() - t0
            row = RunResult(
                benchmark=bench_name,
                method=getattr(method, "name", str(method)),
                seed=seed,
                budget=budget,
                delta=DELTA,
                mu_hat=float("nan"),
                interval=(0.0, 1.0),
                half_width=0.5,
                samples_used=0,
                wall_clock_s=elapsed,
                mc_truth=mc_truth,
                interval_contains_truth=None,
                error_vs_truth=None,
                extras={"error": f"{type(exc).__name__}: {exc}"},
            )
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_handler)
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        default=bool(os.environ.get("QUICK")),
        help="smaller sweep for fast iteration",
    )
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=None,
        help="subset of benchmark names; default = the curated set in this script",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "experiments" / "results"),
        help="directory for runs.jsonl and summary.json",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_path = out_dir / "runs.jsonl"
    summary_path = out_dir / "summary.json"

    budgets = BUDGETS_QUICK if args.quick else BUDGETS_FULL
    seeds = SEEDS_QUICK if args.quick else SEEDS_FULL
    bench_names = args.benchmarks or [name for name, _ in BENCHMARKS]
    citations = dict(BENCHMARKS)

    n_methods = len(_make_methods())
    print(f"# DiSE experimental study")
    print(f"# {len(bench_names)} benchmarks x {n_methods} methods x "
          f"{len(budgets)} budgets x {len(seeds)} seeds "
          f"= {len(bench_names) * n_methods * len(budgets) * len(seeds)} cells")
    print(f"# budgets: {budgets}")
    print(f"# seeds  : {seeds}")
    print(f"# delta  : {DELTA}")
    print(f"# output : {runs_path}, {summary_path}")
    print()

    # Stream JSONL so progress is visible and partial failures don't lose data.
    runs_path.write_text("")  # truncate
    n_done = 0
    n_total = len(bench_names) * 5 * len(budgets) * len(seeds)
    t_study_start = time.perf_counter()

    # Cache benchmark + MC truth per benchmark (avoid recomputing MC).
    bench_cache: dict[str, tuple[Any, float, float]] = {}

    with open(runs_path, "a") as fh:
        for bench_name in bench_names:
            if bench_name not in bench_cache:
                bench, mu_truth, se_truth = _benchmark_with_truth(bench_name)
                bench_cache[bench_name] = (bench, mu_truth, se_truth)
                print(f"[{bench_name}] mc_truth = {mu_truth:.4f} (se={se_truth:.4f})", flush=True)
            bench, mu_truth, _ = bench_cache[bench_name]

            for budget in budgets:
                # Construct methods per cell so each gets a fresh state.
                for method in _make_methods(epsilon=DELTA):
                    for seed in seeds:
                        t_cell = time.perf_counter()
                        row = run_cell(
                            bench_name=bench_name,
                            bench=bench,
                            method=method,
                            budget=budget,
                            seed=seed,
                            mc_truth=mu_truth,
                        )
                        # Tag the row with citation provenance.
                        row.extras["citation"] = citations.get(bench_name, "unknown")
                        fh.write(json.dumps(asdict(row), default=_json_default) + "\n")
                        fh.flush()
                        n_done += 1
                        elapsed = time.perf_counter() - t_cell
                        print(
                            f"  [{n_done:4d}/{n_total}] {bench_name:42s} "
                            f"{row.method:18s} budget={budget:5d} seed={seed} "
                            f"mu={row.mu_hat:.4f} hw={row.half_width:.4f} "
                            f"n={row.samples_used:5d} t={elapsed:6.2f}s "
                            f"contains={row.interval_contains_truth}",
                            flush=True,
                        )

    t_total = time.perf_counter() - t_study_start
    print(f"\n# total wall-clock: {t_total:.1f}s")

    # Build summary from JSONL.
    summary = _summarise(runs_path)
    summary["metadata"] = {
        "benchmarks": [{"name": n, "citation": c} for n, c in BENCHMARKS if n in bench_names],
        "methods": [m.name for m in _make_methods()],
        "budgets": list(budgets),
        "seeds": list(seeds),
        "delta": DELTA,
        "mc_truth_samples": MC_TRUTH_SAMPLES,
        "mc_truth_seed": MC_TRUTH_SEED,
        "total_runs": n_done,
        "total_wall_clock_s": t_total,
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=_json_default))
    print(f"# summary -> {summary_path}")
    return 0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(f"not JSON-serialisable: {type(obj).__name__}")


def _summarise(runs_path: Path) -> dict[str, Any]:
    """Aggregate per (benchmark, method, budget) across seeds."""
    rows: list[dict[str, Any]] = []
    with open(runs_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    by_key: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        k = (r["benchmark"], r["method"], r["budget"])
        by_key.setdefault(k, []).append(r)

    summary_rows: list[dict[str, Any]] = []
    for (bench, method, budget), runs in sorted(by_key.items()):
        half_widths = [r["half_width"] for r in runs]
        mus = [r["mu_hat"] for r in runs]
        samples = [r["samples_used"] for r in runs]
        walls = [r["wall_clock_s"] for r in runs]
        contains = [r["interval_contains_truth"] for r in runs
                    if r["interval_contains_truth"] is not None]
        errs = [r["error_vs_truth"] for r in runs
                if r["error_vs_truth"] is not None]
        summary_rows.append({
            "benchmark": bench,
            "method": method,
            "budget": budget,
            "n_seeds": len(runs),
            "median_mu_hat": statistics.median(mus),
            "median_half_width": statistics.median(half_widths),
            "median_samples": int(statistics.median(samples)),
            "median_wall_clock_s": statistics.median(walls),
            "iqr_half_width": _iqr(half_widths),
            "iqr_samples": _iqr([float(s) for s in samples]),
            "coverage": (sum(1 for c in contains if c) / len(contains))
                        if contains else None,
            "median_error_vs_truth": statistics.median(errs) if errs else None,
            "mc_truth": runs[0]["mc_truth"],
        })
    return {"summary_rows": summary_rows}


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[2] - q[0]


if __name__ == "__main__":
    raise SystemExit(main())
