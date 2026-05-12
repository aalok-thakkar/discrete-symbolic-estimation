"""Experiment runner: aggregate DiSE and baseline runs across multiple seeds.

Public entry points:

* :func:`run_method` — execute one estimator on one benchmark with one seed.
* :func:`run_experiment` — Cartesian product over methods × seeds, return
  a structured :class:`ExperimentReport` with per-run + aggregate stats.
* :func:`save_report` / :func:`load_report` — JSON serialization.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from ..baselines import Baseline, BaselineResult, DiSEBaseline, PlainMonteCarlo, StratifiedRandomMC
from ..distributions import Distribution, ProductDistribution


@dataclass
class RunResult:
    """One row in the experiment table — one method × one benchmark × one seed."""

    benchmark: str
    method: str
    seed: int
    budget: int
    delta: float
    mu_hat: float
    interval: tuple[float, float]
    half_width: float
    samples_used: int
    wall_clock_s: float
    mc_truth: float | None
    interval_contains_truth: bool | None
    error_vs_truth: float | None
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_baseline_result(
        cls,
        result: BaselineResult,
        benchmark: str,
        budget: int,
        seed: int,
        mc_truth: float | None,
    ) -> RunResult:
        lo, hi = result.interval
        contains: bool | None = None
        err: float | None = None
        if mc_truth is not None:
            contains = lo <= mc_truth <= hi
            err = abs(result.mu_hat - mc_truth)
        return cls(
            benchmark=benchmark,
            method=result.name,
            seed=seed,
            budget=budget,
            delta=result.delta,
            mu_hat=result.mu_hat,
            interval=result.interval,
            half_width=result.half_width,
            samples_used=result.samples_used,
            wall_clock_s=result.wall_clock_s,
            mc_truth=mc_truth,
            interval_contains_truth=contains,
            error_vs_truth=err,
            extras=dict(result.extras),
        )


@dataclass
class MethodAggregate:
    """Per-method summary stats across seeds, for one benchmark."""

    benchmark: str
    method: str
    n_seeds: int
    median_mu_hat: float
    median_half_width: float
    median_samples: int
    median_wall_clock_s: float
    coverage: float | None  # fraction of seeds whose interval contains MC truth
    median_error_vs_truth: float | None
    iqr_half_width: float
    iqr_samples: float


@dataclass
class ExperimentReport:
    """Output of :func:`run_experiment`."""

    benchmark: str
    description: str
    mc_truth: float | None
    mc_se: float | None
    runs: list[RunResult]
    aggregates: list[MethodAggregate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "description": self.description,
            "mc_truth": self.mc_truth,
            "mc_se": self.mc_se,
            "runs": [asdict(r) for r in self.runs],
            "aggregates": [asdict(a) for a in self.aggregates],
        }


def save_report(report: ExperimentReport, path: str) -> None:
    """Serialize an :class:`ExperimentReport` to JSON at ``path``.

    Lossless modulo the floating-point representation used by
    :mod:`json`. Round-trip via :func:`load_report` returns a plain
    ``dict[str, Any]`` rather than reconstructing the dataclasses —
    downstream tools (plotting, table generation) consume the dict
    form directly.

    Parameters
    ----------
    report : ExperimentReport
        The report to persist.
    path : str
        Filesystem path; the file is overwritten if it exists.
    """
    with open(path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)


def load_report(path: str) -> dict[str, Any]:
    """Load a JSON experiment report from ``path``.

    Returns the *dict-of-dicts* form (as written by
    :func:`save_report`), not an :class:`ExperimentReport`
    dataclass instance — round-trip reconstruction is intentionally
    asymmetric so consumers (e.g. ``dise plot``) can read reports
    without importing :mod:`dise.experiment`.

    Parameters
    ----------
    path : str
        Filesystem path to a JSON file written by :func:`save_report`.

    Returns
    -------
    dict[str, Any]
        The deserialized report. Keys: ``benchmark``, ``description``,
        ``mc_truth``, ``mc_se``, ``runs``, ``aggregates``.
    """
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def ground_truth_mc(
    program: Callable[..., Any],
    distribution: Mapping[str, Distribution],
    property_fn: Callable[[Any], bool],
    n_samples: int,
    seed: int = 12_345,
) -> tuple[float, float]:
    """Plain Monte-Carlo reference estimate of :math:`\\mu`.

    Used as a *regression check*, not a certified bound. Returns the
    point estimate together with its analytic standard error
    :math:`\\sqrt{\\hat\\mu(1 - \\hat\\mu) / n}`. Downstream
    aggregators compare DiSE's certified interval against
    :math:`\\hat\\mu_{\\text{MC}} \\pm 1.96 \\,\\mathrm{se}` to detect
    severe coverage failures.

    Parameters
    ----------
    program : Callable
        Program under test.
    distribution : Mapping[str, Distribution]
        Operational distribution.
    property_fn : Callable[[Any], bool]
        Boolean property of the program's output.
    n_samples : int
        Sample count (often 20 000 in the paper's tables).
    seed : int, default 12345
        Independent RNG seed so the reference is reproducible across
        all methods being compared.

    Returns
    -------
    tuple[float, float]
        ``(mu, se)``.
    """
    rng = np.random.default_rng(seed)
    dist = ProductDistribution(factors=dict(distribution))
    batch = dist.sample(rng, n_samples)
    keys = list(distribution.keys())
    hits = 0
    for i in range(n_samples):
        x = {k: int(batch[k][i]) for k in keys}
        if bool(property_fn(program(**x))):
            hits += 1
    mu = hits / n_samples
    se = (mu * (1 - mu) / n_samples) ** 0.5
    return mu, se


def run_method(
    method: Baseline,
    program: Callable[..., Any],
    distribution: Mapping[str, Distribution],
    property_fn: Callable[[Any], bool],
    budget: int,
    delta: float,
    seed: int,
    benchmark_name: str,
    mc_truth: float | None,
) -> RunResult:
    """Run one method on one ``(program, distribution, property)`` triple.

    Calls :meth:`Baseline.run`, measures wall-clock time (taking the
    larger of the baseline's self-reported time and the externally
    measured time), and wraps the result in a :class:`RunResult` —
    the row type stored in :class:`ExperimentReport`.

    Parameters
    ----------
    method : Baseline
        The comparator (e.g. :class:`PlainMonteCarlo`,
        :class:`DiSEBaseline`).
    program, distribution, property_fn :
        Forwarded to :meth:`Baseline.run`.
    budget, delta, seed :
        Forwarded to :meth:`Baseline.run`.
    benchmark_name : str
        Identifier copied into the :class:`RunResult` for downstream
        aggregation.
    mc_truth : float or None
        Reference value used to populate
        :attr:`RunResult.interval_contains_truth` and
        :attr:`RunResult.error_vs_truth`. Pass ``None`` to skip those
        checks.

    Returns
    -------
    RunResult
    """
    t0 = time.perf_counter()
    result = method.run(
        program=program,
        distribution=distribution,
        property_fn=property_fn,
        budget=budget,
        delta=delta,
        seed=seed,
    )
    elapsed = time.perf_counter() - t0
    # Update wall_clock if the method didn't measure it.
    if result.wall_clock_s <= 0.0:
        result = BaselineResult(
            name=result.name,
            mu_hat=result.mu_hat,
            interval=result.interval,
            samples_used=result.samples_used,
            wall_clock_s=elapsed,
            delta=result.delta,
            extras=result.extras,
        )
    return RunResult.from_baseline_result(
        result,
        benchmark=benchmark_name,
        budget=budget,
        seed=seed,
        mc_truth=mc_truth,
    )


def _iqr(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    q = statistics.quantiles(values, n=4, method="inclusive")
    return q[2] - q[0]


def _aggregate(
    benchmark: str, method: str, runs: list[RunResult]
) -> MethodAggregate:
    half_widths = [r.half_width for r in runs]
    samples = [r.samples_used for r in runs]
    walls = [r.wall_clock_s for r in runs]
    mus = [r.mu_hat for r in runs]
    errors = [r.error_vs_truth for r in runs if r.error_vs_truth is not None]
    contains = [r.interval_contains_truth for r in runs if r.interval_contains_truth is not None]
    coverage = (sum(1 for c in contains if c) / len(contains)) if contains else None
    return MethodAggregate(
        benchmark=benchmark,
        method=method,
        n_seeds=len(runs),
        median_mu_hat=statistics.median(mus),
        median_half_width=statistics.median(half_widths),
        median_samples=int(statistics.median(samples)),
        median_wall_clock_s=statistics.median(walls),
        coverage=coverage,
        median_error_vs_truth=statistics.median(errors) if errors else None,
        iqr_half_width=_iqr(half_widths),
        iqr_samples=_iqr([float(s) for s in samples]),
    )


def run_experiment(
    benchmark_name: str,
    description: str,
    program: Callable[..., Any],
    distribution: Mapping[str, Distribution],
    property_fn: Callable[[Any], bool],
    methods: Iterable[Baseline],
    budget: int = 5000,
    delta: float = 0.05,
    seeds: Iterable[int] = (0, 1, 2, 3, 4),
    mc_samples: int = 20_000,
    mc_seed: int = 12_345,
    skip_mc: bool = False,
) -> ExperimentReport:
    """Run each method on the same benchmark over multiple seeds.

    Returns a :class:`ExperimentReport` with per-run rows and per-method
    aggregates (median + IQR).
    """
    if skip_mc:
        mc_truth: float | None = None
        mc_se: float | None = None
    else:
        mc_truth, mc_se = ground_truth_mc(
            program, distribution, property_fn, mc_samples, seed=mc_seed
        )

    seeds_list = list(seeds)
    method_list = list(methods)
    runs: list[RunResult] = []
    for seed in seeds_list:
        for method in method_list:
            row = run_method(
                method=method,
                program=program,
                distribution=distribution,
                property_fn=property_fn,
                budget=budget,
                delta=delta,
                seed=seed,
                benchmark_name=benchmark_name,
                mc_truth=mc_truth,
            )
            runs.append(row)

    # Aggregate per method
    aggregates: list[MethodAggregate] = []
    for method in method_list:
        method_runs = [r for r in runs if r.method == method.name]
        aggregates.append(_aggregate(benchmark_name, method.name, method_runs))
    return ExperimentReport(
        benchmark=benchmark_name,
        description=description,
        mc_truth=mc_truth,
        mc_se=mc_se,
        runs=runs,
        aggregates=aggregates,
    )


def default_methods(
    budget: int,
    bootstrap: int = 200,
    batch_size: int = 50,
    epsilon: float = 0.05,
    n_strata: int = 16,
) -> list[Baseline]:
    """Return the canonical comparator set used in the paper's tables.

    Includes :class:`PlainMonteCarlo`, :class:`StratifiedRandomMC`
    (with ``n_strata`` hash buckets), and :class:`DiSEBaseline` with
    the given ``bootstrap``, ``batch_size``, and ``epsilon`` knobs.
    The ``budget`` parameter is accepted for symmetry with the
    individual constructors but is ultimately set by the experiment
    runner per call.

    Parameters
    ----------
    budget : int
        Total sample budget per run; mirrored from the caller for
        documentation. Each :class:`Baseline.run` receives the same
        budget when invoked by :func:`run_experiment`.
    bootstrap : int, default 200
        Bootstrap samples for the DiSE baseline.
    batch_size : int, default 50
        Per-allocation batch size for the DiSE baseline.
    epsilon : float, default 0.05
        Target half-width for DiSE; controls when
        ``terminated_reason='epsilon_reached'`` fires.
    n_strata : int, default 16
        Number of random hash buckets for the stratified-MC baseline.

    Returns
    -------
    list[Baseline]
        Ordered ``[plain_mc, stratified_random, dise]``.
    """
    return [
        PlainMonteCarlo(),
        StratifiedRandomMC(n_strata=n_strata),
        DiSEBaseline(
            bootstrap=bootstrap,
            batch_size=batch_size,
            epsilon=epsilon,
        ),
    ]


__all__ = [
    "ExperimentReport",
    "MethodAggregate",
    "RunResult",
    "default_methods",
    "ground_truth_mc",
    "load_report",
    "run_experiment",
    "run_method",
    "save_report",
]
