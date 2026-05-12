"""Reference baselines for empirical comparison against DiSE.

Each baseline implements the same interface as :func:`dise.estimate` —
take a program, distribution, property, accuracy parameters, and budget;
return a :class:`BaselineResult` carrying ``mu_hat``, a certified
interval, sample-cost, and wall-clock time. They are intended to be
drop-in comparators for the experiment runner in :mod:`dise.experiment`.

Two baselines are provided:

* :class:`PlainMonteCarlo` — vanilla Monte Carlo against the unmodified
  program with a Hoeffding-Bentkus-style certified interval (using the
  Wilson score interval as a representative).
* :class:`StratifiedRandomMC` — random stratification by hashing inputs
  into a fixed number of buckets, with per-bucket Wilson intervals
  combined via Bonferroni. A reasonable straw-man "stratified MC
  without symbolic guidance".

These deliberately do *no* symbolic reasoning and serve as the floor
that DiSE has to beat.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..distributions import Distribution, ProductDistribution
from ..estimator import wilson_halfwidth_for_leaf


@dataclass
class BaselineResult:
    """Common return type for all baselines (mirrors :class:`EstimationResult`)."""

    name: str
    mu_hat: float
    interval: tuple[float, float]
    samples_used: int
    wall_clock_s: float
    delta: float
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def half_width(self) -> float:
        lo, hi = self.interval
        return (hi - lo) / 2.0

    def __repr__(self) -> str:
        lo, hi = self.interval
        return (
            f"BaselineResult(name={self.name!r}, mu_hat={self.mu_hat:.4f}, "
            f"interval=[{lo:.4f}, {hi:.4f}], samples={self.samples_used}, "
            f"wall_clock={self.wall_clock_s:.3f}s)"
        )


class Baseline(ABC):
    """Common protocol shared by DiSE and its comparators.

    Subclasses provide a class-level ``name`` (used as the row label
    in :class:`~dise.experiment.ExperimentReport`) and implement
    :meth:`run` to take a program, distribution, property, budget,
    confidence parameter, and seed, returning a
    :class:`BaselineResult`.

    The protocol is intentionally narrow: anything that can answer
    "given these inputs, here is :math:`\\hat\\mu` and a certified
    interval" can be a baseline. :func:`dise.experiment.run_experiment`
    iterates uniformly over baselines, so adding a new comparator is
    just subclassing :class:`Baseline` and including the instance in
    the ``methods`` list.
    """

    #: Short identifier (e.g. ``"plain_mc"``); appears in reports.
    name: str

    @abstractmethod
    def run(
        self,
        program: Callable[..., Any],
        distribution: Mapping[str, Distribution],
        property_fn: Callable[[Any], bool],
        budget: int,
        delta: float,
        seed: int,
    ) -> BaselineResult:
        """Run the estimator on one ``(program, distribution, property)``
        triple at a fixed seed.

        Parameters
        ----------
        program : Callable
            The program under test.
        distribution : Mapping[str, Distribution]
            Operational input distribution; one factor per kwarg of
            ``program``.
        property_fn : Callable[[Any], bool]
            Boolean predicate on the program's output.
        budget : int
            Total per-run sample budget.
        delta : float
            Confidence parameter; the returned interval must cover
            :math:`\\mu` with probability :math:`\\ge 1 - \\delta`.
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        BaselineResult
            ``mu_hat``, certified ``interval``, ``samples_used``,
            ``wall_clock_s``, and method-specific ``extras``.
        """


# ---------------------------------------------------------------------------
# Plain Monte Carlo with a Wilson certified interval
# ---------------------------------------------------------------------------


class PlainMonteCarlo(Baseline):
    """Vanilla Monte Carlo with a certified Wilson interval.

    Draws ``budget`` samples from ``D``, computes
    ``mu_hat = hits / budget``, and reports a Wilson-score two-sided
    interval at confidence ``1 - delta``. The interval is exact for
    Bernoulli observations (modulo finite-sample bias absorbed by
    the Wilson smoothing).
    """

    name = "plain_mc"

    def run(
        self,
        program: Callable[..., Any],
        distribution: Mapping[str, Distribution],
        property_fn: Callable[[Any], bool],
        budget: int,
        delta: float,
        seed: int,
    ) -> BaselineResult:
        rng = np.random.default_rng(seed)
        dist = ProductDistribution(factors=dict(distribution))
        batch = dist.sample(rng, budget)
        keys = list(distribution.keys())
        hits = 0
        t0 = time.perf_counter()
        for i in range(budget):
            x = {k: int(batch[k][i]) for k in keys}
            out = program(**x)
            if bool(property_fn(out)):
                hits += 1
        wall = time.perf_counter() - t0
        mu_hat = hits / budget if budget > 0 else 0.0
        half = wilson_halfwidth_for_leaf(budget, hits, delta)
        lo = max(0.0, mu_hat - half)
        hi = min(1.0, mu_hat + half)
        return BaselineResult(
            name=self.name,
            mu_hat=mu_hat,
            interval=(lo, hi),
            samples_used=budget,
            wall_clock_s=wall,
            delta=delta,
            extras={"hits": hits, "half_width": half},
        )


# ---------------------------------------------------------------------------
# Stratified Monte Carlo with random hash-based bucketing
# ---------------------------------------------------------------------------


class StratifiedRandomMC(Baseline):
    """Stratified MC with a fixed number of *random* (hash-based) strata.

    For each sample, we deterministically map it to one of ``n_strata``
    buckets by hashing the input. We approximate the bucket weight
    ``w_k`` empirically from the bucket counts (so the strata are
    *post-stratified*) and compute a per-bucket Wilson half-width.

    This is intended as a sanity baseline showing that *adaptive*
    stratification matters: the strata here carry no information about
    the program's control flow.
    """

    name = "stratified_random"

    def __init__(self, n_strata: int = 16) -> None:
        if n_strata < 1:
            raise ValueError("n_strata must be >= 1")
        self.n_strata = n_strata

    def run(
        self,
        program: Callable[..., Any],
        distribution: Mapping[str, Distribution],
        property_fn: Callable[[Any], bool],
        budget: int,
        delta: float,
        seed: int,
    ) -> BaselineResult:
        rng = np.random.default_rng(seed)
        dist = ProductDistribution(factors=dict(distribution))
        batch = dist.sample(rng, budget)
        keys = list(distribution.keys())
        counts = [0] * self.n_strata
        hits = [0] * self.n_strata
        t0 = time.perf_counter()
        # Use a fast, deterministic hash for stratum assignment.
        for i in range(budget):
            x = {k: int(batch[k][i]) for k in keys}
            h = hash(tuple(sorted(x.items())))
            k_idx = (h % self.n_strata + self.n_strata) % self.n_strata
            counts[k_idx] += 1
            out = program(**x)
            if bool(property_fn(out)):
                hits[k_idx] += 1
        wall = time.perf_counter() - t0
        n_nonempty = sum(1 for c in counts if c > 0)
        delta_per = delta / max(n_nonempty, 1)
        mu_hat = 0.0
        eps_stat = 0.0
        for k in range(self.n_strata):
            n_k = counts[k]
            if n_k == 0:
                continue
            w_k = n_k / budget
            p_k = hits[k] / n_k
            mu_hat += w_k * p_k
            eps_stat += w_k * wilson_halfwidth_for_leaf(n_k, hits[k], delta_per)
        eps_stat = min(eps_stat, 1.0)
        lo = max(0.0, mu_hat - eps_stat)
        hi = min(1.0, mu_hat + eps_stat)
        return BaselineResult(
            name=self.name,
            mu_hat=mu_hat,
            interval=(lo, hi),
            samples_used=budget,
            wall_clock_s=wall,
            delta=delta,
            extras={
                "n_strata": self.n_strata,
                "n_nonempty": n_nonempty,
                "bucket_counts": counts,
                "bucket_hits": hits,
            },
        )


# ---------------------------------------------------------------------------
# Convenience helper: DiSE wrapped as a Baseline (for shared experiment loops)
# ---------------------------------------------------------------------------


class DiSEBaseline(Baseline):
    """Adapter exposing :func:`dise.estimate` through the
    :class:`Baseline` protocol so :func:`dise.experiment.run_experiment`
    can iterate uniformly across methods.

    All keyword arguments accepted by :func:`dise.estimate` (e.g.
    ``epsilon``, ``method``, ``bootstrap``, ``batch_size``,
    ``backend``, ``budget_seconds``, ``min_gain_per_cost``,
    ``max_refinement_depth``) can be passed to the constructor and
    will be forwarded on every :meth:`run` call. The three
    Baseline-protocol-mandated kwargs (``budget``, ``delta``, ``seed``)
    are supplied by the runner and override any matching constructor
    kwargs.

    Examples
    --------
    >>> b = DiSEBaseline(epsilon=0.02, method="anytime", bootstrap=200)
    >>> # In a run loop:
    >>> # result = b.run(program=p, distribution=d, property_fn=phi,
    >>> #                budget=5000, delta=0.05, seed=0)
    """

    name = "dise"

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    def run(
        self,
        program: Callable[..., Any],
        distribution: Mapping[str, Distribution],
        property_fn: Callable[[Any], bool],
        budget: int,
        delta: float,
        seed: int,
    ) -> BaselineResult:
        """Run :func:`dise.estimate` with the constructor's stored
        kwargs (overridden by ``budget``, ``delta``, ``seed``)."""
        from ..estimator.api import estimate

        t0 = time.perf_counter()
        result = estimate(
            program=program,
            distribution=distribution,
            property_fn=property_fn,
            budget=budget,
            delta=delta,
            seed=seed,
            **{k: v for k, v in self._kwargs.items() if k not in ("budget", "delta", "seed")},
        )
        wall = time.perf_counter() - t0
        return BaselineResult(
            name=self.name,
            mu_hat=result.mu_hat,
            interval=result.interval,
            samples_used=result.samples_used,
            wall_clock_s=wall,
            delta=delta,
            extras={
                "refinements": result.refinements_done,
                "n_leaves": result.n_leaves,
                "n_open_leaves": result.n_open_leaves,
                "eps_stat": result.eps_stat,
                "W_open": result.W_open,
                "terminated_reason": result.terminated_reason,
            },
        )


__all__ = [
    "Baseline",
    "BaselineResult",
    "DiSEBaseline",
    "PlainMonteCarlo",
    "StratifiedRandomMC",
]
