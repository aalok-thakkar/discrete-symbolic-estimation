"""User-facing :func:`estimate` entry point and :class:`EstimationResult`."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..distributions import Distribution, ProductDistribution
from ..scheduler import (
    ASIPScheduler,
    IterationLog,
    SchedulerConfig,
    SchedulerResult,
)
from ..smt import SMTBackend, default_backend


@dataclass
class EstimationResult:
    """User-facing result of :func:`estimate`."""

    mu_hat: float
    interval: tuple[float, float]
    eps_stat: float
    W_open: float
    delta: float
    samples_used: int
    refinements_done: int
    n_leaves: int
    n_open_leaves: int
    n_closed_leaves: int
    terminated_reason: str
    iterations: list[IterationLog] = field(repr=False, default_factory=list)

    @property
    def half_width(self) -> float:
        lo, hi = self.interval
        return (hi - lo) / 2.0

    def __repr__(self) -> str:
        lo, hi = self.interval
        return (
            f"EstimationResult(mu_hat={self.mu_hat:.4f}, "
            f"interval=[{lo:.4f}, {hi:.4f}], "
            f"eps_stat={self.eps_stat:.4g}, W_open={self.W_open:.4g}, "
            f"samples={self.samples_used}, refinements={self.refinements_done}, "
            f"terminated={self.terminated_reason!r})"
        )


def estimate(
    program: Callable[..., Any],
    distribution: Mapping[str, Distribution],
    property_fn: Callable[[Any], bool],
    epsilon: float = 0.05,
    delta: float = 0.05,
    budget: int = 10_000,
    bootstrap: int = 200,
    batch_size: int = 50,
    seed: int = 0,
    backend: SMTBackend | None = None,
    verbose: bool = False,
    max_refinement_depth: int = 50,
    closure_min_samples: int = 5,
    max_concolic_branches: int = 10_000,
) -> EstimationResult:
    """Run DiSE on ``program`` against ``property_fn`` under ``distribution``.

    Parameters
    ----------
    program:
        A callable taking the variables in ``distribution`` as kwargs and
        returning an int (or a tuple / structure containing ints).
    distribution:
        Map from variable name to a :class:`~dise.distributions.Distribution`.
    property_fn:
        Boolean property of the program's output.
    epsilon:
        Target half-width on the certified interval.
    delta:
        Confidence parameter — interval holds with probability at least
        ``1 - delta``.
    budget:
        Maximum number of concolic samples to run.
    bootstrap:
        Number of initial samples drawn from ``D`` before adaptive
        action selection begins.
    batch_size:
        Number of samples per allocation action.
    seed:
        Seed for the random generator.
    backend:
        SMT backend instance. Defaults to :func:`~dise.smt.default_backend`
        (Z3 if installed, else Mock).
    verbose:
        Pass-through to the scheduler for diagnostic prints (currently a no-op).

    Returns
    -------
    EstimationResult
        Includes ``mu_hat``, the certified ``interval``, and diagnostics.
    """
    if backend is None:
        backend = default_backend()
    rng = np.random.default_rng(seed)
    dist = ProductDistribution(factors=dict(distribution))
    config = SchedulerConfig(
        epsilon=epsilon,
        delta=delta,
        budget_samples=budget,
        bootstrap_samples=bootstrap,
        batch_size=batch_size,
        max_refinement_depth=max_refinement_depth,
        closure_min_samples=closure_min_samples,
        max_concolic_branches=max_concolic_branches,
        verbose=verbose,
    )
    scheduler = ASIPScheduler(
        program=program,
        distribution=dist,
        property_fn=property_fn,
        smt=backend,
        config=config,
        rng=rng,
    )
    sched_result: SchedulerResult = scheduler.run()
    state = sched_result.final_estimator
    return EstimationResult(
        mu_hat=state.mu_hat,
        interval=state.interval,
        eps_stat=state.eps_stat,
        W_open=state.W_open,
        delta=state.delta,
        samples_used=sched_result.samples_used,
        refinements_done=sched_result.refinements_done,
        n_leaves=state.n_leaves,
        n_open_leaves=state.n_open_leaves,
        n_closed_leaves=state.n_closed_leaves,
        terminated_reason=sched_result.terminated_reason,
        iterations=sched_result.iterations,
    )


__all__ = ["EstimationResult", "estimate"]
