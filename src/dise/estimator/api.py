"""User-facing entry points and result type.

* :func:`estimate` — the general entry point. Takes a Boolean property
  on the program's output.
* :func:`failure_probability` — convenience wrapper for the classical
  *assertion-violation* framing: estimate
  :math:`\\Pr_D[P(x) \\text{ raises AssertionError}]` (or any
  user-specified exception class).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

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
    budget: int | None = 10_000,
    budget_seconds: float | None = None,
    min_gain_per_cost: float = 0.0,
    method: Literal["wilson", "anytime", "bernstein", "empirical-bernstein"] = "wilson",
    bootstrap: int = 200,
    batch_size: int = 50,
    seed: int = 0,
    backend: SMTBackend | None = None,
    verbose: bool = False,
    max_refinement_depth: int = 50,
    closure_min_samples: int = 5,
    delta_close: float = 0.005,
    closure_epsilon: float = 0.02,
    max_concolic_branches: int = 10_000,
) -> EstimationResult:
    r"""Run DiSE on ``program`` against ``property_fn`` under ``distribution``.

    Estimate :math:`\mu = \Pr_D[\varphi(P(x)) = 1]` and return a
    certified two-sided interval that contains :math:`\mu` with
    probability at least :math:`1 - \delta` (modulo the soundness of
    the SMT backend; see :doc:`docs/algorithm`).

    Parameters
    ----------
    program:
        Callable taking the variables in ``distribution`` as kwargs and
        returning an int (or a tuple / structure containing ints).
    distribution:
        Map from variable name to a :class:`~dise.distributions.Distribution`.
    property_fn:
        Boolean property of the program's output.
    epsilon:
        Target half-width on the certified interval (always active —
        the *primary* termination condition).
    delta:
        Confidence parameter — interval covers truth with probability
        at least ``1 - delta``.
    budget:
        Optional cap on concolic runs. ``None`` disables the cap and
        relies on ``epsilon`` (and the gain-per-cost floor) for
        termination. Default ``10_000`` is conservative; for
        soundness-mode runs pass ``budget=None``.
    budget_seconds:
        Optional wall-clock cap in seconds. ``None`` disables.
    min_gain_per_cost:
        Threshold below which the algorithm declares no positive-gain
        action exists (diminishing-returns floor). Default ``0``
        preserves the brief's strict semantics; e.g. ``1e-9`` makes
        unbounded runs terminate as soon as further work is wasted.
    method:
        Certified half-width construction (see
        :func:`~dise.estimator.compute_estimator_state`). ``"wilson"``
        (default) is tightest at fixed sample counts; ``"anytime"``
        is the time-uniform variant sound under ASIP's adaptive
        stopping rule (recommended for ATVA-style certificates,
        see :doc:`docs/algorithm` §13).
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
    max_refinement_depth:
        Maximum depth in the frontier tree (caps refinement recursion).
    closure_min_samples:
        Minimum samples at a leaf before sample-based closure can fire.
    delta_close:
        Per-leaf confidence for sound concentration-bounded sample
        closure. The closure rule uses a Wilson-anytime upper bound on
        the leaf's disagreement rate at this confidence level. Default
        ``0.005``.
    closure_epsilon:
        Maximum disagreement rate the sample-based closure rule admits.
        Each sample-based closure adds ``closure_epsilon * w_leaf`` to
        the certified-interval half-width (via the ``W_close``
        accumulator on :class:`~dise.regions.Frontier`). SMT-verified
        closures contribute zero. Default ``0.02``.
    max_concolic_branches:
        Per-run cap on the number of branches the concolic tracer records.

    Returns
    -------
    EstimationResult
        Includes ``mu_hat``, the certified ``interval``, and diagnostics.

    See Also
    --------
    failure_probability : convenience wrapper for assertion-violation
        properties.
    """
    if backend is None:
        backend = default_backend()
    rng = np.random.default_rng(seed)
    dist = ProductDistribution(factors=dict(distribution))
    config = SchedulerConfig(
        epsilon=epsilon,
        delta=delta,
        budget_samples=budget,
        budget_seconds=budget_seconds,
        min_gain_per_cost=min_gain_per_cost,
        method=method,
        bootstrap_samples=bootstrap,
        batch_size=batch_size,
        max_refinement_depth=max_refinement_depth,
        closure_min_samples=closure_min_samples,
        delta_close=delta_close,
        closure_epsilon=closure_epsilon,
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


def failure_probability(
    program: Callable[..., Any],
    distribution: Mapping[str, Distribution],
    *,
    catch: type[BaseException] | tuple[type[BaseException], ...] = AssertionError,
    epsilon: float = 0.05,
    delta: float = 0.05,
    budget: int | None = None,
    budget_seconds: float | None = None,
    min_gain_per_cost: float = 0.0,
    method: Literal["wilson", "anytime", "bernstein", "empirical-bernstein"] = "wilson",
    bootstrap: int = 200,
    batch_size: int = 50,
    seed: int = 0,
    backend: SMTBackend | None = None,
    verbose: bool = False,
    max_refinement_depth: int = 50,
    closure_min_samples: int = 5,
    delta_close: float = 0.005,
    closure_epsilon: float = 0.02,
    max_concolic_branches: int = 10_000,
) -> EstimationResult:
    r"""Estimate :math:`\Pr_D[P \text{ raises an exception of type } \texttt{catch}]`.

    The classical assertion-violation framing of probabilistic program
    verification: given a program with assertions (or any guarded
    operations that raise a specific exception class), compute the
    *failure probability* under an operational distribution ``D``.

    Internally wraps ``program`` so that exceptions of the specified
    type become Boolean failures (``output == 1``) and delegates to
    :func:`estimate`. The default ``catch=AssertionError`` covers the
    canonical ``assert`` use case; pass a tuple of exception classes
    (e.g. ``(AssertionError, ValueError)``) to broaden the failure
    semantics. Exceptions outside ``catch`` are *not* caught — they
    propagate, indicating a real bug in the harness.

    .. warning::

       The default ``budget=None`` is *unlimited*. For
       unconditionally-bounded runs, configure at least one of
       ``budget`` (sample cap), ``budget_seconds`` (wall-clock cap),
       or ``min_gain_per_cost > 0`` (diminishing-returns floor). On
       a pathologically hard target with all three permissive the
       algorithm may not terminate.

    See :func:`estimate` for the full parameter reference; this
    wrapper accepts the same keyword arguments.

    Examples
    --------

    >>> from dise import failure_probability, Uniform
    >>> def safe_mul(a, b):
    ...     s = a * b
    ...     assert s < (1 << 8), "overflow"
    ...     return s
    >>> result = failure_probability(
    ...     program=safe_mul,
    ...     distribution={"a": Uniform(1, 31), "b": Uniform(1, 31)},
    ...     epsilon=0.05,
    ...     budget=2000,                # bound the run for the example
    ... )                                                # doctest: +SKIP
    """

    def wrapped(**kw: int) -> int:
        try:
            program(**kw)
            return 0  # success: no assertion violated
        except catch:
            return 1  # failure

    return estimate(
        program=wrapped,
        distribution=distribution,
        property_fn=lambda v: v == 1,
        epsilon=epsilon,
        delta=delta,
        budget=budget,
        budget_seconds=budget_seconds,
        min_gain_per_cost=min_gain_per_cost,
        method=method,
        bootstrap=bootstrap,
        batch_size=batch_size,
        seed=seed,
        backend=backend,
        verbose=verbose,
        max_refinement_depth=max_refinement_depth,
        closure_min_samples=closure_min_samples,
        delta_close=delta_close,
        closure_epsilon=closure_epsilon,
        max_concolic_branches=max_concolic_branches,
    )


__all__ = ["EstimationResult", "estimate", "failure_probability"]
