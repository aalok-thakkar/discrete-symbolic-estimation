"""DiSE: Discrete Symbolic Estimation.

Distribution-aware reliability estimation for deterministic integer /
bitvector programs under structured discrete operational distributions.

Given a program :math:`P : \\mathcal X \\to \\mathcal Y`, a discrete
operational distribution :math:`D` over :math:`\\mathcal X`, a Boolean
property :math:`\\varphi`, a target half-width
:math:`\\varepsilon \\in (0, 1)`, and a confidence level
:math:`1 - \\delta`, DiSE returns an estimate :math:`\\hat\\mu` of

.. math::

    \\mu \\;=\\; \\Pr_{x \\sim D}\\!\\big[\\varphi(P(x)) = 1\\big]

together with a certified two-sided half-width
:math:`\\varepsilon_{\\text{stat}} + W_{\\text{open}}` such that

.. math::

    \\Pr\\!\\big[\\,|\\hat\\mu - \\mu| \\le \\varepsilon_{\\text{stat}} + W_{\\text{open}}\\,\\big]
    \\;\\ge\\; 1 - \\delta.

See :doc:`/docs/algorithm` for the algorithm (ASIP) and theorems,
:doc:`/docs/tutorial` for a worked example, and
:doc:`/docs/api-reference` for a module-by-module reference.

Quick examples
==============

Output-property framing::

    from dise import estimate, BoundedGeometric

    def gcd_with_steps(a: int, b: int) -> int:
        steps = 0
        while b != 0:
            a, b = b, a % b
            steps += 1
        return steps

    result = estimate(
        program=gcd_with_steps,
        distribution={
            "a": BoundedGeometric(p=0.1, N=100),
            "b": BoundedGeometric(p=0.1, N=100),
        },
        property_fn=lambda steps: steps <= 5,
        epsilon=0.05, delta=0.05,
    )

Assertion-violation framing::

    from dise import failure_probability, Uniform

    def safe_mul(a, b):
        s = a * b
        assert s < (1 << 8), "8-bit overflow"
        return s

    result = failure_probability(
        program=safe_mul,
        distribution={"a": Uniform(1, 31), "b": Uniform(1, 31)},
        epsilon=0.05, delta=0.05,
    )

Soundness-mode (anytime-valid) certificate under adaptive stopping::

    result = estimate(..., method="anytime", budget=None,
                      budget_seconds=600.0)

Public top-level API
====================

Estimation:
    :func:`estimate`, :func:`failure_probability`,
    :class:`EstimationResult`

Distributions:
    :class:`Distribution` (ABC), :class:`Geometric`,
    :class:`BoundedGeometric`, :class:`Uniform`,
    :class:`Categorical`, :class:`Poisson`,
    :class:`ProductDistribution`

SMT backends:
    :class:`SMTBackend` (ABC), :class:`MockBackend`,
    :class:`Z3Backend` (may be ``None`` if z3 is not installed),
    :class:`CachedBackend`, :class:`CacheStats`,
    :func:`default_backend`, :func:`has_z3`

Baselines (for experimental comparison):
    :class:`Baseline` (ABC), :class:`BaselineResult`,
    :class:`PlainMonteCarlo`, :class:`StratifiedRandomMC`,
    :class:`DiSEBaseline`

Specialized internals (import from submodules):
    * :mod:`dise.scheduler` — :class:`ASIPScheduler`,
      :class:`SchedulerConfig`, :class:`SchedulerResult`,
      :class:`IterationLog`
    * :mod:`dise.regions` — :class:`Frontier`, :class:`FrontierNode`,
      :class:`Region`, :class:`AxisAlignedBox`,
      :class:`GeneralRegion`, :class:`Status`, :func:`build_region`
    * :mod:`dise.concolic` — :class:`SymbolicInt`,
      :func:`run_concolic`, :class:`ConcolicResult`,
      :class:`BranchRecord`, :class:`Tracer`
    * :mod:`dise.sampler` — :class:`Sampler`,
      :class:`RejectionSampler`, :class:`IntegerLatticeMHSampler`
    * :mod:`dise.estimator` — :func:`compute_estimator_state`,
      :class:`EstimatorState`, plus the half-width primitives
      (:func:`wilson_halfwidth_for_leaf`,
      :func:`wilson_halfwidth_anytime`,
      :func:`bernstein_halfwidth`,
      :func:`empirical_bernstein_halfwidth_mp`)
    * :mod:`dise.experiment` — :func:`run_experiment`,
      :class:`ExperimentReport`, :class:`RunResult`,
      :class:`MethodAggregate`, :func:`default_methods`
    * :mod:`dise.benchmarks` — :class:`Benchmark`,
      :func:`list_benchmarks`, :func:`get_benchmark`
    * :mod:`dise.integrations.hypothesis` — Hypothesis-strategy
      adapters: :func:`estimate_from_strategy`,
      :func:`estimate_from_strategies`
"""

from __future__ import annotations

__version__ = "0.1.0"

from .baselines import (
    Baseline,
    BaselineResult,
    DiSEBaseline,
    PlainMonteCarlo,
    StratifiedRandomMC,
)
from .distributions import (
    BoundedGeometric,
    Categorical,
    Distribution,
    Geometric,
    Poisson,
    ProductDistribution,
    Uniform,
)
from .estimator.api import EstimationResult, estimate, failure_probability
from .smt import (
    CachedBackend,
    CacheStats,
    MockBackend,
    SMTBackend,
    Z3Backend,  # may be None if z3 not installed
    default_backend,
    has_z3,
)

__all__ = [
    "__version__",
    # estimation entry points
    "estimate",
    "failure_probability",
    "EstimationResult",
    # distributions
    "Distribution",
    "Geometric",
    "BoundedGeometric",
    "Uniform",
    "Categorical",
    "Poisson",
    "ProductDistribution",
    # SMT backends
    "SMTBackend",
    "MockBackend",
    "Z3Backend",
    "CachedBackend",
    "CacheStats",
    "default_backend",
    "has_z3",
    # baselines
    "Baseline",
    "BaselineResult",
    "PlainMonteCarlo",
    "StratifiedRandomMC",
    "DiSEBaseline",
]
