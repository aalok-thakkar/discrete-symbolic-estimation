"""DiSE: Discrete Symbolic Estimation.

Distribution-aware reliability estimation for deterministic integer /
bitvector programs under structured discrete operational distributions.

Quickstart::

    from dise import estimate, BoundedGeometric

    def gcd_with_steps(a: int, b: int) -> int:
        steps = 0
        while b != 0:
            a, b = b, a % b
            steps += 1
        return steps

    result = estimate(
        program=lambda a, b: gcd_with_steps(a, b),
        distribution={
            "a": BoundedGeometric(p=0.1, N=100),
            "b": BoundedGeometric(p=0.1, N=100),
        },
        property_fn=lambda steps: steps <= 5,
        epsilon=0.05,
        delta=0.05,
        budget=5000,
    )
    print(result)
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
