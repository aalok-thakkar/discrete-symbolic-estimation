"""Property-based tests using Hypothesis.

These tests check structural invariants that should hold across the
distributions, frontier, and estimator state under any randomized input.
"""

from __future__ import annotations

import math

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from dise.distributions import BoundedGeometric, Categorical, Geometric, Uniform
from dise.estimator import (
    bernstein_halfwidth,
    empirical_bernstein_halfwidth_mp,
    wilson_halfwidth_for_leaf,
)

# ---------------------------------------------------------------------------
# Distributions: PMF non-negative, normalizes (on bounded supports)
# ---------------------------------------------------------------------------


@given(p=st.floats(min_value=1e-3, max_value=0.99))
def test_geometric_pmf_nonneg_and_sums_to_one(p: float) -> None:
    d = Geometric(p=p)
    s = sum(d.pmf(k) for k in range(1, 2000))
    assert s > 0.0
    assert s <= 1.0 + 1e-9
    # tail mass is bounded
    a, b = d.support_bounds(1e-9)
    assert d.mass(a, b) >= 1.0 - 1e-9


@given(
    p=st.floats(min_value=1e-3, max_value=0.99),
    N=st.integers(min_value=1, max_value=200),
)
def test_bounded_geometric_normalizes(p: float, N: int) -> None:
    d = BoundedGeometric(p=p, N=N)
    s = sum(d.pmf(k) for k in range(1, N + 1))
    assert math.isclose(s, 1.0, abs_tol=1e-9)


@given(
    lo=st.integers(min_value=-50, max_value=50),
    span=st.integers(min_value=0, max_value=20),
)
def test_uniform_normalizes(lo: int, span: int) -> None:
    hi = lo + span
    d = Uniform(lo=lo, hi=hi)
    s = sum(d.pmf(k) for k in range(lo, hi + 1))
    assert math.isclose(s, 1.0, abs_tol=1e-12)


@given(
    weights=st.lists(
        st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10,
    )
)
def test_categorical_normalizes(weights: list[float]) -> None:
    total = sum(weights)
    if total <= 0:
        return  # skip degenerate
    probs = tuple(w / total for w in weights)
    # Re-normalize to fight float error so __post_init__ accepts.
    s = sum(probs)
    if not math.isclose(s, 1.0, abs_tol=1e-9):
        probs = tuple(p / s for p in probs)
    d = Categorical(probs=probs)
    assert math.isclose(sum(d.pmf(k) for k in range(len(probs))), 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Distribution invariants over CDF/mass
# ---------------------------------------------------------------------------


@given(
    p=st.floats(min_value=1e-3, max_value=0.99),
    N=st.integers(min_value=10, max_value=200),
    lo=st.integers(min_value=1, max_value=50),
    span=st.integers(min_value=0, max_value=50),
)
def test_bounded_geometric_mass_matches_cdf(p: float, N: int, lo: int, span: int) -> None:
    d = BoundedGeometric(p=p, N=N)
    hi = lo + span
    assert math.isclose(d.mass(lo, hi), d.cdf(hi) - d.cdf(lo - 1), abs_tol=1e-12)


@given(
    p=st.floats(min_value=1e-3, max_value=0.99),
    N=st.integers(min_value=10, max_value=100),
)
def test_bounded_geometric_cdf_monotone(p: float, N: int) -> None:
    d = BoundedGeometric(p=p, N=N)
    prev = -1.0
    for k in range(N + 2):
        cur = d.cdf(k)
        assert cur >= prev - 1e-12
        prev = cur


# ---------------------------------------------------------------------------
# Estimator primitives
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=1, max_value=1000),
    h_frac=st.floats(min_value=0.0, max_value=1.0),
    delta=st.floats(min_value=1e-3, max_value=0.5),
)
def test_wilson_halfwidth_in_unit_interval(n: int, h_frac: float, delta: float) -> None:
    h = int(round(h_frac * n))
    h = max(0, min(h, n))
    half = wilson_halfwidth_for_leaf(n, h, delta)
    assert half > 0.0
    assert half <= 1.0


@given(
    variance=st.floats(min_value=0.0, max_value=1.0),
    delta=st.floats(min_value=1e-3, max_value=0.5),
    b=st.floats(min_value=1e-6, max_value=1.0),
)
def test_bernstein_halfwidth_nonneg(variance: float, delta: float, b: float) -> None:
    h = bernstein_halfwidth(variance, delta, per_sample_bound=b)
    assert h > 0.0


@given(
    v=st.floats(min_value=0.0, max_value=1.0),
    n=st.integers(min_value=2, max_value=10_000),
    delta=st.floats(min_value=1e-3, max_value=0.5),
)
def test_mp_eb_halfwidth_shrinks_with_n(v: float, n: int, delta: float) -> None:
    """Maurer-Pontil EB half-width is non-increasing as n grows (other args fixed)."""
    h_small = empirical_bernstein_halfwidth_mp(v, n, delta)
    h_big = empirical_bernstein_halfwidth_mp(v, n * 2, delta)
    # Allow tiny FP slack.
    assert h_big <= h_small + 1e-9


# ---------------------------------------------------------------------------
# Sampling invariants
# ---------------------------------------------------------------------------


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    p=st.floats(min_value=0.05, max_value=0.5),
    lo=st.integers(min_value=1, max_value=10),
    span=st.integers(min_value=0, max_value=10),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_geometric_truncated_in_bounds(p: float, lo: int, span: int, seed: int) -> None:
    d = Geometric(p=p)
    hi = lo + span
    rng = np.random.default_rng(seed)
    samples = d.sample_truncated(rng, lo, hi, 200)
    assert samples.min() >= lo
    assert samples.max() <= hi
