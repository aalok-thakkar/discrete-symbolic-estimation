"""Tests for ``dise.distributions``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dise.distributions import (
    BoundedGeometric,
    Categorical,
    Geometric,
    Poisson,
    ProductDistribution,
    Uniform,
)

# ----------------------------------------------------------------------------
# Geometric
# ----------------------------------------------------------------------------


def test_geometric_pmf_normalizes() -> None:
    d = Geometric(p=0.3)
    s = sum(d.pmf(k) for k in range(1, 500))
    assert math.isclose(s, 1.0, abs_tol=1e-9)


def test_geometric_pmf_zero_outside_support() -> None:
    d = Geometric(p=0.3)
    assert d.pmf(0) == 0.0
    assert d.pmf(-3) == 0.0


def test_geometric_mass_matches_cdf() -> None:
    d = Geometric(p=0.4)
    for lo, hi in [(1, 5), (3, 10), (1, 1), (5, 5), (7, 20), (1, 100)]:
        assert math.isclose(
            d.mass(lo, hi), d.cdf(hi) - d.cdf(lo - 1), abs_tol=1e-12
        )


def test_geometric_mass_random_intervals() -> None:
    rng = np.random.default_rng(0)
    d = Geometric(p=0.25)
    for _ in range(20):
        lo = int(rng.integers(1, 30))
        hi = lo + int(rng.integers(0, 30))
        assert math.isclose(
            d.mass(lo, hi), d.cdf(hi) - d.cdf(lo - 1), abs_tol=1e-12
        )


def test_geometric_mass_empty_returns_zero() -> None:
    d = Geometric(p=0.5)
    assert d.mass(5, 3) == 0.0


def test_geometric_empirical_mean() -> None:
    rng = np.random.default_rng(0)
    p = 0.2
    d = Geometric(p=p)
    samples = d.sample(rng, 100_000)
    assert abs(samples.mean() - 1.0 / p) < 0.3


def test_geometric_truncated_in_bounds() -> None:
    rng = np.random.default_rng(0)
    d = Geometric(p=0.3)
    samples = d.sample_truncated(rng, 3, 8, 5000)
    assert samples.min() >= 3
    assert samples.max() <= 8


def test_geometric_truncated_matches_pmf() -> None:
    rng = np.random.default_rng(42)
    d = Geometric(p=0.3)
    lo, hi = 3, 8
    samples = d.sample_truncated(rng, lo, hi, 200_000)
    Z = d.mass(lo, hi)
    for k in range(lo, hi + 1):
        empirical = float((samples == k).mean())
        theoretical = d.pmf(k) / Z
        assert abs(empirical - theoretical) < 0.01


def test_geometric_support_bounds_covers_mass() -> None:
    d = Geometric(p=0.1)
    a, b = d.support_bounds(1e-6)
    assert d.mass(a, b) >= 1.0 - 1e-6


def test_geometric_invalid_p_raises() -> None:
    with pytest.raises(ValueError):
        Geometric(p=0.0)
    with pytest.raises(ValueError):
        Geometric(p=1.0)


# ----------------------------------------------------------------------------
# BoundedGeometric
# ----------------------------------------------------------------------------


def test_bounded_geometric_pmf_normalizes() -> None:
    d = BoundedGeometric(p=0.1, N=50)
    s = sum(d.pmf(k) for k in range(1, 51))
    assert math.isclose(s, 1.0, abs_tol=1e-12)


def test_bounded_geometric_cdf_endpoints() -> None:
    d = BoundedGeometric(p=0.1, N=50)
    assert d.cdf(0) == 0.0
    assert math.isclose(d.cdf(50), 1.0, abs_tol=1e-12)
    assert math.isclose(d.cdf(60), 1.0, abs_tol=1e-12)


def test_bounded_geometric_mass_matches_cdf() -> None:
    d = BoundedGeometric(p=0.1, N=50)
    for lo, hi in [(1, 10), (5, 25), (40, 50), (1, 50), (25, 25)]:
        assert math.isclose(
            d.mass(lo, hi), d.cdf(hi) - d.cdf(lo - 1), abs_tol=1e-12
        )


def test_bounded_geometric_sample_within_support() -> None:
    rng = np.random.default_rng(0)
    d = BoundedGeometric(p=0.1, N=30)
    samples = d.sample(rng, 5000)
    assert samples.min() >= 1
    assert samples.max() <= 30


def test_bounded_geometric_truncated_matches_pmf() -> None:
    rng = np.random.default_rng(0)
    d = BoundedGeometric(p=0.1, N=30)
    lo, hi = 5, 15
    samples = d.sample_truncated(rng, lo, hi, 200_000)
    Z = d.mass(lo, hi)
    for k in range(lo, hi + 1):
        empirical = float((samples == k).mean())
        theoretical = d.pmf(k) / Z
        assert abs(empirical - theoretical) < 0.01


def test_bounded_geometric_support_bounds() -> None:
    d = BoundedGeometric(p=0.1, N=50)
    assert d.support_bounds(1e-10) == (1, 50)


# ----------------------------------------------------------------------------
# Uniform
# ----------------------------------------------------------------------------


def test_uniform_pmf_normalizes() -> None:
    d = Uniform(lo=1, hi=10)
    assert math.isclose(sum(d.pmf(k) for k in range(1, 11)), 1.0)
    assert d.pmf(0) == 0.0
    assert d.pmf(11) == 0.0


def test_uniform_mass() -> None:
    d = Uniform(lo=1, hi=10)
    assert math.isclose(d.mass(3, 7), 5.0 / 10.0)
    assert d.mass(11, 20) == 0.0


def test_uniform_truncated() -> None:
    rng = np.random.default_rng(0)
    d = Uniform(lo=1, hi=10)
    samples = d.sample_truncated(rng, 3, 7, 1000)
    assert samples.min() >= 3
    assert samples.max() <= 7
    counts = np.bincount(samples, minlength=11)[3:8]
    assert (counts > 0).all()


def test_uniform_support_bounds() -> None:
    d = Uniform(lo=-5, hi=5)
    assert d.support_bounds() == (-5, 5)


def test_uniform_invalid_range_raises() -> None:
    with pytest.raises(ValueError):
        Uniform(lo=10, hi=1)


# ----------------------------------------------------------------------------
# Categorical
# ----------------------------------------------------------------------------


def test_categorical_pmf() -> None:
    d = Categorical(probs=(0.1, 0.2, 0.3, 0.4))
    assert d.pmf(0) == 0.1
    assert d.pmf(3) == 0.4
    assert d.pmf(-1) == 0.0
    assert d.pmf(4) == 0.0


def test_categorical_sample_empirical() -> None:
    rng = np.random.default_rng(0)
    d = Categorical(probs=(0.1, 0.2, 0.3, 0.4))
    samples = d.sample(rng, 100_000)
    for k in range(4):
        assert abs(float((samples == k).mean()) - d.pmf(k)) < 0.01


def test_categorical_truncated() -> None:
    rng = np.random.default_rng(0)
    d = Categorical(probs=(0.1, 0.2, 0.3, 0.4))
    samples = d.sample_truncated(rng, 1, 2, 10_000)
    assert samples.min() >= 1 and samples.max() <= 2
    # Conditional on {1, 2}: probs 0.2/0.5=0.4 and 0.3/0.5=0.6
    assert abs(float((samples == 1).mean()) - 0.4) < 0.02


def test_categorical_invalid_probs_raise() -> None:
    with pytest.raises(ValueError):
        Categorical(probs=(0.5, 0.4))  # doesn't sum to 1
    with pytest.raises(ValueError):
        Categorical(probs=(0.5, 0.5, -0.0001, 0.0001))
    with pytest.raises(ValueError):
        Categorical(probs=())


# ----------------------------------------------------------------------------
# Poisson
# ----------------------------------------------------------------------------


def test_poisson_pmf_normalizes() -> None:
    d = Poisson(lam=3.0)
    s = sum(d.pmf(k) for k in range(0, 100))
    assert math.isclose(s, 1.0, abs_tol=1e-9)


def test_poisson_empirical_mean() -> None:
    rng = np.random.default_rng(0)
    d = Poisson(lam=5.0)
    samples = d.sample(rng, 50_000)
    assert abs(samples.mean() - 5.0) < 0.2


def test_poisson_truncated() -> None:
    rng = np.random.default_rng(0)
    d = Poisson(lam=3.0)
    samples = d.sample_truncated(rng, 1, 5, 1000)
    assert samples.min() >= 1
    assert samples.max() <= 5


def test_poisson_support_bounds() -> None:
    d = Poisson(lam=10.0)
    a, b = d.support_bounds(1e-6)
    assert d.mass(a, b) >= 1.0 - 1e-6


# ----------------------------------------------------------------------------
# ProductDistribution
# ----------------------------------------------------------------------------


def test_product_pmf_factorizes() -> None:
    d = ProductDistribution(
        factors={"a": Geometric(p=0.2), "b": Uniform(lo=1, hi=5)}
    )
    x = {"a": 3, "b": 2}
    expected = Geometric(p=0.2).pmf(3) * Uniform(lo=1, hi=5).pmf(2)
    assert math.isclose(d.pmf(x), expected)


def test_product_sample_one_in_support() -> None:
    rng = np.random.default_rng(0)
    d = ProductDistribution(
        factors={"a": Geometric(p=0.2), "b": Uniform(lo=1, hi=5)}
    )
    for _ in range(50):
        x = d.sample_one(rng)
        assert set(x.keys()) == {"a", "b"}
        assert 1 <= x["b"] <= 5
        assert x["a"] >= 1


def test_product_sample_batch_shapes() -> None:
    rng = np.random.default_rng(0)
    d = ProductDistribution(
        factors={"a": Geometric(p=0.2), "b": Uniform(lo=1, hi=5)}
    )
    batch = d.sample(rng, 100)
    assert batch["a"].shape == (100,)
    assert batch["b"].shape == (100,)


def test_product_support_bounds() -> None:
    d = ProductDistribution(
        factors={
            "a": Uniform(lo=1, hi=10),
            "b": BoundedGeometric(p=0.1, N=30),
        }
    )
    bounds = d.support_bounds()
    assert bounds["a"] == (1, 10)
    assert bounds["b"] == (1, 30)


def test_product_variables_preserves_order() -> None:
    d = ProductDistribution(
        factors={"x": Uniform(lo=1, hi=3), "y": Uniform(lo=1, hi=3), "z": Uniform(lo=1, hi=3)}
    )
    assert d.variables == ("x", "y", "z")


def test_product_empty_factors_raises() -> None:
    with pytest.raises(ValueError):
        ProductDistribution(factors={})
