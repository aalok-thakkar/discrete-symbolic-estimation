"""Tests for ``dise.sampler``."""

from __future__ import annotations

import numpy as np
import pytest

from dise.distributions import ProductDistribution, Uniform
from dise.regions import AxisAlignedBox, GeneralRegion, build_region
from dise.sampler import IntegerLatticeMHSampler, RejectionSampler
from dise.smt import MockBackend, has_z3
from dise.smt import Z3Backend as _Z3Backend


def _backend_params():
    out = [pytest.param("mock", id="mock")]
    if has_z3():
        out.append(pytest.param("z3", id="z3"))
    return out


@pytest.fixture(params=_backend_params())
def backend(request):
    return MockBackend() if request.param == "mock" else _Z3Backend()


# ---------------------------------------------------------------------------
# AxisAlignedBox via RejectionSampler
# ---------------------------------------------------------------------------


def test_axis_aligned_sample_via_sampler(backend) -> None:
    d = ProductDistribution(factors={"x": Uniform(1, 10), "y": Uniform(1, 10)})
    box = AxisAlignedBox({"x": (3, 7), "y": (2, 5)}, backend.true())
    rng = np.random.default_rng(0)
    s = RejectionSampler()
    batch = s.sample(box, d, backend, rng, 500)
    assert batch.n == 500
    for x in batch.iter_assignments():
        assert 3 <= x["x"] <= 7
        assert 2 <= x["y"] <= 5


# ---------------------------------------------------------------------------
# GeneralRegion via RejectionSampler — predicate-aware
# ---------------------------------------------------------------------------


def test_general_region_marginal_mean_matches_truth(backend) -> None:
    """On {(a, b) in [1, 10]^2 : a + b <= 8}, the conditional marginal mean of
    `a` under Uniform(1, 10) x Uniform(1, 10) is finite and recoverable."""
    d = ProductDistribution(factors={"a": Uniform(1, 10), "b": Uniform(1, 10)})
    a = backend.make_int_var("a")
    b = backend.make_int_var("b")
    formula = backend.op("<=", backend.op("+", a, b), backend.const(8))
    region = build_region(formula, d, backend)
    assert isinstance(region, GeneralRegion)
    rng = np.random.default_rng(0)
    s = RejectionSampler()
    batch = s.sample(region, d, backend, rng, 2000)
    # Truth (computed by enumeration):
    pairs = [(a, b) for a in range(1, 11) for b in range(1, 11) if a + b <= 8]
    truth_mean_a = float(np.mean([p[0] for p in pairs]))
    assert abs(float(batch.inputs["a"].mean()) - truth_mean_a) < 0.2


def test_low_acceptance_returns_partial_batch(backend) -> None:
    """Rare-event region: tight budget leads to partial batch."""
    d = ProductDistribution(factors={"a": Uniform(1, 100), "b": Uniform(1, 100)})
    a = backend.make_int_var("a")
    b = backend.make_int_var("b")
    # {(a, b) in [1, 100]^2 : a + b <= 25} — 300 of 10000 pairs = 3% acceptance.
    formula = backend.op("<=", backend.op("+", a, b), backend.const(25))
    region = build_region(formula, d, backend)
    assert isinstance(region, GeneralRegion)
    rng = np.random.default_rng(0)
    # Tight budget: 1000 attempts for 100 requested samples; only ~30 expected.
    s = RejectionSampler(max_attempts_per_sample=10)
    batch = s.sample(region, d, backend, rng, 100)
    assert batch.n < 100
    assert batch.rejection_ratio is not None
    assert 0.0 < batch.rejection_ratio < 0.1
    # Every accepted sample satisfies the predicate
    for x in batch.iter_assignments():
        assert x["a"] + x["b"] <= 25


def test_zero_n_returns_empty(backend) -> None:
    d = ProductDistribution(factors={"x": Uniform(1, 10)})
    box = AxisAlignedBox({"x": (1, 10)}, backend.true())
    s = RejectionSampler()
    rng = np.random.default_rng(0)
    batch = s.sample(box, d, backend, rng, 0)
    assert batch.n == 0


# ---------------------------------------------------------------------------
# MCMC stub
# ---------------------------------------------------------------------------


def test_mcmc_sampler_delegates_for_axis_aligned(backend) -> None:
    """For non-GeneralRegion inputs, the MCMC sampler delegates to the
    region's native (closed-form) sampling."""
    d = ProductDistribution(factors={"x": Uniform(1, 10)})
    box = AxisAlignedBox({"x": (1, 10)}, backend.true())
    s = IntegerLatticeMHSampler()
    batch = s.sample(box, d, backend, np.random.default_rng(0), 50)
    assert batch.n == 50
    for x in batch.iter_assignments():
        assert 1 <= x["x"] <= 10


def test_mcmc_sampler_on_general_region(backend) -> None:
    """MCMC samples a non-trivial general region; outputs satisfy the predicate."""
    d = ProductDistribution(
        factors={"a": Uniform(1, 50), "b": Uniform(1, 50)}
    )
    a = backend.make_int_var("a")
    b = backend.make_int_var("b")
    formula = backend.op("<=", backend.op("+", a, b), backend.const(30))
    region = build_region(formula, d, backend)
    s = IntegerLatticeMHSampler(n_burn_in=200, thin=3)
    batch = s.sample(region, d, backend, np.random.default_rng(0), 200)
    assert batch.n == 200
    # Every sample satisfies the predicate.
    for x in batch.iter_assignments():
        assert x["a"] + x["b"] <= 30


def test_mcmc_sampler_marginal_mean_matches_truth(backend) -> None:
    """MCMC's stationary distribution recovers the correct conditional mean."""
    d = ProductDistribution(
        factors={"a": Uniform(1, 20), "b": Uniform(1, 20)}
    )
    a = backend.make_int_var("a")
    b = backend.make_int_var("b")
    formula = backend.op("<=", backend.op("+", a, b), backend.const(15))
    region = build_region(formula, d, backend)
    s = IntegerLatticeMHSampler(n_burn_in=2000, thin=20, sigma_scale=0.5)
    batch = s.sample(region, d, backend, np.random.default_rng(0), 2000)
    pairs = [(av, bv) for av in range(1, 21) for bv in range(1, 21) if av + bv <= 15]
    truth_mean_a = float(np.mean([p[0] for p in pairs]))
    # MCMC has correlation; bias up to 1.5 is acceptable given the small chain.
    assert abs(float(batch.inputs["a"].mean()) - truth_mean_a) < 1.5


def test_invalid_max_attempts_raises() -> None:
    with pytest.raises(ValueError):
        RejectionSampler(max_attempts_per_sample=0)


def test_mcmc_invalid_config_raises() -> None:
    with pytest.raises(ValueError):
        IntegerLatticeMHSampler(n_burn_in=-1)
    with pytest.raises(ValueError):
        IntegerLatticeMHSampler(thin=0)
    with pytest.raises(ValueError):
        IntegerLatticeMHSampler(sigma_scale=0.0)
    with pytest.raises(ValueError):
        IntegerLatticeMHSampler(init_attempts=0)
