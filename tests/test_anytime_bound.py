"""Tests for the anytime-valid Wilson bound and ``method="anytime"``."""

from __future__ import annotations

import math

import pytest

from dise.distributions import ProductDistribution, Uniform
from dise.estimator import (
    compute_estimator_state,
    wilson_halfwidth_anytime,
    wilson_halfwidth_for_leaf,
)
from dise.regions import Frontier, Status
from dise.smt import MockBackend

# ---------------------------------------------------------------------------
# wilson_halfwidth_anytime: basic properties
# ---------------------------------------------------------------------------


def test_anytime_is_at_least_fixed_n() -> None:
    """The anytime bound is no tighter than the fixed-n Wilson bound."""
    for n, h, delta in [(10, 5, 0.05), (100, 50, 0.05), (1000, 300, 0.01)]:
        fixed = wilson_halfwidth_for_leaf(n, h, delta)
        anytime = wilson_halfwidth_anytime(n, h, delta)
        assert anytime >= fixed - 1e-12


def test_anytime_nonzero_on_all_hits() -> None:
    """Like the fixed-n Wilson, the anytime version never collapses to 0."""
    half = wilson_halfwidth_anytime(100, 100, 0.05)
    assert half > 0.0


def test_anytime_shrinks_with_n() -> None:
    """Wider bound for smaller n; tighter for larger n (the usual rate)."""
    h_small = wilson_halfwidth_anytime(10, 5, 0.05)
    h_big = wilson_halfwidth_anytime(1000, 500, 0.05)
    assert h_big < h_small


def test_anytime_invalid_delta_raises() -> None:
    with pytest.raises(ValueError):
        wilson_halfwidth_anytime(10, 5, 0.0)
    with pytest.raises(ValueError):
        wilson_halfwidth_anytime(10, 5, 1.0)


def test_anytime_zero_n_returns_one() -> None:
    assert wilson_halfwidth_anytime(0, 0, 0.05) == 1.0


# ---------------------------------------------------------------------------
# compute_estimator_state with method="anytime"
# ---------------------------------------------------------------------------


def test_anytime_method_state() -> None:
    """``method='anytime'`` produces a valid EstimatorState."""
    dist = ProductDistribution(factors={"x": Uniform(1, 10)})
    smt = MockBackend()
    f = Frontier(dist, smt)
    f.root.n_samples = 50
    f.root.n_hits = 30
    state = compute_estimator_state(f, delta=0.05, method="anytime")
    assert state.mu_hat == 0.6
    assert state.eps_stat > 0.0
    lo, hi = state.interval
    assert 0.0 <= lo <= hi <= 1.0


def test_anytime_method_wider_than_wilson() -> None:
    """The anytime interval is wider than the fixed-n Wilson interval."""
    dist = ProductDistribution(factors={"x": Uniform(1, 10)})
    smt = MockBackend()
    f = Frontier(dist, smt)
    f.root.n_samples = 50
    f.root.n_hits = 30
    state_wilson = compute_estimator_state(f, delta=0.05, method="wilson")
    state_any = compute_estimator_state(f, delta=0.05, method="anytime")
    assert state_any.eps_stat >= state_wilson.eps_stat


def test_anytime_method_collapses_on_fully_resolved() -> None:
    """All leaves closed ⇒ eps_stat = 0 (no open leaves to bound)."""
    import numpy as np

    dist = ProductDistribution(factors={"x": Uniform(1, 10)})
    smt = MockBackend()
    f = Frontier(dist, smt)
    x = smt.make_int_var("x")
    f.refine(f.root, smt.op("<=", x, smt.const(5)), np.random.default_rng(0))
    left, right = f.root.children
    left.status = Status.CLOSED_TRUE
    right.status = Status.CLOSED_FALSE
    state = compute_estimator_state(f, delta=0.05, method="anytime")
    assert state.eps_stat == 0.0


def test_unknown_method_raises() -> None:
    dist = ProductDistribution(factors={"x": Uniform(1, 10)})
    smt = MockBackend()
    f = Frontier(dist, smt)
    with pytest.raises(ValueError):
        compute_estimator_state(f, delta=0.05, method="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Bonferroni-in-time identity sanity check
# ---------------------------------------------------------------------------


def test_basel_bonferroni_sums_to_delta() -> None:
    """The per-step delta_n = 6*delta/(pi^2*n^2) sums to delta."""
    delta = 0.05
    s = sum(6.0 * delta / (math.pi * math.pi * n * n) for n in range(1, 10_000))
    assert abs(s - delta) < 1e-3
