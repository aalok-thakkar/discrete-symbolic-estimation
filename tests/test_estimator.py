"""Tests for ``dise.estimator``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dise.distributions import ProductDistribution, Uniform
from dise.estimator import (
    bernstein_halfwidth,
    compute_estimator_state,
    wilson_halfwidth_for_leaf,
)
from dise.regions import Frontier, Status
from dise.smt import MockBackend


@pytest.fixture
def small_dist():
    return ProductDistribution(factors={"x": Uniform(1, 10), "y": Uniform(1, 10)})


@pytest.fixture
def smt():
    return MockBackend()


# ---------------------------------------------------------------------------
# bernstein_halfwidth basic properties
# ---------------------------------------------------------------------------


def test_bernstein_increases_with_variance() -> None:
    h1 = bernstein_halfwidth(0.01, 0.05)
    h2 = bernstein_halfwidth(0.10, 0.05)
    assert h1 < h2


def test_bernstein_increases_as_delta_shrinks() -> None:
    h_loose = bernstein_halfwidth(0.05, 0.10)
    h_tight = bernstein_halfwidth(0.05, 0.01)
    assert h_tight > h_loose


def test_bernstein_zero_variance_is_positive() -> None:
    # The log term still contributes via per_sample_bound:
    h = bernstein_halfwidth(0.0, 0.05, per_sample_bound=1.0)
    assert h > 0.0


def test_bernstein_invalid_delta_raises() -> None:
    with pytest.raises(ValueError):
        bernstein_halfwidth(0.05, 0.0)
    with pytest.raises(ValueError):
        bernstein_halfwidth(0.05, 1.0)


# ---------------------------------------------------------------------------
# wilson_halfwidth_for_leaf
# ---------------------------------------------------------------------------


def test_wilson_halfwidth_n_zero_is_one() -> None:
    assert wilson_halfwidth_for_leaf(0, 0, 0.05) == 1.0


def test_wilson_halfwidth_never_zero_on_all_hits() -> None:
    h = wilson_halfwidth_for_leaf(100, 100, 0.05)
    assert h > 0.0


def test_wilson_halfwidth_shrinks_with_n() -> None:
    h_small = wilson_halfwidth_for_leaf(10, 5, 0.05)
    h_big = wilson_halfwidth_for_leaf(1000, 500, 0.05)
    assert h_big < h_small


# ---------------------------------------------------------------------------
# Full estimator state — closed-only frontier
# ---------------------------------------------------------------------------


def test_closed_frontier_zero_variance(small_dist, smt) -> None:
    f = Frontier(small_dist, smt)
    rng = np.random.default_rng(0)
    x = smt.make_int_var("x")
    f.refine(f.root, smt.op("<=", x, smt.const(5)), rng)
    left, right = f.root.children
    left.status = Status.CLOSED_TRUE
    right.status = Status.CLOSED_FALSE
    state = compute_estimator_state(f, delta=0.05)
    assert math.isclose(state.mu_hat, 0.5)
    assert state.variance == 0.0
    assert state.W_open == 0.0
    # eps_stat still positive (Bernstein log term over per_sample_bound),
    # but the dominant width is from logging, not data.
    # The interval should be [0.5 - eps_stat, 0.5 + eps_stat].


# ---------------------------------------------------------------------------
# Estimator from samples on a single open leaf
# ---------------------------------------------------------------------------


def test_single_open_leaf_recovers_mu_hat(small_dist, smt) -> None:
    f = Frontier(small_dist, smt)
    leaf = f.root
    # Manually inject 100 samples, 50 hits.
    leaf.n_samples = 100
    leaf.n_hits = 50
    state = compute_estimator_state(f, delta=0.05)
    assert math.isclose(state.mu_hat, 0.5, abs_tol=1e-12)
    # variance contribution is w^2 * mu_var / n = 1 * Wilson_var / 100
    assert state.variance > 0
    # eps_stat > 0 (statistical uncertainty)
    assert state.eps_stat > 0
    # W_open == 1.0 (only the root, open)
    assert math.isclose(state.W_open, 1.0)


def test_all_hits_open_leaf_has_nonzero_variance(small_dist, smt) -> None:
    f = Frontier(small_dist, smt)
    leaf = f.root
    leaf.n_samples = 100
    leaf.n_hits = 100
    state = compute_estimator_state(f, delta=0.05)
    # mu_hat = 1 (h/n)
    assert state.mu_hat == 1.0
    # but variance > 0 thanks to Wilson smoothing
    assert state.variance > 0


# ---------------------------------------------------------------------------
# Open-dominated frontier → interval near [0, 1]
# ---------------------------------------------------------------------------


def test_open_dominated_frontier_wide_interval(small_dist, smt) -> None:
    f = Frontier(small_dist, smt)
    state = compute_estimator_state(f, delta=0.05)
    lo, hi = state.interval
    # Initial state: 1 open leaf, no samples, mu_hat=0.5, W_open=1.0
    # Interval = [max(0, 0.5 - eps_stat - 1), min(1, 0.5 + eps_stat + 1)] = [0, 1]
    assert lo == 0.0
    assert hi == 1.0


# ---------------------------------------------------------------------------
# Interval contains [0, 1] clipping
# ---------------------------------------------------------------------------


def test_interval_always_within_unit_interval(small_dist, smt) -> None:
    f = Frontier(small_dist, smt)
    state = compute_estimator_state(f, delta=0.05)
    lo, hi = state.interval
    assert 0.0 <= lo <= hi <= 1.0


# ---------------------------------------------------------------------------
# Wilson method also works
# ---------------------------------------------------------------------------


def test_wilson_method_runs(small_dist, smt) -> None:
    f = Frontier(small_dist, smt)
    f.root.n_samples = 50
    f.root.n_hits = 30
    state_w = compute_estimator_state(f, delta=0.05, method="wilson")
    state_b = compute_estimator_state(f, delta=0.05, method="bernstein")
    assert state_w.mu_hat == state_b.mu_hat == 0.6
    # Both produce some half-width
    assert state_w.eps_stat > 0
    assert state_b.eps_stat > 0


def test_unknown_method_raises(small_dist, smt) -> None:
    f = Frontier(small_dist, smt)
    with pytest.raises(ValueError):
        compute_estimator_state(f, delta=0.05, method="bogus")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# n_total_samples + leaf counts
# ---------------------------------------------------------------------------


def test_leaf_counts(small_dist, smt) -> None:
    f = Frontier(small_dist, smt)
    rng = np.random.default_rng(0)
    x = smt.make_int_var("x")
    f.refine(f.root, smt.op("<=", x, smt.const(5)), rng)
    left, right = f.root.children
    left.status = Status.CLOSED_TRUE
    # right remains OPEN
    f.add_observation(right, ("c1",), 1)
    f.add_observation(right, ("c1",), 1)
    state = compute_estimator_state(f, delta=0.05)
    assert state.n_open_leaves == 1
    assert state.n_closed_leaves == 1
    assert state.n_total_samples == 2
