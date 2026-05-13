"""Tests for ``dise.regions.Frontier``."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dise.distributions import ProductDistribution, Uniform
from dise.regions import Frontier, Status
from dise.smt import MockBackend, has_z3
from dise.smt import Z3Backend as _Z3Backend


def _backend_params():
    out = [pytest.param("mock", id="mock")]
    if has_z3():
        out.append(pytest.param("z3", id="z3"))
    return out


@pytest.fixture(params=_backend_params())
def backend(request):
    if request.param == "mock":
        return MockBackend()
    return _Z3Backend()


@pytest.fixture
def small_uniform_dist():
    return ProductDistribution(factors={"x": Uniform(1, 10), "y": Uniform(1, 10)})


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


def test_root_is_unconstrained(backend, small_uniform_dist) -> None:
    f = Frontier(small_uniform_dist, backend)
    assert f.root.w_hat == 1.0
    assert f.root.status == Status.OPEN
    assert f.n_leaves() == 1
    assert f.root.is_leaf


def test_total_leaf_mass_equals_one(backend, small_uniform_dist) -> None:
    f = Frontier(small_uniform_dist, backend)
    assert math.isclose(f.total_leaf_mass(), 1.0)


def test_open_mass_equals_one_initially(backend, small_uniform_dist) -> None:
    f = Frontier(small_uniform_dist, backend)
    assert math.isclose(f.open_mass(), 1.0)


# ---------------------------------------------------------------------------
# Wilson smoothing: mu_var does not collapse on all-hits
# ---------------------------------------------------------------------------


def test_mu_var_nonzero_on_all_hits(backend, small_uniform_dist) -> None:
    f = Frontier(small_uniform_dist, backend)
    leaf = f.root
    # Simulate 100 samples all with phi=1
    for _ in range(100):
        leaf.n_samples += 1
        leaf.n_hits += 1
    assert leaf.mu_hat == 1.0
    # Wilson per-sample variance: (101 * 1) / (102**2) ≈ 0.0097
    assert leaf.mu_var > 0.0
    assert math.isclose(leaf.mu_var, 101 / (102**2), rel_tol=1e-12)
    assert leaf.mu_var > 0.009 and leaf.mu_var < 0.011


def test_mu_var_zero_for_closed_leaves(backend, small_uniform_dist) -> None:
    f = Frontier(small_uniform_dist, backend)
    leaf = f.root
    leaf.status = Status.CLOSED_TRUE
    assert leaf.mu_var == 0.0
    assert leaf.mu_mean_var == 0.0
    leaf.status = Status.CLOSED_FALSE
    assert leaf.mu_var == 0.0


# ---------------------------------------------------------------------------
# Refinement
# ---------------------------------------------------------------------------


def test_refine_partitions_mass(backend, small_uniform_dist, rng) -> None:
    f = Frontier(small_uniform_dist, backend)
    x = backend.make_int_var("x")
    clause = backend.op("<=", x, backend.const(5))
    children = f.refine(f.root, clause, rng)
    assert len(children) == 2
    # x in [1, 5] ∪ [6, 10] under Uniform(1,10): masses 0.5 and 0.5
    masses = sorted(c.w_hat for c in children if c.status != Status.EMPTY)
    assert all(math.isclose(m, 0.5) for m in masses)
    assert math.isclose(sum(masses), 1.0)


def test_refine_total_leaf_mass_preserved(backend, small_uniform_dist, rng) -> None:
    f = Frontier(small_uniform_dist, backend)
    x = backend.make_int_var("x")
    clause = backend.op("<=", x, backend.const(5))
    f.refine(f.root, clause, rng)
    assert math.isclose(f.total_leaf_mass(), 1.0)


def test_refine_drops_empty_branch(backend, small_uniform_dist, rng) -> None:
    f = Frontier(small_uniform_dist, backend)
    # First refine: x <= 5
    x = backend.make_int_var("x")
    f.refine(f.root, backend.op("<=", x, backend.const(5)), rng)
    left_child = f.root.children[0]
    # Then refine left_child by x > 5 (which is empty)
    children = f.refine(left_child, backend.op(">", x, backend.const(5)), rng)
    statuses = {c.status for c in children}
    # one EMPTY, one OPEN
    assert Status.EMPTY in statuses
    assert Status.OPEN in statuses


def test_refine_then_refine_again(backend, small_uniform_dist, rng) -> None:
    f = Frontier(small_uniform_dist, backend)
    x = backend.make_int_var("x")
    y = backend.make_int_var("y")
    f.refine(f.root, backend.op("<=", x, backend.const(5)), rng)
    leaf = f.root.children[0]  # x <= 5
    f.refine(leaf, backend.op("<=", y, backend.const(5)), rng)
    # Now four leaves; their masses sum to 1
    leaves = f.leaves()
    nonempty = [n for n in leaves if n.status != Status.EMPTY]
    assert math.isclose(sum(n.w_hat for n in nonempty), 1.0)


def test_refine_clears_parent_observations(backend, small_uniform_dist, rng) -> None:
    f = Frontier(small_uniform_dist, backend)
    f.add_observation(f.root, (1, 2), 1)
    f.add_observation(f.root, (1, 2), 1)
    assert f.root.n_samples == 2
    x = backend.make_int_var("x")
    f.refine(f.root, backend.op("<=", x, backend.const(5)), rng)
    # parent's observations were dropped
    assert f.root.n_samples == 0
    assert f.root.observed_sequences == []


def test_refine_internal_node_raises(backend, small_uniform_dist, rng) -> None:
    f = Frontier(small_uniform_dist, backend)
    x = backend.make_int_var("x")
    f.refine(f.root, backend.op("<=", x, backend.const(5)), rng)
    with pytest.raises(ValueError):
        f.refine(f.root, backend.op("<=", x, backend.const(3)), rng)


# ---------------------------------------------------------------------------
# find_leaf_for
# ---------------------------------------------------------------------------


def test_find_leaf_for_consistent_with_refinement(backend, small_uniform_dist, rng) -> None:
    f = Frontier(small_uniform_dist, backend)
    x = backend.make_int_var("x")
    f.refine(f.root, backend.op("<=", x, backend.const(5)), rng)
    # Sample x=3 goes to left child; x=8 to right child
    leaf_left = f.find_leaf_for({"x": 3, "y": 5})
    leaf_right = f.find_leaf_for({"x": 8, "y": 5})
    assert leaf_left is not leaf_right
    assert leaf_left.region.contains({"x": 3, "y": 5})
    assert leaf_right.region.contains({"x": 8, "y": 5})


def test_find_leaf_for_root_when_no_refinement(backend, small_uniform_dist) -> None:
    f = Frontier(small_uniform_dist, backend)
    assert f.find_leaf_for({"x": 3, "y": 5}) is f.root


# ---------------------------------------------------------------------------
# Closure rule
# ---------------------------------------------------------------------------


def test_try_close_succeeds_on_agreement(backend, small_uniform_dist) -> None:
    """All-agree on a tiny sample succeeds when the concentration check
    is disabled (``closure_epsilon=1.0``). With default ``closure_epsilon``
    of 0.02 the same input would correctly *not* close — see
    :func:`test_try_close_concentration_check`."""
    f = Frontier(small_uniform_dist, backend)
    seq = ("c1", "c2")
    for _ in range(5):
        f.add_observation(f.root, seq, 1)
    closed = f.try_close(f.root, min_samples=5, closure_epsilon=1.0)
    assert closed is True
    assert f.root.status == Status.CLOSED_TRUE


def test_try_close_fails_on_disagreement(backend, small_uniform_dist) -> None:
    f = Frontier(small_uniform_dist, backend)
    f.add_observation(f.root, ("c1", "c2"), 1)
    f.add_observation(f.root, ("c1", "c2"), 0)  # phi differs
    for _ in range(3):
        f.add_observation(f.root, ("c1", "c2"), 1)
    closed = f.try_close(f.root, min_samples=5, closure_epsilon=1.0)
    assert closed is False
    assert f.root.status == Status.OPEN


def test_try_close_fails_on_different_sequences(backend, small_uniform_dist) -> None:
    """All phi agree, but branch sequences differ — closure must NOT fire."""
    f = Frontier(small_uniform_dist, backend)
    for _ in range(3):
        f.add_observation(f.root, ("a", "b"), 1)
    for _ in range(3):
        f.add_observation(f.root, ("a", "c"), 1)  # different sequence
    closed = f.try_close(f.root, min_samples=5, closure_epsilon=1.0)
    assert closed is False


def test_try_close_below_min_samples(backend, small_uniform_dist) -> None:
    f = Frontier(small_uniform_dist, backend)
    for _ in range(3):
        f.add_observation(f.root, ("c1",), 1)
    closed = f.try_close(f.root, min_samples=5, closure_epsilon=1.0)
    assert closed is False


# ---------------------------------------------------------------------------
# Sound concentration-bounded closure
# ---------------------------------------------------------------------------


def test_try_close_concentration_check(backend, small_uniform_dist) -> None:
    """Sound closure rejects all-agree on too few samples.

    With ``closure_epsilon = 0.02`` and ``delta_close = 0.005``, the
    Wilson-anytime upper bound on the disagreement rate at n=5 is far
    above 0.02 — so closure must NOT fire even though every observation
    agrees.
    """
    f = Frontier(small_uniform_dist, backend)
    for _ in range(5):
        f.add_observation(f.root, ("c",), 1)
    closed = f.try_close(
        f.root, min_samples=5, delta_close=0.005, closure_epsilon=0.02
    )
    assert closed is False
    assert f.root.status == Status.OPEN


def test_try_close_concentration_succeeds_with_enough_samples(
    backend, small_uniform_dist
) -> None:
    """With enough all-agree samples, the Wilson-anytime bound dips below
    ``closure_epsilon`` and sound closure fires. The closure budget
    ``closure_epsilon * w_leaf`` is accumulated in
    ``W_close_accumulated``."""
    f = Frontier(small_uniform_dist, backend)
    # n = 1500 is enough for wilson_halfwidth_anytime(n, 0, 0.005) < 0.02.
    for _ in range(1500):
        f.add_observation(f.root, ("c",), 1)
    closed = f.try_close(
        f.root, min_samples=5, delta_close=0.005, closure_epsilon=0.02
    )
    assert closed is True
    assert f.root.status == Status.CLOSED_TRUE
    # Root has w_hat = 1.0, so W_close_accumulated = 1.0 * 0.02 = 0.02.
    assert abs(f.W_close_accumulated - 0.02) < 1e-9


def test_try_close_loose_epsilon_recovers_old_behavior(
    backend, small_uniform_dist
) -> None:
    """``closure_epsilon = 1.0`` recovers the original (unsound) all-agree
    rule and contributes no closure-uncertainty mass."""
    f = Frontier(small_uniform_dist, backend)
    for _ in range(5):
        f.add_observation(f.root, ("c",), 1)
    closed = f.try_close(f.root, min_samples=5, closure_epsilon=1.0)
    assert closed is True
    # With epsilon=1.0, the entire leaf mass goes into W_close — but in
    # the certified-interval computation it is clipped to [0, 1] anyway.
    # Verify the accumulator records the contribution.
    assert abs(f.W_close_accumulated - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# compute_mu_hat
# ---------------------------------------------------------------------------


def test_compute_mu_hat_with_closed_leaves(backend, small_uniform_dist, rng) -> None:
    f = Frontier(small_uniform_dist, backend)
    x = backend.make_int_var("x")
    f.refine(f.root, backend.op("<=", x, backend.const(5)), rng)
    left, right = f.root.children
    left.status = Status.CLOSED_TRUE
    right.status = Status.CLOSED_FALSE
    mu, var = f.compute_mu_hat()
    assert math.isclose(mu, 0.5)
    assert var == 0.0


def test_compute_mu_hat_open_leaves_contribute_variance(backend, small_uniform_dist, rng) -> None:
    f = Frontier(small_uniform_dist, backend)
    # All open, n_samples=0 — mu_hat starts at 0.5 per leaf, variance > 0
    mu, var = f.compute_mu_hat()
    assert mu == 0.5  # 1.0 * 0.5
    assert var > 0
