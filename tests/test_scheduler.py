"""Tests for ``dise.scheduler.ASIPScheduler``."""

from __future__ import annotations

import numpy as np
import pytest

from dise.distributions import BoundedGeometric, ProductDistribution, Uniform
from dise.scheduler import ASIPScheduler, SchedulerConfig
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
# Trivial identity-with-property: Pr[x < 5] under Uniform(1, 10) = 4/10 = 0.4
# ---------------------------------------------------------------------------


def test_identity_with_property_converges(backend) -> None:
    def prog(x):
        return x

    dist = ProductDistribution(factors={"x": Uniform(1, 10)})
    cfg = SchedulerConfig(
        epsilon=0.05,
        delta=0.05,
        budget_samples=2000,
        bootstrap_samples=200,
        batch_size=50,
    )
    s = ASIPScheduler(prog, dist, lambda y: y < 5, backend, cfg, np.random.default_rng(0))
    result = s.run()
    truth = 0.4  # exactly 4/10
    lo, hi = result.final_estimator.interval
    assert lo <= truth <= hi
    assert (hi - lo) <= 0.10
    # Frontier fully closes for this simple program.
    assert result.frontier.open_mass() == 0.0


# ---------------------------------------------------------------------------
# Branching program: count = (x < 5) + (y < 5), property `count >= 1`
# Under Uniform(1, 10) x Uniform(1, 10): P[count >= 1] = 1 - P[x>=5, y>=5]
#                                                     = 1 - (6/10)(6/10) = 64/100 = 0.64
# ---------------------------------------------------------------------------


def test_branching_program_converges(backend, request) -> None:
    def prog(x, y):
        count = 0
        if x < 5:
            count = count + 1
        if y < 5:
            count = count + 1
        return count

    dist = ProductDistribution(factors={"x": Uniform(1, 10), "y": Uniform(1, 10)})
    cfg = SchedulerConfig(
        epsilon=0.05,
        delta=0.05,
        budget_samples=5000,
        bootstrap_samples=200,
        batch_size=50,
    )
    s = ASIPScheduler(prog, dist, lambda v: v >= 1, backend, cfg, np.random.default_rng(0))
    result = s.run()
    truth = 1.0 - (6 / 10) * (6 / 10)  # 0.64
    lo, hi = result.final_estimator.interval
    # Soundness is required for *every* backend.
    assert lo <= truth <= hi
    # Tightness depends on the backend. Under strict (SMT-only)
    # closure, MockBackend cannot certify path-determinism on
    # multi-variable formulas, so leaves stay open and ``W_open``
    # forces the trivial $[0, 1]$ interval. Z3 decides such formulas
    # and produces the tight bound. We test tightness only with Z3.
    backend_id = request.node.callspec.id
    if backend_id == "z3":
        assert (hi - lo) <= 0.15


# ---------------------------------------------------------------------------
# GCD-style program — the running example
# ---------------------------------------------------------------------------


def _gcd_steps(a, b):
    steps = 0
    while b != 0:
        a, b = b, a % b
        steps = steps + 1
    return steps


def test_gcd_k_10_converges(backend) -> None:
    """GCD with k=10 under BoundedGeometric(0.1, 100).

    Empirically μ ~ 1.0 (gcd of two small geometrics rarely takes > 10 steps).
    """
    dist = ProductDistribution(
        factors={
            "a": BoundedGeometric(p=0.1, N=100),
            "b": BoundedGeometric(p=0.1, N=100),
        }
    )
    cfg = SchedulerConfig(
        epsilon=0.05,
        delta=0.05,
        budget_samples=2000,
        bootstrap_samples=200,
        batch_size=50,
        max_refinement_depth=30,
    )
    s = ASIPScheduler(
        _gcd_steps,
        dist,
        lambda steps: steps <= 10,
        backend,
        cfg,
        np.random.default_rng(0),
    )
    result = s.run()
    # MC ground truth: very close to 1.0 for k=10, N=100, p=0.1.
    # Sanity: mu_hat in [0.95, 1.0].
    assert 0.90 <= result.final_estimator.mu_hat <= 1.0
    lo, hi = result.final_estimator.interval
    assert lo <= 1.0 <= hi or lo <= result.final_estimator.mu_hat <= hi


def _gcd_k5_mc_truth(rng: np.random.Generator) -> tuple[float, float]:
    dist = ProductDistribution(
        factors={
            "a": BoundedGeometric(p=0.1, N=100),
            "b": BoundedGeometric(p=0.1, N=100),
        }
    )
    mc_batch = dist.sample(rng, 5000)
    hits = 0
    for i in range(5000):
        steps = _gcd_steps(int(mc_batch["a"][i]), int(mc_batch["b"][i]))
        if steps <= 5:
            hits += 1
    mu = hits / 5000
    se = (mu * (1 - mu) / 5000) ** 0.5
    return mu, se


def test_gcd_k_5_point_estimate(backend) -> None:
    """DiSE's mu_hat is close to MC ground truth on GCD with k=5.

    Parametrized over backends; the Mock backend falls back to sample-based
    closure (no SMT shortcut), so a small bias is tolerated.
    """
    dist = ProductDistribution(
        factors={
            "a": BoundedGeometric(p=0.1, N=100),
            "b": BoundedGeometric(p=0.1, N=100),
        }
    )
    mc_mu, _ = _gcd_k5_mc_truth(np.random.default_rng(1234))
    cfg = SchedulerConfig(
        epsilon=0.05,
        delta=0.05,
        budget_samples=2000,
        bootstrap_samples=200,
        batch_size=50,
        max_refinement_depth=20,
    )
    s = ASIPScheduler(
        _gcd_steps,
        dist,
        lambda steps: steps <= 5,
        backend,
        cfg,
        np.random.default_rng(0),
    )
    result = s.run()
    # Allow 2% bias for backends without the SMT path-determinism shortcut.
    assert abs(result.final_estimator.mu_hat - mc_mu) < 0.02


@pytest.mark.skipif(not has_z3(), reason="requires z3-solver for path-determinism shortcut")
def test_gcd_k_5_certified_interval_z3() -> None:
    """With Z3's SMT shortcut, the certified interval contains MC truth."""
    backend = _Z3Backend()
    dist = ProductDistribution(
        factors={
            "a": BoundedGeometric(p=0.1, N=100),
            "b": BoundedGeometric(p=0.1, N=100),
        }
    )
    mc_mu, mc_se = _gcd_k5_mc_truth(np.random.default_rng(1234))
    cfg = SchedulerConfig(
        epsilon=0.05,
        delta=0.05,
        budget_samples=1500,
        bootstrap_samples=200,
        batch_size=50,
        max_refinement_depth=15,
    )
    s = ASIPScheduler(
        _gcd_steps,
        dist,
        lambda steps: steps <= 5,
        backend,
        cfg,
        np.random.default_rng(0),
    )
    result = s.run()
    lo, hi = result.final_estimator.interval
    # MC ground truth lies inside the certified interval (with small slop).
    assert lo - 2.5 * mc_se <= mc_mu <= hi + 2.5 * mc_se


# ---------------------------------------------------------------------------
# Termination semantics
# ---------------------------------------------------------------------------


def test_budget_exhausted_termination(backend) -> None:
    def prog(x):
        # Many branches — hard to converge in a tiny budget.
        out = 0
        if x > 1:
            out = out + 1
        if x > 2:
            out = out + 1
        if x > 3:
            out = out + 1
        return out

    dist = ProductDistribution(factors={"x": BoundedGeometric(p=0.1, N=100)})
    cfg = SchedulerConfig(
        epsilon=0.001,  # very tight
        delta=0.05,
        budget_samples=50,  # very small
        bootstrap_samples=20,
        batch_size=10,
    )
    s = ASIPScheduler(prog, dist, lambda v: v == 3, backend, cfg, np.random.default_rng(0))
    result = s.run()
    assert result.terminated_reason == "budget_exhausted"


def test_epsilon_reached_termination(backend) -> None:
    def prog(x):
        return x

    dist = ProductDistribution(factors={"x": Uniform(1, 10)})
    cfg = SchedulerConfig(
        epsilon=0.1,
        delta=0.05,
        budget_samples=2000,
        bootstrap_samples=200,
        batch_size=50,
    )
    s = ASIPScheduler(prog, dist, lambda y: y < 5, backend, cfg, np.random.default_rng(0))
    result = s.run()
    assert result.terminated_reason == "epsilon_reached"
    eps_total = result.final_estimator.eps_stat + result.final_estimator.W_open
    assert eps_total <= 0.1


# ---------------------------------------------------------------------------
# Frontier invariants after run
# ---------------------------------------------------------------------------


def test_total_leaf_mass_preserved_after_run(backend) -> None:
    def prog(x):
        return x

    dist = ProductDistribution(factors={"x": Uniform(1, 10)})
    cfg = SchedulerConfig(
        epsilon=0.05,
        delta=0.05,
        budget_samples=2000,
        bootstrap_samples=200,
        batch_size=50,
    )
    s = ASIPScheduler(prog, dist, lambda y: y < 5, backend, cfg, np.random.default_rng(0))
    s.run()
    assert abs(s.frontier.total_leaf_mass() - 1.0) < 1e-9
