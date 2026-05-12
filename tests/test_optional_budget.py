"""Tests for optional budget, wall-clock cap, and assertion-violation API."""

from __future__ import annotations

import numpy as np
import pytest

from dise import (
    BoundedGeometric,
    Uniform,
    estimate,
    failure_probability,
)
from dise.distributions import ProductDistribution
from dise.scheduler import ASIPScheduler, SchedulerConfig
from dise.smt import MockBackend

# ---------------------------------------------------------------------------
# Optional sample budget
# ---------------------------------------------------------------------------


def _identity(x: int) -> int:
    return x


def test_estimate_with_budget_none_terminates_at_epsilon() -> None:
    """``budget=None`` ⇒ algorithm halts at ``epsilon_reached``."""
    result = estimate(
        program=_identity,
        distribution={"x": Uniform(1, 10)},
        property_fn=lambda y: y < 5,
        epsilon=0.05,
        delta=0.05,
        budget=None,
        seed=0,
    )
    assert result.terminated_reason == "epsilon_reached"
    # The closed form is exact: 4 / 10.
    assert abs(result.mu_hat - 0.4) < 1e-9
    lo, hi = result.interval
    assert lo <= 0.4 <= hi


def test_scheduler_config_accepts_none_budget() -> None:
    """``SchedulerConfig.budget_samples = None`` is accepted."""
    cfg = SchedulerConfig(budget_samples=None, budget_seconds=None)
    assert cfg.budget_samples is None
    assert cfg.budget_seconds is None


# ---------------------------------------------------------------------------
# Wall-clock cap
# ---------------------------------------------------------------------------


def test_budget_seconds_cap_fires() -> None:
    """A very small ``budget_seconds`` should terminate the run quickly
    (either via the time cap, or — for trivial benchmarks — via
    epsilon_reached before the timer triggers). Whichever fires, the run
    must terminate cleanly."""
    # A program that's slow per concolic run (lots of branches) so the
    # time cap matters before epsilon.
    def slow_prog(x: int) -> int:
        s = 0
        for _ in range(50):
            if x > 0:
                s = s + 1
        return s

    result = estimate(
        program=slow_prog,
        distribution={"x": BoundedGeometric(p=0.1, N=100)},
        property_fn=lambda v: v == 50,
        epsilon=0.001,  # very tight to keep the loop hungry
        delta=0.05,
        budget=None,
        budget_seconds=0.05,  # 50 ms cap
        bootstrap=10,
        batch_size=5,
        seed=0,
    )
    assert result.terminated_reason in {"time_exhausted", "epsilon_reached"}


def test_budget_seconds_none_means_no_time_cap() -> None:
    """When ``budget_seconds`` is None, the run isn't constrained by time."""
    result = estimate(
        program=_identity,
        distribution={"x": Uniform(1, 10)},
        property_fn=lambda y: y < 5,
        epsilon=0.05,
        delta=0.05,
        budget=2000,
        budget_seconds=None,
        seed=0,
    )
    assert result.terminated_reason != "time_exhausted"


# ---------------------------------------------------------------------------
# Diminishing-returns floor
# ---------------------------------------------------------------------------


def test_min_gain_per_cost_zero_keeps_old_behavior() -> None:
    """``min_gain_per_cost = 0`` matches the legacy strict-zero semantics."""
    result = estimate(
        program=_identity,
        distribution={"x": Uniform(1, 10)},
        property_fn=lambda y: y < 5,
        epsilon=0.05,
        delta=0.05,
        budget=2000,
        min_gain_per_cost=0.0,
        seed=0,
    )
    # Trivial program → still converges to epsilon_reached.
    assert result.terminated_reason == "epsilon_reached"


def test_min_gain_per_cost_positive_terminates_early() -> None:
    """A large floor on gain/cost cuts the run short via
    ``no_actions_available``."""
    result = estimate(
        program=_identity,
        distribution={"x": Uniform(1, 10)},
        property_fn=lambda y: y < 5,
        epsilon=1e-9,  # unreachable
        delta=0.05,
        budget=None,
        min_gain_per_cost=1.0,  # extremely strict — nothing qualifies
        bootstrap=20,
        seed=0,
    )
    # Without the diminishing-returns floor + unreachable epsilon, this
    # would run forever. With it, the loop bails out.
    assert result.terminated_reason in {"no_actions_available", "epsilon_reached"}


# ---------------------------------------------------------------------------
# failure_probability convenience wrapper
# ---------------------------------------------------------------------------


def test_failure_probability_assertion_held() -> None:
    """A program whose assertion always holds → mu_hat = 0."""

    def always_ok(x: int) -> int:
        assert x > 0
        return x

    result = failure_probability(
        program=always_ok,
        distribution={"x": Uniform(1, 10)},
        epsilon=0.05,
        delta=0.05,
        budget=2000,
        seed=0,
    )
    assert result.mu_hat == 0.0
    lo, hi = result.interval
    assert lo == 0.0


def test_failure_probability_assertion_always_fails() -> None:
    """A program whose assertion *always* fails → mu_hat = 1."""

    def always_fail(x: int) -> int:
        assert x < 0  # x ~ Uniform(1, 10), never < 0
        return x

    result = failure_probability(
        program=always_fail,
        distribution={"x": Uniform(1, 10)},
        epsilon=0.05,
        delta=0.05,
        budget=2000,
        seed=0,
    )
    assert result.mu_hat == 1.0


def test_failure_probability_partial_failure() -> None:
    """Half of inputs trigger the assertion (a < 5)."""

    def half_fail(a: int) -> int:
        assert a >= 5, "low value"
        return a

    result = failure_probability(
        program=half_fail,
        distribution={"a": Uniform(1, 10)},
        epsilon=0.05,
        delta=0.05,
        budget=2000,
        seed=0,
    )
    # 4 of 10 violate (a in {1, 2, 3, 4}).
    assert abs(result.mu_hat - 0.4) < 1e-9


def test_failure_probability_default_budget_is_none() -> None:
    """``failure_probability`` defaults to ``budget=None`` (unlimited)."""

    def trivial(x: int) -> int:
        return x

    result = failure_probability(
        program=trivial,
        distribution={"x": Uniform(1, 10)},
        epsilon=0.05,
        seed=0,
    )
    # Without a budget the algorithm must terminate at epsilon_reached
    # (or no_actions_available) — but never at budget_exhausted.
    assert result.terminated_reason != "budget_exhausted"


def test_failure_probability_custom_exception() -> None:
    """``catch`` can target any exception class (e.g. ValueError)."""

    def raises_value_error(x: int) -> int:
        if x < 5:
            raise ValueError("low")
        return x

    result = failure_probability(
        program=raises_value_error,
        distribution={"x": Uniform(1, 10)},
        catch=ValueError,
        epsilon=0.05,
        budget=2000,
        seed=0,
    )
    assert abs(result.mu_hat - 0.4) < 1e-9


def test_failure_probability_other_exceptions_propagate() -> None:
    """Exceptions outside ``catch`` propagate (don't get counted)."""

    def raises_runtime(x: int) -> int:
        raise RuntimeError("oops")

    with pytest.raises(RuntimeError):
        failure_probability(
            program=raises_runtime,
            distribution={"x": Uniform(1, 10)},
            catch=AssertionError,
            epsilon=0.05,
            budget=200,
            seed=0,
        )


# ---------------------------------------------------------------------------
# assertion_overflow benchmark
# ---------------------------------------------------------------------------


def test_assertion_overflow_benchmark_registered() -> None:
    from dise.benchmarks import get_benchmark

    bench = get_benchmark("assertion_overflow_mul_w=8_U(1,31)")
    assert "Pr[a*b" in bench.description
    assert bench.metadata.get("kind") == "assertion_violation"


def test_assertion_overflow_runs() -> None:
    from dise.benchmarks import get_benchmark

    bench = get_benchmark("assertion_overflow_mul_w=8_U(1,31)")
    rng = np.random.default_rng(0)
    smt = MockBackend()
    dist = ProductDistribution(factors=dict(bench.distribution))
    # Use the scheduler directly to confirm the benchmark wires up.
    cfg = SchedulerConfig(
        epsilon=0.1,
        delta=0.05,
        budget_samples=1000,
        bootstrap_samples=100,
        batch_size=25,
    )
    s = ASIPScheduler(
        program=bench.program,
        distribution=dist,
        property_fn=bench.property_fn,
        smt=smt,
        config=cfg,
        rng=rng,
    )
    result = s.run()
    assert 0.0 <= result.final_estimator.mu_hat <= 1.0
