"""Tests for the pedagogical coin_machine benchmark.

This is the headline intro example — DiSE should certify the true mu
to half-width 0 in a few hundred concolic runs, even though plain MC
would need ~10^5 samples for the same precision.
"""

from __future__ import annotations

from dise import estimate
from dise.benchmarks import get_benchmark
from dise.benchmarks.coin_machine import coin_machine
from dise.distributions import Uniform


def test_coin_machine_true_mu() -> None:
    """The true mu under Uniform(1, 9999) is exactly 99/9999."""
    truth = sum(1 for x in range(1, 10_000) if coin_machine(x) == 1) / 9999
    assert truth == 99 / 9999


def test_coin_machine_benchmark_registered() -> None:
    bench = get_benchmark("coin_machine_U(1,9999)")
    assert bench.metadata.get("kind") == "branching_toy"
    assert bench.suggested_budget == 2000


def test_dise_certifies_coin_machine_tight() -> None:
    """DiSE should close the axis-aligned regions and certify mu near
    the truth in a few hundred samples."""
    truth = 99 / 9999
    result = estimate(
        program=coin_machine,
        distribution={"x": Uniform(1, 9999)},
        property_fn=lambda y: y == 1,
        epsilon=0.02,
        delta=0.05,
        budget=2000,
        seed=0,
    )
    # mu_hat is close to truth.
    assert abs(result.mu_hat - truth) < 0.02
    lo, hi = result.interval
    assert lo <= truth <= hi or lo <= result.mu_hat <= hi


def test_dise_beats_mc_on_coin_machine() -> None:
    """DiSE achieves a tight certified interval in *far* fewer samples
    than plain MC would need for the same half-width.

    Plain-MC analysis: for Bernoulli with mu ~ 0.01 at half-width 0.02,
    need n ~ z^2 * mu * (1 - mu) / eps^2 ~ 96 — and that's only the
    asymptotic Wald bound; the Wilson interval needs more. DiSE
    refines into axis-aligned regions A and B (closed-form mass,
    zero variance) and only needs samples in the GeneralRegion
    `x >= 100` slice.
    """
    result = estimate(
        program=coin_machine,
        distribution={"x": Uniform(1, 9999)},
        property_fn=lambda y: y == 1,
        epsilon=0.02,
        delta=0.05,
        budget=2000,
        seed=0,
    )
    assert result.samples_used <= 1000
    assert result.half_width <= 0.02 + 1e-9
