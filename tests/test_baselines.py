"""Tests for ``dise.baselines``."""

from __future__ import annotations

import pytest

from dise.baselines import (
    DiSEBaseline,
    PlainMonteCarlo,
    StratifiedRandomMC,
)
from dise.distributions import Uniform


def _identity(x: int) -> int:
    return x


def _dist():
    return {"x": Uniform(1, 10)}


def test_plain_mc_runs() -> None:
    b = PlainMonteCarlo()
    result = b.run(
        program=_identity,
        distribution=_dist(),
        property_fn=lambda y: y < 5,
        budget=500,
        delta=0.05,
        seed=0,
    )
    assert result.name == "plain_mc"
    assert 0.0 <= result.mu_hat <= 1.0
    lo, hi = result.interval
    assert 0.0 <= lo <= hi <= 1.0
    assert result.samples_used == 500
    # truth is 0.4 = 4/10; allow loose tolerance
    assert abs(result.mu_hat - 0.4) < 0.1


def test_plain_mc_certified_interval_contains_truth() -> None:
    """The Wilson interval should contain truth with high probability."""
    b = PlainMonteCarlo()
    truth = 0.4
    misses = 0
    for seed in range(20):
        result = b.run(
            program=_identity,
            distribution=_dist(),
            property_fn=lambda y: y < 5,
            budget=300,
            delta=0.05,
            seed=seed,
        )
        lo, hi = result.interval
        if not (lo <= truth <= hi):
            misses += 1
    # At most ~5% misses expected; allow up to 3 out of 20.
    assert misses <= 3


def test_stratified_random_mc_runs() -> None:
    b = StratifiedRandomMC(n_strata=8)
    result = b.run(
        program=_identity,
        distribution=_dist(),
        property_fn=lambda y: y < 5,
        budget=500,
        delta=0.05,
        seed=0,
    )
    assert result.name == "stratified_random"
    assert 0.0 <= result.mu_hat <= 1.0
    assert result.samples_used == 500
    # extras carries bucket-level info
    assert "bucket_counts" in result.extras
    assert sum(result.extras["bucket_counts"]) == 500


def test_dise_baseline_runs() -> None:
    b = DiSEBaseline(bootstrap=50, batch_size=25)
    result = b.run(
        program=_identity,
        distribution=_dist(),
        property_fn=lambda y: y < 5,
        budget=400,
        delta=0.05,
        seed=0,
    )
    assert result.name == "dise"
    assert 0.0 <= result.mu_hat <= 1.0
    assert "terminated_reason" in result.extras


def test_stratified_invalid_strata_raises() -> None:
    with pytest.raises(ValueError):
        StratifiedRandomMC(n_strata=0)
