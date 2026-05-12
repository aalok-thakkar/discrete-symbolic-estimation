"""Tests for ``dise.experiment``."""

from __future__ import annotations

from pathlib import Path

import pytest

from dise.baselines import DiSEBaseline, PlainMonteCarlo
from dise.distributions import Uniform
from dise.experiment import (
    ExperimentReport,
    default_methods,
    load_report,
    run_experiment,
    save_report,
)


def _identity(x: int) -> int:
    return x


def _dist():
    return {"x": Uniform(1, 10)}


def test_run_experiment_basic() -> None:
    report = run_experiment(
        benchmark_name="identity_lt_5",
        description="x < 5 under Uniform(1, 10)",
        program=_identity,
        distribution=_dist(),
        property_fn=lambda y: y < 5,
        methods=[PlainMonteCarlo(), DiSEBaseline(bootstrap=20, batch_size=20)],
        budget=200,
        delta=0.05,
        seeds=range(2),
        mc_samples=1000,
    )
    assert isinstance(report, ExperimentReport)
    # 2 methods × 2 seeds = 4 runs.
    assert len(report.runs) == 4
    # MC truth is reasonable.
    assert report.mc_truth is not None
    assert 0.2 < report.mc_truth < 0.6
    # Aggregates: one per method.
    assert len(report.aggregates) == 2
    methods = {a.method for a in report.aggregates}
    assert methods == {"plain_mc", "dise"}


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    report = run_experiment(
        benchmark_name="identity_lt_5",
        description="x < 5 under Uniform(1, 10)",
        program=_identity,
        distribution=_dist(),
        property_fn=lambda y: y < 5,
        methods=[PlainMonteCarlo()],
        budget=200,
        delta=0.05,
        seeds=range(2),
        mc_samples=500,
    )
    out = tmp_path / "report.json"
    save_report(report, str(out))
    loaded = load_report(str(out))
    assert loaded["benchmark"] == "identity_lt_5"
    assert len(loaded["runs"]) == 2
    # JSON-roundtrip preserves the fields.
    assert loaded["aggregates"][0]["method"] == "plain_mc"


def test_default_methods_returns_three() -> None:
    methods = default_methods(budget=500)
    names = [m.name for m in methods]
    assert names == ["plain_mc", "stratified_random", "dise"]


def test_run_experiment_skip_mc(tmp_path: Path) -> None:
    report = run_experiment(
        benchmark_name="identity",
        description="no-mc",
        program=_identity,
        distribution=_dist(),
        property_fn=lambda y: y < 5,
        methods=[PlainMonteCarlo()],
        budget=100,
        delta=0.05,
        seeds=range(1),
        skip_mc=True,
    )
    assert report.mc_truth is None
    assert report.mc_se is None
    # Coverage is undefined when there is no MC truth.
    assert report.aggregates[0].coverage is None


def test_run_invalid_strata_passthrough() -> None:
    with pytest.raises(ValueError):
        from dise.baselines import StratifiedRandomMC

        StratifiedRandomMC(n_strata=0)
