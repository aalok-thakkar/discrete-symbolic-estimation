"""Tests for the PrPl-EB (Waudby-Smith & Ramdas 2024 Thm 2) betting half-width.

The new ``method="betting"`` option in :func:`compute_estimator_state`
(and end-to-end via :func:`dise.estimate`) uses the closed-form
predictable-plug-in empirical-Bernstein anytime-valid confidence
sequence. These tests check:

* Sanity (monotone in delta, monotone in n in the limit, asymptotic
  behavior).
* That the bound is tighter than the union-bound-in-time Wilson
  construction in the low-variance regime — the headline reason to
  prefer it.
* End-to-end plumbing through ``estimate`` and ``failure_probability``.
"""

from __future__ import annotations

import math
import random

from dise import Uniform, estimate
from dise.estimator import (
    prpl_eb_center,
    prpl_eb_halfwidth_anytime,
    wilson_halfwidth_anytime,
)
from dise.scheduler import SchedulerConfig


# ---------------------------------------------------------------------------
# Unit tests on the half-width function
# ---------------------------------------------------------------------------


def test_empty_history_returns_max_width() -> None:
    """No observations → trivially wide bound."""
    assert prpl_eb_halfwidth_anytime([], delta=0.05) == 1.0


def test_zero_observations_finite_width() -> None:
    """All-zero Bernoulli stream produces a non-trivial bound."""
    w = prpl_eb_halfwidth_anytime([0] * 100, delta=0.05)
    assert 0.0 < w < 1.0


def test_one_observations_finite_width() -> None:
    """All-one Bernoulli stream is symmetric to all-zero."""
    w = prpl_eb_halfwidth_anytime([1] * 100, delta=0.05)
    assert 0.0 < w < 1.0


def test_width_monotone_in_delta() -> None:
    """Tighter confidence (smaller delta) gives wider intervals."""
    phis = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0] * 20
    w_loose = prpl_eb_halfwidth_anytime(phis, delta=0.10)
    w_tight = prpl_eb_halfwidth_anytime(phis, delta=0.01)
    assert w_loose < w_tight


def test_width_decreases_with_more_samples() -> None:
    """The bound shrinks as the observation count grows (low-variance case)."""
    rng = random.Random(0)
    phis = [1 if rng.random() < 0.1 else 0 for _ in range(2000)]
    w_short = prpl_eb_halfwidth_anytime(phis[:200], delta=0.05)
    w_long = prpl_eb_halfwidth_anytime(phis, delta=0.05)
    assert w_long < w_short


def test_tighter_than_wilson_anytime_in_low_variance() -> None:
    """Headline claim: in the low-variance regime, PrPl-EB beats the
    union-bound-in-time Wilson construction. The benefit is the
    motivation for switching from ``"anytime"`` to ``"betting"``."""
    rng = random.Random(0)
    phis = [1 if rng.random() < 0.05 else 0 for _ in range(1000)]
    n, h = len(phis), sum(phis)
    w_betting = prpl_eb_halfwidth_anytime(phis, delta=0.05)
    w_wilson_anytime = wilson_halfwidth_anytime(n, h, delta=0.05)
    assert w_betting < w_wilson_anytime, (
        f"betting={w_betting:.4f} should beat wilson_anytime="
        f"{w_wilson_anytime:.4f} on low-variance Bernoulli"
    )


def test_truncation_parameter_c_valid_range() -> None:
    """``c`` must be in (0, 1); both endpoints rejected."""
    import pytest

    with pytest.raises(ValueError):
        prpl_eb_halfwidth_anytime([0, 1], delta=0.05, c=0.0)
    with pytest.raises(ValueError):
        prpl_eb_halfwidth_anytime([0, 1], delta=0.05, c=1.0)


def test_delta_validation() -> None:
    """``delta`` must be in (0, 1)."""
    import pytest

    with pytest.raises(ValueError):
        prpl_eb_halfwidth_anytime([0, 1], delta=0.0)
    with pytest.raises(ValueError):
        prpl_eb_halfwidth_anytime([0, 1], delta=1.0)


# ---------------------------------------------------------------------------
# Center function
# ---------------------------------------------------------------------------


def test_center_within_unit_interval() -> None:
    """The PrPl-weighted center is in [0, 1]."""
    rng = random.Random(0)
    phis = [1 if rng.random() < 0.3 else 0 for _ in range(500)]
    m = prpl_eb_center(phis, delta=0.05)
    assert 0.0 <= m <= 1.0


def test_center_tracks_sample_mean_asymptotically() -> None:
    """For large n, the PrPl-weighted center is close to the sample mean."""
    rng = random.Random(0)
    phis = [1 if rng.random() < 0.3 else 0 for _ in range(5000)]
    center = prpl_eb_center(phis, delta=0.05)
    sample_mean = sum(phis) / len(phis)
    # Allow a few percent — PrPl is a weighted estimator with predictable
    # weights, not exactly the plain mean, but the two converge.
    assert abs(center - sample_mean) < 0.05


# ---------------------------------------------------------------------------
# End-to-end via estimate()
# ---------------------------------------------------------------------------


def test_scheduler_config_accepts_betting_method() -> None:
    cfg = SchedulerConfig(method="betting")
    assert cfg.method == "betting"


def test_estimate_with_betting_method() -> None:
    """``method='betting'`` plumbs through ``estimate`` cleanly and
    converges to the correct answer on a trivial axis-aligned program."""

    def f(x: int) -> int:
        return x

    result = estimate(
        program=f,
        distribution={"x": Uniform(1, 10)},
        property_fn=lambda y: y < 5,
        epsilon=0.05,
        delta=0.05,
        budget=2000,
        seed=0,
        method="betting",
    )
    # Closed-form truth: 4/10.
    assert abs(result.mu_hat - 0.4) < 1e-9
    lo, hi = result.interval
    assert lo <= 0.4 <= hi


def test_betting_interval_contains_truth_on_rare_event() -> None:
    """On the canonical low-mu pedagogical example, betting must produce
    a sound interval — the regime where it should outperform other
    methods."""

    def coin(x: int) -> int:
        if x < 10:
            return 0
        if x < 100:
            return 1
        if x % 1000 == 0:
            return 1
        return 0

    result = estimate(
        program=coin,
        distribution={"x": Uniform(1, 9999)},
        property_fn=lambda y: y == 1,
        epsilon=0.05,
        delta=0.05,
        budget=2000,
        seed=0,
        method="betting",
    )
    truth = 99 / 9999  # ≈ 0.0099
    lo, hi = result.interval
    assert lo <= truth <= hi, (
        f"betting interval [{lo:.4f}, {hi:.4f}] missed truth {truth:.4f}"
    )
