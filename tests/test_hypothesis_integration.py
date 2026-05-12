"""Tests for ``dise.integrations.hypothesis``."""

from __future__ import annotations

import math

import pytest

# Tests use Hypothesis as a library (not as a test runner) — skip the
# whole module gracefully if it's not installed.
pytest.importorskip("hypothesis")

import hypothesis.strategies as st

from dise.distributions import Uniform
from dise.integrations.hypothesis import (
    auto_from_strategy,
    estimate_from_strategies,
    estimate_from_strategy,
    from_integers,
    from_sampled_from,
)

# ---------------------------------------------------------------------------
# Explicit constructors
# ---------------------------------------------------------------------------


def test_from_integers_returns_uniform() -> None:
    d = from_integers(1, 100)
    assert isinstance(d, Uniform)
    assert d.lo == 1 and d.hi == 100


def test_from_integers_invalid_range() -> None:
    with pytest.raises(ValueError):
        from_integers(10, 5)


def test_from_sampled_from_consecutive() -> None:
    d = from_sampled_from([3, 4, 5, 6, 7])
    assert isinstance(d, Uniform)
    assert d.lo == 3 and d.hi == 7


def test_from_sampled_from_nonconsecutive_raises() -> None:
    with pytest.raises(NotImplementedError):
        from_sampled_from([1, 3, 5, 7])


# ---------------------------------------------------------------------------
# Auto-detect via hypothesis strategy introspection
# ---------------------------------------------------------------------------


def test_auto_from_strategy_integers() -> None:
    s = st.integers(min_value=1, max_value=100)
    d = auto_from_strategy(s)
    assert isinstance(d, Uniform)
    assert d.lo == 1 and d.hi == 100


def test_auto_from_strategy_sampled_from() -> None:
    s = st.sampled_from([3, 4, 5])
    d = auto_from_strategy(s)
    assert isinstance(d, Uniform)
    assert d.lo == 3 and d.hi == 5


def test_auto_from_strategy_unsupported_raises() -> None:
    # st.lists() is not in Tier 1.
    s = st.lists(st.integers())
    with pytest.raises(NotImplementedError):
        auto_from_strategy(s)


def test_auto_from_strategy_unbounded_integers_raises() -> None:
    s = st.integers()  # no min/max
    with pytest.raises(NotImplementedError):
        auto_from_strategy(s)


# ---------------------------------------------------------------------------
# estimate_from_strategy / estimate_from_strategies
# ---------------------------------------------------------------------------


def test_estimate_from_strategy_simple() -> None:
    """Pr[x > 50] under Uniform(1, 100) = 50/100 = 0.5."""
    result = estimate_from_strategy(
        st.integers(min_value=1, max_value=100),
        property_fn=lambda x: x > 50,
        budget=2000,
        seed=0,
    )
    assert math.isclose(result.mu_hat, 0.5, abs_tol=1e-9)
    lo, hi = result.interval
    assert lo <= 0.5 <= hi


def test_estimate_from_strategies_overflow_property() -> None:
    """Pr[a*b >= 256] for a, b ~ Uniform(1, 31)."""
    result = estimate_from_strategies(
        strategies={
            "a": st.integers(min_value=1, max_value=31),
            "b": st.integers(min_value=1, max_value=31),
        },
        property_fn=lambda a, b: a * b >= 256,
        budget=2000,
        seed=0,
    )
    # ≈ 0.40 by enumeration; allow loose bound (Mock backend, no SMT shortcut).
    assert 0.30 < result.mu_hat < 0.50


def test_estimate_from_strategy_with_explicit_program() -> None:
    """Pass an explicit ``program`` to transform inputs before the property."""

    def double(x: int) -> int:
        return x * 2

    result = estimate_from_strategy(
        st.integers(min_value=1, max_value=10),
        program=double,
        property_fn=lambda y: y > 10,  # y > 10 iff x > 5
        budget=2000,
        seed=0,
    )
    # x > 5 has probability 5/10 = 0.5. Under MockBackend the linear
    # arithmetic clause `2*x > 10` falls to a GeneralRegion (no simple
    # bound extraction), so we tolerate IS noise on mu_hat. The Z3
    # backend handles this exactly.
    assert abs(result.mu_hat - 0.5) < 0.05
