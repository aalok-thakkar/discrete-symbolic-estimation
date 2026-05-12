"""Tests for ``dise.smt.CachedBackend``."""

from __future__ import annotations

import pytest

from dise.smt import CachedBackend, MockBackend, has_z3
from dise.smt import Z3Backend as _Z3Backend


def _backend_params():
    out = [pytest.param("mock", id="mock")]
    if has_z3():
        out.append(pytest.param("z3", id="z3"))
    return out


@pytest.fixture(params=_backend_params())
def backend(request):
    return MockBackend() if request.param == "mock" else _Z3Backend()


def test_cache_passes_through(backend) -> None:
    cb = CachedBackend(backend)
    x = cb.make_int_var("x")
    expr = cb.op("<", x, cb.const(10))
    assert cb.is_satisfiable(expr) == "sat"
    assert cb.is_axis_aligned(expr) is True
    assert cb.evaluate(expr, {"x": 5}) is True


def test_cache_hit_on_repeated_query(backend) -> None:
    cb = CachedBackend(backend)
    x = cb.make_int_var("x")
    expr = cb.op("<", x, cb.const(10))
    for _ in range(5):
        cb.is_satisfiable(expr)
    # First call was a miss; remaining 4 should be hits.
    assert cb.stats.is_satisfiable_hits == 4
    assert cb.stats.is_satisfiable_misses == 1
    assert cb.stats.hit_rate >= 0.7


def test_cache_evaluate_hit(backend) -> None:
    cb = CachedBackend(backend)
    x = cb.make_int_var("x")
    expr = cb.op("<", x, cb.const(10))
    for _ in range(3):
        assert cb.evaluate(expr, {"x": 5}) is True
    assert cb.stats.evaluate_hits == 2
    assert cb.stats.evaluate_misses == 1


def test_cache_max_entries() -> None:
    inner = MockBackend()
    cb = CachedBackend(inner, max_entries=4)
    x = cb.make_int_var("x")
    # Issue 6 distinct queries; cache should bound to 4 entries.
    for k in range(6):
        cb.is_satisfiable(cb.op("<", x, cb.const(k)))
    assert len(cb._cache_sat) <= 4


def test_cache_invalid_max_entries_raises() -> None:
    with pytest.raises(ValueError):
        CachedBackend(MockBackend(), max_entries=0)


def test_inner_property_exposed(backend) -> None:
    cb = CachedBackend(backend)
    assert cb.inner is backend
