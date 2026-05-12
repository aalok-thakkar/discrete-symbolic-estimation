"""Tests for ``dise.regions`` (concrete regions + build_region)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dise.distributions import (
    BoundedGeometric,
    ProductDistribution,
    Uniform,
)
from dise.regions import (
    AxisAlignedBox,
    EmptyRegion,
    GeneralRegion,
    UnconstrainedRegion,
    build_region,
)
from dise.smt import MockBackend, has_z3
from dise.smt import Z3Backend as _Z3Backend


def _backend_params():
    out = [pytest.param("mock", id="mock")]
    if has_z3():
        out.append(pytest.param("z3", id="z3"))
    return out


@pytest.fixture(params=_backend_params())
def backend(request):
    name = request.param
    if name == "mock":
        return MockBackend()
    return _Z3Backend()


# ---------------------------------------------------------------------------
# AxisAlignedBox
# ---------------------------------------------------------------------------


def test_axis_aligned_mass_matches_enumeration(backend) -> None:
    d = ProductDistribution(
        factors={"x": Uniform(1, 10), "y": Uniform(1, 10)}
    )
    box = AxisAlignedBox(bounds={"x": (3, 7), "y": (2, 5)}, formula=backend.true())
    w, var = box.mass(d, backend, np.random.default_rng(0))
    # Brute-force: count points in the box, divide by total
    total = 0.0
    in_box = 0.0
    for x in range(1, 11):
        for y in range(1, 11):
            p = d.factors["x"].pmf(x) * d.factors["y"].pmf(y)
            total += p
            if 3 <= x <= 7 and 2 <= y <= 5:
                in_box += p
    assert math.isclose(w, in_box / total, abs_tol=1e-12)
    assert var == 0.0


def test_axis_aligned_sample_in_box(backend) -> None:
    d = ProductDistribution(
        factors={"x": Uniform(1, 10), "y": Uniform(1, 10)}
    )
    box = AxisAlignedBox(bounds={"x": (3, 7), "y": (2, 5)}, formula=backend.true())
    rng = np.random.default_rng(0)
    batch = box.sample(d, backend, rng, 500)
    assert batch.n == 500
    assert batch.inputs["x"].min() >= 3
    assert batch.inputs["x"].max() <= 7
    assert batch.inputs["y"].min() >= 2
    assert batch.inputs["y"].max() <= 5


def test_axis_aligned_contains(backend) -> None:
    box = AxisAlignedBox(bounds={"x": (3, 7)}, formula=backend.true())
    assert box.contains({"x": 5}) is True
    assert box.contains({"x": 3}) is True
    assert box.contains({"x": 7}) is True
    assert box.contains({"x": 2}) is False
    assert box.contains({"x": 8}) is False


# ---------------------------------------------------------------------------
# GeneralRegion
# ---------------------------------------------------------------------------


def test_general_region_mass_within_is_ci(backend) -> None:
    """Mass of {(a,b) in [1,10]^2: a+b<=10} under uniform.

    Truth: number of pairs with a+b<=10 in [1,10]x[1,10].
    For a=1..9, b can be 1..(10-a) giving sum_{a=1}^9 (10-a) = 9+8+...+1 = 45.
    Plus a=10, b=0 — but b>=1 so excluded.
    So truth = 45/100 = 0.45.
    """
    d = ProductDistribution(
        factors={"a": Uniform(1, 10), "b": Uniform(1, 10)}
    )
    a = backend.make_int_var("a")
    b = backend.make_int_var("b")
    formula = backend.op("<=", backend.op("+", a, b), backend.const(10))
    region = build_region(formula, d, backend)
    assert isinstance(region, GeneralRegion)
    rng = np.random.default_rng(0)
    w, var = region.mass(d, backend, rng, n_mc=5000)
    assert abs(w - 0.45) < 3.0 * math.sqrt(var + 1e-12) + 0.05


def test_general_region_contains(backend) -> None:
    d = ProductDistribution(
        factors={"a": Uniform(1, 10), "b": Uniform(1, 10)}
    )
    a = backend.make_int_var("a")
    b = backend.make_int_var("b")
    formula = backend.op("<=", backend.op("+", a, b), backend.const(10))
    region = build_region(formula, d, backend)
    assert region.contains({"a": 3, "b": 4}) is True
    assert region.contains({"a": 9, "b": 9}) is False


def test_general_region_sample_satisfies_predicate(backend) -> None:
    d = ProductDistribution(
        factors={"a": Uniform(1, 10), "b": Uniform(1, 10)}
    )
    a = backend.make_int_var("a")
    b = backend.make_int_var("b")
    formula = backend.op("<=", backend.op("+", a, b), backend.const(10))
    region = build_region(formula, d, backend)
    rng = np.random.default_rng(0)
    batch = region.sample(d, backend, rng, 200)
    assert batch.n > 0
    for x in batch.iter_assignments():
        assert x["a"] + x["b"] <= 10


# ---------------------------------------------------------------------------
# build_region
# ---------------------------------------------------------------------------


def test_build_region_axis_aligned_yields_box(backend) -> None:
    d = ProductDistribution(
        factors={"x": BoundedGeometric(0.1, 100), "y": Uniform(1, 50)}
    )
    x = backend.make_int_var("x")
    y = backend.make_int_var("y")
    formula = backend.conjunction(
        backend.op(">=", x, backend.const(3)),
        backend.op("<=", x, backend.const(8)),
        backend.op("<", y, backend.const(10)),
    )
    region = build_region(formula, d, backend)
    assert isinstance(region, AxisAlignedBox)
    # bounds reflect both formula and distribution support
    assert region.bounds["x"] == (3, 8)
    # y < 10 intersected with [1, 50] => [1, 9]
    assert region.bounds["y"] == (1, 9)


def test_build_region_one_sided_yields_box(backend) -> None:
    """One-sided constraint + distribution support => closed box."""
    d = ProductDistribution(factors={"x": BoundedGeometric(0.1, 100)})
    x = backend.make_int_var("x")
    formula = backend.op("<", x, backend.const(10))
    region = build_region(formula, d, backend)
    assert isinstance(region, AxisAlignedBox)
    # x < 10 intersected with [1, 100] => [1, 9]
    assert region.bounds["x"] == (1, 9)


def test_build_region_unsat_yields_empty(backend) -> None:
    d = ProductDistribution(factors={"x": Uniform(1, 10)})
    x = backend.make_int_var("x")
    formula = backend.conjunction(
        backend.op(">", x, backend.const(10)),
        backend.op("<", x, backend.const(5)),
    )
    region = build_region(formula, d, backend)
    assert isinstance(region, EmptyRegion)


def test_build_region_non_axis_aligned_yields_general(backend) -> None:
    d = ProductDistribution(
        factors={"a": Uniform(1, 10), "b": Uniform(1, 10)}
    )
    a = backend.make_int_var("a")
    b = backend.make_int_var("b")
    formula = backend.op("<", backend.op("+", a, b), backend.const(10))
    region = build_region(formula, d, backend)
    assert isinstance(region, GeneralRegion)


# ---------------------------------------------------------------------------
# UnconstrainedRegion + EmptyRegion
# ---------------------------------------------------------------------------


def test_unconstrained_mass_is_one(backend) -> None:
    d = ProductDistribution(factors={"x": Uniform(1, 10)})
    r = UnconstrainedRegion(backend.true())
    w, var = r.mass(d, backend, np.random.default_rng(0))
    assert w == 1.0
    assert var == 0.0


def test_empty_region_zero_mass(backend) -> None:
    d = ProductDistribution(factors={"x": Uniform(1, 10)})
    r = EmptyRegion(backend.false())
    w, var = r.mass(d, backend, np.random.default_rng(0))
    assert w == 0.0
    assert var == 0.0
    assert r.contains({"x": 5}) is False
