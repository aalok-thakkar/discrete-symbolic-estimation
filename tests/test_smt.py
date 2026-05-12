"""Tests for ``dise.smt``.

Tests split into:

* :class:`TestBothBackends` — parametrized over Mock and (if installed) Z3;
  covers the axis-aligned subset where Mock should agree with Z3.
* :class:`TestZ3Only`       — tests requiring real SMT reasoning.
* :class:`TestMockOnly`     — tests of Mock-specific conservative behavior.
"""

from __future__ import annotations

import pytest

from dise.smt import MockBackend, SMTBackend, default_backend, has_z3
from dise.smt import Z3Backend as _Z3Backend

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _backend_params() -> list[pytest.param]:
    out = [pytest.param("mock", id="mock")]
    if has_z3():
        out.append(pytest.param("z3", id="z3"))
    return out


@pytest.fixture(params=_backend_params())
def backend(request: pytest.FixtureRequest) -> SMTBackend:
    name = request.param
    if name == "mock":
        return MockBackend()
    assert _Z3Backend is not None
    return _Z3Backend()


@pytest.fixture
def z3_backend() -> SMTBackend:
    if not has_z3() or _Z3Backend is None:
        pytest.skip("z3-solver not installed")
    return _Z3Backend()


@pytest.fixture
def mock_backend() -> MockBackend:
    return MockBackend()


# ---------------------------------------------------------------------------
# TestBothBackends: axis-aligned subset shared by both
# ---------------------------------------------------------------------------


class TestBothBackends:
    def test_make_int_var_is_idempotent(self, backend: SMTBackend) -> None:
        x1 = backend.make_int_var("x")
        x2 = backend.make_int_var("x")
        formula = backend.op("==", x1, x2)
        assert backend.is_satisfiable(formula) == "sat"
        not_eq = backend.op("!=", x1, x2)
        assert backend.is_satisfiable(not_eq) == "unsat"

    def test_const_satisfiable(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        expr = backend.op("<", x, backend.const(10))
        assert backend.is_satisfiable(expr) == "sat"

    def test_contradiction_unsatisfiable(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        expr = backend.conjunction(
            backend.op(">", x, backend.const(5)),
            backend.op("<", x, backend.const(3)),
        )
        assert backend.is_satisfiable(expr) == "unsat"

    def test_simple_interval_satisfiable(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        expr = backend.conjunction(
            backend.op(">", x, backend.const(5)),
            backend.op("<", x, backend.const(10)),
        )
        assert backend.is_satisfiable(expr) == "sat"

    def test_true_and_false_constants(self, backend: SMTBackend) -> None:
        assert backend.is_satisfiable(backend.true()) == "sat"
        assert backend.is_satisfiable(backend.false()) == "unsat"

    def test_repr_expr_returns_string(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        expr = backend.op("<", x, backend.const(10))
        s = backend.repr_expr(expr)
        assert isinstance(s, str) and len(s) > 0

    def test_axis_aligned_single_var(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        expr = backend.op("<", x, backend.const(10))
        assert backend.is_axis_aligned(expr) is True

    def test_axis_aligned_conjunction(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        y = backend.make_int_var("y")
        expr = backend.conjunction(
            backend.op("<", x, backend.const(10)),
            backend.op(">", y, backend.const(3)),
        )
        assert backend.is_axis_aligned(expr) is True

    def test_not_axis_aligned_mixed_vars(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        y = backend.make_int_var("y")
        expr = backend.op("<", backend.op("+", x, y), backend.const(10))
        assert backend.is_axis_aligned(expr) is False

    def test_project_simple_interval(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        y = backend.make_int_var("y")
        expr = backend.conjunction(
            backend.op(">=", x, backend.const(3)),
            backend.op("<=", x, backend.const(8)),
            backend.op("<", y, backend.const(5)),
        )
        assert backend.project_to_variable(expr, "x") == (3, 8)

    def test_project_strict_inequality(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        expr = backend.conjunction(
            backend.op(">", x, backend.const(3)),
            backend.op("<", x, backend.const(8)),
        )
        assert backend.project_to_variable(expr, "x") == (4, 7)

    def test_project_equality(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        expr = backend.op("==", x, backend.const(7))
        assert backend.project_to_variable(expr, "x") == (7, 7)

    def test_project_unbounded_returns_none(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        expr = backend.op(">=", x, backend.const(3))
        assert backend.project_to_variable(expr, "x") is None

    def test_project_unrelated_variable_returns_none(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        backend.make_int_var("y")
        expr = backend.op("<", x, backend.const(10))
        assert backend.project_to_variable(expr, "y") is None

    def test_free_vars_basic(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        y = backend.make_int_var("y")
        expr = backend.conjunction(
            backend.op("<", x, backend.const(10)),
            backend.op(">", y, backend.const(3)),
        )
        assert backend.free_vars(expr) == frozenset({"x", "y"})

    def test_free_vars_const_is_empty(self, backend: SMTBackend) -> None:
        assert backend.free_vars(backend.const(5)) == frozenset()

    def test_evaluate_true(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        expr = backend.op("<", x, backend.const(10))
        assert backend.evaluate(expr, {"x": 5}) is True

    def test_evaluate_false(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        expr = backend.op("<", x, backend.const(10))
        assert backend.evaluate(expr, {"x": 20}) is False

    def test_evaluate_conjunction(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        y = backend.make_int_var("y")
        expr = backend.conjunction(
            backend.op("<", x, backend.const(10)),
            backend.op(">", y, backend.const(3)),
        )
        assert backend.evaluate(expr, {"x": 5, "y": 7}) is True
        assert backend.evaluate(expr, {"x": 5, "y": 3}) is False

    def test_evaluate_arithmetic(self, backend: SMTBackend) -> None:
        x = backend.make_int_var("x")
        y = backend.make_int_var("y")
        expr = backend.op("==", backend.op("+", x, y), backend.const(10))
        assert backend.evaluate(expr, {"x": 3, "y": 7}) is True
        assert backend.evaluate(expr, {"x": 3, "y": 6}) is False


# ---------------------------------------------------------------------------
# TestZ3Only: queries requiring real SMT reasoning
# ---------------------------------------------------------------------------


class TestZ3Only:
    def test_arithmetic_chain(self, z3_backend: SMTBackend) -> None:
        x = z3_backend.make_int_var("x")
        # x + 1 == 6  AND  x == 5
        expr = z3_backend.conjunction(
            z3_backend.op("==", x, z3_backend.const(5)),
            z3_backend.op("==", z3_backend.op("+", x, z3_backend.const(1)), z3_backend.const(6)),
        )
        assert z3_backend.is_satisfiable(expr) == "sat"

    def test_arithmetic_contradiction(self, z3_backend: SMTBackend) -> None:
        x = z3_backend.make_int_var("x")
        # x == 5  AND  x + 1 == 7
        expr = z3_backend.conjunction(
            z3_backend.op("==", x, z3_backend.const(5)),
            z3_backend.op("==", z3_backend.op("+", x, z3_backend.const(1)), z3_backend.const(7)),
        )
        assert z3_backend.is_satisfiable(expr) == "unsat"

    def test_non_axis_aligned_satisfiable(self, z3_backend: SMTBackend) -> None:
        x = z3_backend.make_int_var("x")
        y = z3_backend.make_int_var("y")
        expr = z3_backend.op("<", z3_backend.op("+", x, y), z3_backend.const(10))
        assert z3_backend.is_satisfiable(expr) == "sat"

    def test_non_axis_aligned_unsat(self, z3_backend: SMTBackend) -> None:
        x = z3_backend.make_int_var("x")
        y = z3_backend.make_int_var("y")
        # x + y < 10  AND  x >= 10  AND  y >= 1
        expr = z3_backend.conjunction(
            z3_backend.op("<", z3_backend.op("+", x, y), z3_backend.const(10)),
            z3_backend.op(">=", x, z3_backend.const(10)),
            z3_backend.op(">=", y, z3_backend.const(1)),
        )
        assert z3_backend.is_satisfiable(expr) == "unsat"


# ---------------------------------------------------------------------------
# TestMockOnly: conservative-by-design behavior
# ---------------------------------------------------------------------------


class TestMockOnly:
    def test_returns_unknown_for_non_axis_aligned(self, mock_backend: MockBackend) -> None:
        x = mock_backend.make_int_var("x")
        y = mock_backend.make_int_var("y")
        expr = mock_backend.op("<", mock_backend.op("+", x, y), mock_backend.const(10))
        assert mock_backend.is_satisfiable(expr) == "unknown"

    def test_returns_unknown_for_complex_single_var(self, mock_backend: MockBackend) -> None:
        x = mock_backend.make_int_var("x")
        # (x + 1) < 10 — single variable but not in simple "var op const" form.
        expr = mock_backend.op("<", mock_backend.op("+", x, mock_backend.const(1)), mock_backend.const(10))
        assert mock_backend.is_satisfiable(expr) == "unknown"

    def test_evaluate_works_on_arithmetic(self, mock_backend: MockBackend) -> None:
        # evaluate is fully concrete and should not return unknown.
        x = mock_backend.make_int_var("x")
        expr = mock_backend.op("<", mock_backend.op("+", x, mock_backend.const(1)), mock_backend.const(10))
        assert mock_backend.evaluate(expr, {"x": 5}) is True
        assert mock_backend.evaluate(expr, {"x": 20}) is False


# ---------------------------------------------------------------------------
# default_backend
# ---------------------------------------------------------------------------


def test_default_backend_returns_backend() -> None:
    b = default_backend()
    assert isinstance(b, SMTBackend)
