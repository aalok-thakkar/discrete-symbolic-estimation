"""Tests for ``dise.concolic``."""

from __future__ import annotations

import pytest

from dise.concolic import run_concolic
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
# Simple programs
# ---------------------------------------------------------------------------


def test_single_branch_taken(backend) -> None:
    def prog(x):
        if x < 10:
            return "small"
        return "big"

    # property comparison is on a Python string => no extra branch recorded
    result = run_concolic(prog, {"x": 5}, lambda y: y == "small", backend)
    assert result.output == "small"
    assert result.phi_value == 1
    assert result.terminated is True
    assert result.n_branches == 1


def test_single_branch_not_taken(backend) -> None:
    def prog(x):
        if x < 10:
            return "small"
        return "big"

    result = run_concolic(prog, {"x": 50}, lambda y: y == "small", backend)
    assert result.output == "big"
    assert result.phi_value == 0
    assert result.n_branches == 1


# ---------------------------------------------------------------------------
# GCD-style program
# ---------------------------------------------------------------------------


def _gcd_with_steps(a, b):
    steps = 0
    while b != 0:
        a, b = b, a % b
        steps += 1
    return a, steps


def test_gcd_48_18_branch_count(backend) -> None:
    # gcd(48, 18): 48 % 18 = 12; 18 % 12 = 6; 12 % 6 = 0 → 3 loop iters.
    # Branch decisions for `while b != 0`: TRUE, TRUE, TRUE, FALSE = 4 program
    # branches. Plus one branch from `out[0] == 6` on the symbolic output.
    result = run_concolic(
        _gcd_with_steps,
        {"a": 48, "b": 18},
        lambda out: out[0] == 6,
        backend,
    )
    assert result.output == (6, 3)
    assert result.phi_value == 1
    assert result.n_branches == 5  # 4 program branches + 1 property branch
    assert result.terminated is True


def test_gcd_coprime(backend) -> None:
    # gcd(7, 5): 7%5=2; 5%2=1; 2%1=0 → 3 loop iters, output gcd=1
    result = run_concolic(
        _gcd_with_steps,
        {"a": 7, "b": 5},
        lambda out: out[0] == 1,
        backend,
    )
    assert result.output == (1, 3)
    assert result.phi_value == 1


# ---------------------------------------------------------------------------
# Loop bodies record per-iteration branches
# ---------------------------------------------------------------------------


def test_loop_records_per_iter_branches(backend) -> None:
    def prog(x):
        count = 0
        for _ in range(5):
            if x > 0:
                count += 1
        return count

    result = run_concolic(prog, {"x": 3}, lambda y: y == 5, backend)
    # `for _ in range(5)` uses Python iteration (no comparison branches);
    # `if x > 0:` records one branch per iteration → 5 program branches.
    # `count` is a plain Python int (it was initialized to 0, an int
    # literal), so the property comparison `5 == 5` is a Python int/int
    # equality and records no branch.
    assert result.n_branches == 5
    assert result.output == 5
    assert result.phi_value == 1


# ---------------------------------------------------------------------------
# Compound clauses: x + y < 10 records ONE branch (the conjunction is
# evaluated as a single Python `<` operation on the result of `+`).
# ---------------------------------------------------------------------------


def test_compound_clause_is_one_branch(backend) -> None:
    def prog(x, y):
        if x + y < 10:
            return 1
        return 0

    # Property: `out == 1`. The output is an int (1 or 0 — Python literals),
    # not a SymbolicInt, so no property branch is recorded.
    result = run_concolic(prog, {"x": 3, "y": 4}, lambda y: y == 1, backend)
    assert result.n_branches == 1
    assert result.output == 1


# ---------------------------------------------------------------------------
# bool() records `x != 0`
# ---------------------------------------------------------------------------


def test_bool_records_nonzero_branch(backend) -> None:
    def prog(x):
        if x:
            return 1
        return 0

    result = run_concolic(prog, {"x": 5}, lambda y: y == 1, backend)
    assert result.n_branches == 1


# ---------------------------------------------------------------------------
# Arithmetic combination
# ---------------------------------------------------------------------------


def test_arithmetic_chain(backend) -> None:
    def prog(x, y):
        z = x * 2 + y
        if z > 10:
            return "big"
        return "small"

    result = run_concolic(prog, {"x": 3, "y": 4}, lambda v: v == "big", backend)
    # z = 6 + 4 = 10 → 10 > 10 is False → "small"
    assert result.output == "small"
    assert result.phi_value == 0


def test_unary_neg(backend) -> None:
    def prog(x):
        return -x

    result = run_concolic(prog, {"x": 5}, lambda y: y == -5, backend)
    assert result.output == -5
    assert result.phi_value == 1


# ---------------------------------------------------------------------------
# Divergence: max_branches limit
# ---------------------------------------------------------------------------


def test_max_branches_truncates_diverging_run(backend) -> None:
    def infinite_loop(x):
        while True:
            x = x + 1
            if x > 0:  # always True
                pass
        return x

    result = run_concolic(
        infinite_loop, {"x": 0}, lambda y: True, backend, max_branches=20
    )
    assert result.terminated is False
    assert result.n_branches == 20


# ---------------------------------------------------------------------------
# Path condition validity: each clause is satisfiable individually
# ---------------------------------------------------------------------------


def test_path_condition_clauses_satisfiable(backend) -> None:
    def prog(x):
        if x < 10:
            if x > 3:
                return "mid"
            return "low"
        return "high"

    # String output ⇒ property is `"mid" == "mid"`, no symbolic branch.
    result = run_concolic(prog, {"x": 5}, lambda v: v == "mid", backend)
    assert result.output == "mid"
    pc = backend.conjunction(*[br.clause_taken for br in result.path_condition])
    assert backend.is_satisfiable(pc) in ("sat", "unknown")


# ---------------------------------------------------------------------------
# Reverse-arithmetic operators (e.g. const + SymbolicInt)
# ---------------------------------------------------------------------------


def test_reverse_arithmetic(backend) -> None:
    def prog(x):
        # 100 - x triggers __rsub__
        if 100 - x < 50:
            return "big_x"
        return "small_x"

    result = run_concolic(prog, {"x": 60}, lambda v: v == "big_x", backend)
    assert result.output == "big_x"
    assert result.phi_value == 1
