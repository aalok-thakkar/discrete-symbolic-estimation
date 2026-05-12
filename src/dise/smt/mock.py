"""Mock SMT backend.

Conservative backend used when ``z3-solver`` is unavailable. Handles axis-
aligned conjunctions over integer variables; returns ``"unknown"`` for
anything more complex. This is sufficient to run DiSE end-to-end on
axis-aligned benchmarks (uniform inputs, bound checks, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

from .base import SUPPORTED_OPS, SatResult, SMTBackend


@dataclass(frozen=True)
class MockExpr:
    """A small AST node used by :class:`MockBackend`.

    The ``op`` string is one of ``"var"``, ``"const"``, or any operator in
    ``SUPPORTED_OPS``. ``args`` is a tuple whose contents depend on ``op``:

    * ``"var"``:    ``(name,)``         where ``name: str``
    * ``"const"``:  ``(value,)``        where ``value: int``
    * anything else: ``(MockExpr, ...)``
    """

    op: str
    args: tuple

    def __repr__(self) -> str:
        return _render(self)


def _render(e: MockExpr) -> str:
    if e.op == "var":
        return str(e.args[0])
    if e.op == "const":
        return str(e.args[0])
    if e.op == "neg":
        return f"(-{_render(e.args[0])})"
    if e.op == "not":
        return f"(not {_render(e.args[0])})"
    if e.op in {"and", "or"}:
        inner = f" {e.op} ".join(_render(a) for a in e.args)
        return f"({inner})"
    # binary
    if len(e.args) == 2:
        return f"({_render(e.args[0])} {e.op} {_render(e.args[1])})"
    return f"{e.op}({', '.join(_render(a) for a in e.args)})"


_TRUE = MockExpr("const", (True,))
_FALSE = MockExpr("const", (False,))


def _is_bool_const(e: MockExpr) -> bool:
    return e.op == "const" and isinstance(e.args[0], bool)


def _eval(e: MockExpr, env: dict[str, int]) -> int | bool:
    if e.op == "var":
        name = e.args[0]
        if name not in env:
            raise KeyError(f"missing variable '{name}' in evaluate")
        return env[name]
    if e.op == "const":
        return e.args[0]
    # Short-circuit logical ops so a False conjunct can mask later
    # arithmetic that would otherwise raise (e.g. ``mod`` by zero).
    if e.op == "and":
        for a in e.args:
            if not bool(_eval(a, env)):
                return False
        return True
    if e.op == "or":
        for a in e.args:
            if bool(_eval(a, env)):
                return True
        return False
    if e.op == "not":
        return not bool(_eval(e.args[0], env))
    args = [_eval(a, env) for a in e.args]
    if e.op == "+":
        return args[0] + args[1]
    if e.op == "-":
        return args[0] - args[1]
    if e.op == "*":
        return args[0] * args[1]
    if e.op == "div":
        if args[1] == 0:
            raise ZeroDivisionError("div by zero in mock evaluation")
        return args[0] // args[1]
    if e.op == "mod":
        if args[1] == 0:
            raise ZeroDivisionError("mod by zero in mock evaluation")
        return args[0] % args[1]
    if e.op == "neg":
        return -args[0]
    if e.op == "==":
        return args[0] == args[1]
    if e.op == "!=":
        return args[0] != args[1]
    if e.op == "<":
        return args[0] < args[1]
    if e.op == "<=":
        return args[0] <= args[1]
    if e.op == ">":
        return args[0] > args[1]
    if e.op == ">=":
        return args[0] >= args[1]
    raise ValueError(f"unknown mock op: {e.op}")


def _free_vars(e: MockExpr) -> frozenset[str]:
    if e.op == "var":
        return frozenset({e.args[0]})
    if e.op == "const":
        return frozenset()
    out: set[str] = set()
    for a in e.args:
        out |= _free_vars(a)
    return frozenset(out)


def _top_level_conjuncts(e: MockExpr) -> list[MockExpr]:
    if e.op == "and":
        out: list[MockExpr] = []
        for a in e.args:
            out.extend(_top_level_conjuncts(a))
        return out
    if _is_bool_const(e) and e.args[0] is True:
        return []
    return [e]


def _extract_simple_bound(
    clause: MockExpr, var: str
) -> tuple[int | None, int | None] | None:
    """If ``clause`` is of the form ``Var(var) <op> Const`` or its swap,
    return ``(lo_bound, hi_bound)``; otherwise None."""
    if clause.op not in {"<", "<=", ">", ">=", "==", "!="}:
        return None
    if len(clause.args) != 2:
        return None
    lhs, rhs = clause.args
    op = clause.op
    if lhs.op == "var" and lhs.args[0] == var and rhs.op == "const" and isinstance(rhs.args[0], int) and not isinstance(rhs.args[0], bool):
        c = rhs.args[0]
    elif rhs.op == "var" and rhs.args[0] == var and lhs.op == "const" and isinstance(lhs.args[0], int) and not isinstance(lhs.args[0], bool):
        c = lhs.args[0]
        op = _flip_op(op)
    else:
        return None
    if op == "<":
        return (None, c - 1)
    if op == "<=":
        return (None, c)
    if op == ">":
        return (c + 1, None)
    if op == ">=":
        return (c, None)
    if op == "==":
        return (c, c)
    # "!=" cannot be reduced to a closed-interval bound. Signal failure
    # so the caller falls back to GeneralRegion (which evaluates the
    # full predicate); otherwise the disallowed point would be silently
    # included in an AxisAlignedBox.
    return None


def _flip_op(op: str) -> str:
    return {
        "<": ">",
        "<=": ">=",
        ">": "<",
        ">=": "<=",
        "==": "==",
        "!=": "!=",
    }[op]


class MockBackend(SMTBackend):
    """Backend with no SMT solver. Sound but very conservative."""

    def __init__(self) -> None:
        self._vars: dict[str, MockExpr] = {}

    def make_int_var(self, name: str) -> MockExpr:
        if name not in self._vars:
            self._vars[name] = MockExpr("var", (name,))
        return self._vars[name]

    def const(self, value: int) -> MockExpr:
        return MockExpr("const", (int(value),))

    def op(self, op_name: str, *args: MockExpr) -> MockExpr:
        if op_name not in SUPPORTED_OPS:
            raise ValueError(f"unsupported op {op_name!r}")
        for a in args:
            if not isinstance(a, MockExpr):
                raise TypeError(f"expected MockExpr arg, got {type(a)!r}")
        if op_name == "and":
            return self.conjunction(*args)
        if op_name == "or":
            if len(args) == 0:
                return _FALSE
            if len(args) == 1:
                return args[0]
            return MockExpr("or", tuple(args))
        if op_name == "not":
            return self.negation(args[0])
        # Trivial simplifications: comparisons of structurally identical operands.
        if op_name in {"==", "<=", ">="} and len(args) == 2 and args[0] == args[1]:
            return _TRUE
        if op_name in {"!=", "<", ">"} and len(args) == 2 and args[0] == args[1]:
            return _FALSE
        return MockExpr(op_name, tuple(args))

    def true(self) -> MockExpr:
        return _TRUE

    def false(self) -> MockExpr:
        return _FALSE

    def conjunction(self, *args: MockExpr) -> MockExpr:
        flat: list[MockExpr] = []
        for a in args:
            if _is_bool_const(a):
                if a.args[0] is False:
                    return _FALSE
                # True: skip
                continue
            if a.op == "and":
                flat.extend(a.args)
            else:
                flat.append(a)
        if len(flat) == 0:
            return _TRUE
        if len(flat) == 1:
            return flat[0]
        return MockExpr("and", tuple(flat))

    def negation(self, expr: MockExpr) -> MockExpr:
        if _is_bool_const(expr):
            return _TRUE if expr.args[0] is False else _FALSE
        if expr.op == "not":
            return expr.args[0]
        # Push negation through arithmetic comparisons:
        flip = {"<": ">=", "<=": ">", ">": "<=", ">=": "<", "==": "!=", "!=": "=="}
        if expr.op in flip:
            return MockExpr(flip[expr.op], expr.args)
        return MockExpr("not", (expr,))

    def is_satisfiable(self, formula: MockExpr, timeout_ms: int = 5000) -> SatResult:
        if _is_bool_const(formula):
            return "sat" if formula.args[0] else "unsat"
        # Try axis-aligned interval intersection.
        clauses = _top_level_conjuncts(formula)
        # Each clause must be single-variable.
        bounds: dict[str, tuple[int, int]] = {}
        for c in clauses:
            fv = _free_vars(c)
            if len(fv) == 0:
                # constant — evaluate
                try:
                    val = _eval(c, {})
                    if not bool(val):
                        return "unsat"
                    continue
                except Exception:
                    return "unknown"
            if len(fv) != 1:
                return "unknown"
            (v,) = fv
            extracted = _extract_simple_bound(c, v)
            if extracted is None:
                return "unknown"
            lo_c, hi_c = extracted
            cur_lo, cur_hi = bounds.get(v, (-10**18, 10**18))
            if lo_c is not None:
                cur_lo = max(cur_lo, lo_c)
            if hi_c is not None:
                cur_hi = min(cur_hi, hi_c)
            if cur_lo > cur_hi:
                return "unsat"
            bounds[v] = (cur_lo, cur_hi)
        return "sat"

    def is_axis_aligned(self, formula: MockExpr) -> bool:
        if _is_bool_const(formula):
            return True
        for c in _top_level_conjuncts(formula):
            if len(_free_vars(c)) > 1:
                return False
        return True

    def top_level_conjuncts(self, formula: MockExpr) -> list[MockExpr]:
        return _top_level_conjuncts(formula)

    def extract_var_bound(
        self, clause: MockExpr, var: str
    ) -> tuple[int | None, int | None] | None:
        return _extract_simple_bound(clause, var)

    def project_to_variable(
        self, formula: MockExpr, var: str
    ) -> tuple[int, int] | None:
        clauses = _top_level_conjuncts(formula)
        lo: int | None = None
        hi: int | None = None
        for c in clauses:
            if var not in _free_vars(c):
                continue
            if _free_vars(c) != {var}:
                return None
            extracted = _extract_simple_bound(c, var)
            if extracted is None:
                return None
            c_lo, c_hi = extracted
            if c_lo is not None:
                lo = c_lo if lo is None else max(lo, c_lo)
            if c_hi is not None:
                hi = c_hi if hi is None else min(hi, c_hi)
        if lo is None or hi is None:
            return None
        if lo > hi:
            return None
        return (lo, hi)

    def free_vars(self, formula: MockExpr) -> frozenset[str]:
        return _free_vars(formula)

    def evaluate(self, formula: MockExpr, assignment: dict[str, int]) -> bool:
        result = _eval(formula, assignment)
        if not isinstance(result, bool):
            raise ValueError(
                f"evaluate did not produce a Boolean: {result!r} from {formula!r}"
            )
        return result

    def repr_expr(self, expr: MockExpr) -> str:
        return _render(expr)


__all__ = ["MockBackend", "MockExpr"]
