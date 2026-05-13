"""Z3-backed SMT backend for DiSE.

Uses ``z3-solver``. The module is imported lazily by ``dise.smt.__init__``
so that DiSE can fall back to :class:`~dise.smt.MockBackend` when z3 is not
installed.
"""

from __future__ import annotations

from typing import Any

import z3  # noqa: F401  (raises ImportError if unavailable)

from .base import SUPPORTED_OPS, SatResult, SMTBackend


def _is_int_value(e: Any) -> bool:
    try:
        return z3.is_int_value(e)
    except Exception:
        return False


class Z3Backend(SMTBackend):
    """SMT backend backed by Z3.

    Variables are integer-sorted. Operators map to z3's native
    arithmetic/Boolean operators. ``div`` uses z3's Euclidean integer
    division (note: this differs from Python's floor division on negatives;
    the prototype's benchmarks operate on non-negative integers).
    """

    def __init__(self) -> None:
        self._vars: dict[str, z3.ArithRef] = {}

    def make_int_var(self, name: str) -> z3.ArithRef:
        if name not in self._vars:
            self._vars[name] = z3.Int(name)
        return self._vars[name]

    def const(self, value: int) -> z3.ArithRef:
        return z3.IntVal(int(value))

    def op(self, op_name: str, *args: Any) -> Any:
        if op_name not in SUPPORTED_OPS:
            raise ValueError(f"unsupported op {op_name!r}")
        if op_name == "+":
            return args[0] + args[1]
        if op_name == "-":
            return args[0] - args[1]
        if op_name == "*":
            return args[0] * args[1]
        if op_name == "div":
            return args[0] / args[1]
        if op_name == "mod":
            return args[0] % args[1]
        if op_name == "neg":
            return -args[0]
        if op_name == "==":
            return args[0] == args[1]
        if op_name == "!=":
            return args[0] != args[1]
        if op_name == "<":
            return args[0] < args[1]
        if op_name == "<=":
            return args[0] <= args[1]
        if op_name == ">":
            return args[0] > args[1]
        if op_name == ">=":
            return args[0] >= args[1]
        if op_name == "and":
            return self.conjunction(*args)
        if op_name == "or":
            if len(args) == 0:
                return self.false()
            if len(args) == 1:
                return args[0]
            return z3.Or(*args)
        if op_name == "not":
            return self.negation(args[0])
        raise AssertionError(f"unreachable: {op_name}")

    def true(self) -> Any:
        return z3.BoolVal(True)

    def false(self) -> Any:
        return z3.BoolVal(False)

    def conjunction(self, *args: Any) -> Any:
        flat = []
        for a in args:
            if z3.is_true(a):
                continue
            if z3.is_false(a):
                return self.false()
            if z3.is_and(a):
                flat.extend(a.children())
            else:
                flat.append(a)
        if len(flat) == 0:
            return self.true()
        if len(flat) == 1:
            return flat[0]
        return z3.And(*flat)

    def negation(self, expr: Any) -> Any:
        if z3.is_true(expr):
            return self.false()
        if z3.is_false(expr):
            return self.true()
        if z3.is_not(expr):
            return expr.children()[0]
        # Push negation through inequality / equality comparisons:
        if expr.num_args() == 2:
            lhs, rhs = expr.children()
            kind = expr.decl().kind()
            if kind == z3.Z3_OP_LT:
                return lhs >= rhs
            if kind == z3.Z3_OP_LE:
                return lhs > rhs
            if kind == z3.Z3_OP_GT:
                return lhs <= rhs
            if kind == z3.Z3_OP_GE:
                return lhs < rhs
            if kind == z3.Z3_OP_EQ:
                return lhs != rhs
            if kind == z3.Z3_OP_DISTINCT:
                return lhs == rhs
        return z3.Not(expr)

    def is_satisfiable(self, formula: Any, timeout_ms: int = 5000) -> SatResult:
        if z3.is_true(formula):
            return "sat"
        if z3.is_false(formula):
            return "unsat"
        s = z3.Solver()
        s.set("timeout", int(timeout_ms))
        s.add(formula)
        r = s.check()
        if r == z3.sat:
            return "sat"
        if r == z3.unsat:
            return "unsat"
        return "unknown"

    # ---- axis-aligned helpers ----

    def top_level_conjuncts(self, formula: Any) -> list[Any]:
        if z3.is_true(formula):
            return []
        if z3.is_and(formula):
            out: list[Any] = []
            for c in formula.children():
                out.extend(self.top_level_conjuncts(c))
            return out
        return [formula]

    def is_axis_aligned(self, formula: Any) -> bool:
        for c in self.top_level_conjuncts(formula):
            if len(self.free_vars(c)) > 1:
                return False
        return True

    def extract_var_bound(
        self, clause: Any, var: str
    ) -> tuple[int | None, int | None] | None:
        var_expr = self.make_int_var(var)
        return self._extract_simple_bound(clause, var_expr)

    def project_to_variable(self, formula: Any, var: str) -> tuple[int, int] | None:
        var_expr = self.make_int_var(var)
        lo: int | None = None
        hi: int | None = None
        for c in self.top_level_conjuncts(formula):
            fv = self.free_vars(c)
            if var not in fv:
                continue
            if fv != frozenset({var}):
                return None
            extracted = self._extract_simple_bound(c, var_expr)
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

    def _extract_simple_bound(
        self, clause: Any, var_expr: Any
    ) -> tuple[int | None, int | None] | None:
        # Match `var op const` or `const op var` where op in
        # {<, <=, >, >=, ==, !=}. Handle Not(.) for "!=" inversions.
        if z3.is_not(clause):
            inner = clause.children()[0]
            base = self._extract_simple_bound(inner, var_expr)
            if base is None:
                return None
            # Negation of an interval clause may not be an interval — return None.
            return None
        if clause.num_args() != 2:
            return None
        lhs, rhs = clause.children()
        kind = clause.decl().kind()
        if z3.eq(lhs, var_expr) and _is_int_value(rhs):
            c = rhs.as_long()
            side = "left"
        elif z3.eq(rhs, var_expr) and _is_int_value(lhs):
            c = lhs.as_long()
            side = "right"
        else:
            return None
        if kind == z3.Z3_OP_LT:
            return (None, c - 1) if side == "left" else (c + 1, None)
        if kind == z3.Z3_OP_LE:
            return (None, c) if side == "left" else (c, None)
        if kind == z3.Z3_OP_GT:
            return (c + 1, None) if side == "left" else (None, c - 1)
        if kind == z3.Z3_OP_GE:
            return (c, None) if side == "left" else (None, c)
        if kind == z3.Z3_OP_EQ:
            return (c, c)
        return None

    def free_vars(self, formula: Any) -> frozenset[str]:
        seen: set[str] = set()

        def walk(e: Any) -> None:
            if z3.is_const(e):
                d = e.decl()
                # Uninterpreted constants are user-declared variables.
                if d.kind() == z3.Z3_OP_UNINTERPRETED:
                    seen.add(d.name())
                return
            for child in e.children():
                walk(child)

        walk(formula)
        return frozenset(seen)

    def evaluate(self, formula: Any, assignment: dict[str, int]) -> bool:
        # Note: the error message is intentionally static. When the
        # formula contains arithmetic that Z3's simplifier cannot reduce
        # to a Boolean (typically ``mod`` / ``div`` by zero from a path
        # condition whose loop terminated earlier than the recorded
        # path), this exception is hit on the hot mass-split path —
        # constructing an f-string with ``{result}`` here was ~140 ms /
        # call due to Z3's Python pretty-printer walking the expression.
        subs = [(self.make_int_var(name), z3.IntVal(int(val))) for name, val in assignment.items()]
        result = z3.simplify(formula if not subs else z3.substitute(formula, *subs))
        if z3.is_true(result):
            return True
        if z3.is_false(result):
            return False
        raise ValueError("evaluate: formula did not reduce to a Boolean constant")

    def repr_expr(self, expr: Any) -> str:
        return str(expr)


__all__ = ["Z3Backend"]
