"""SMT backend interface.

The :class:`SMTBackend` ABC defines the contract that the rest of DiSE uses
to reason about path conditions, region feasibility, and projection. Two
implementations exist:

* :class:`dise.smt.Z3Backend`     — full LIA via z3-solver
* :class:`dise.smt.MockBackend`   — conservative; returns ``"unknown"`` for
  non-trivial queries. Sufficient to run axis-aligned programs end-to-end
  when z3 is unavailable.

All backends are *sound*: ``is_satisfiable`` never claims ``sat`` or ``unsat``
incorrectly. Callers MUST treat ``"unknown"`` as "could not refute" — never
close a region based on an ``unsat`` derived from ``"unknown"``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

# An opaque SMT expression. Backends use their own concrete types; the rest
# of DiSE treats SMTExpr as a black box that is only manipulated via the
# backend's methods.
SMTExpr = Any

SatResult = Literal["sat", "unsat", "unknown"]


# Supported operator names. Backends are required to implement every entry.
ARITH_OPS = frozenset({"+", "-", "*", "div", "mod", "neg"})
COMPARE_OPS = frozenset({"==", "!=", "<", "<=", ">", ">="})
LOGIC_OPS = frozenset({"and", "or", "not"})
SUPPORTED_OPS = ARITH_OPS | COMPARE_OPS | LOGIC_OPS


@dataclass(frozen=True)
class Clause:
    """A single clause in a path condition (atomic Boolean expression)."""

    expr: SMTExpr
    repr_string: str

    def __repr__(self) -> str:
        return f"Clause({self.repr_string})"


class SMTBackend(ABC):
    """Abstract SMT backend. See module docstring for guarantees."""

    @abstractmethod
    def make_int_var(self, name: str) -> SMTExpr:
        """Return (and cache) an integer variable named ``name``."""

    @abstractmethod
    def const(self, value: int) -> SMTExpr:
        """Return a constant integer expression."""

    @abstractmethod
    def op(self, op_name: str, *args: SMTExpr) -> SMTExpr:
        """Apply ``op_name`` (see ``SUPPORTED_OPS``) to ``args``."""

    @abstractmethod
    def true(self) -> SMTExpr: ...

    @abstractmethod
    def false(self) -> SMTExpr: ...

    @abstractmethod
    def conjunction(self, *args: SMTExpr) -> SMTExpr:
        """Logical AND of ``args``. Returns ``true()`` if empty."""

    @abstractmethod
    def negation(self, expr: SMTExpr) -> SMTExpr:
        """Logical NOT of ``expr``."""

    @abstractmethod
    def is_satisfiable(self, formula: SMTExpr, timeout_ms: int = 5000) -> SatResult:
        """Sound: never returns ``sat``/``unsat`` incorrectly. May return ``unknown``."""

    @abstractmethod
    def is_axis_aligned(self, formula: SMTExpr) -> bool:
        """True iff ``formula`` is a (possibly trivial) conjunction of clauses
        each mentioning at most one variable."""

    @abstractmethod
    def project_to_variable(self, formula: SMTExpr, var: str) -> tuple[int, int] | None:
        """If ``formula`` projected onto ``var`` is a closed interval ``[lo, hi]``,
        return it; otherwise return ``None``.

        Returning ``None`` means "I cannot extract a fully-closed interval
        from the formula alone." The caller should fall back to the
        variable's distribution support (or use :meth:`extract_var_bound`
        and combine with support bounds explicitly).
        """

    @abstractmethod
    def extract_var_bound(
        self, clause: SMTExpr, var: str
    ) -> tuple[int | None, int | None] | None:
        """If ``clause`` is structurally a simple bound on ``var``
        (``var op const`` or ``const op var`` for ``op`` in ``<``, ``<=``,
        ``>``, ``>=``, ``==``), return ``(lo, hi)`` where each side may
        be ``None`` (unbounded that side). For ``==``, returns
        ``(c, c)``. Returns ``None`` if the clause is too complex (e.g.
        arithmetic on ``var``, mixed variables, or ``var != const``).
        """

    @abstractmethod
    def top_level_conjuncts(self, formula: SMTExpr) -> list[SMTExpr]:
        """Return the top-level conjuncts of ``formula``. If ``formula`` is
        not an AND, returns ``[formula]`` (or ``[]`` for trivial ``True``).
        """

    @abstractmethod
    def free_vars(self, formula: SMTExpr) -> frozenset[str]:
        """Return the set of variable names appearing in ``formula``."""

    @abstractmethod
    def evaluate(self, formula: SMTExpr, assignment: dict[str, int]) -> bool:
        """Evaluate ``formula`` on a concrete assignment. Raises if the
        result is not a Boolean constant (e.g. variables are missing)."""

    @abstractmethod
    def repr_expr(self, expr: SMTExpr) -> str:
        """Human-readable rendering of ``expr`` for debugging / logs."""


__all__ = [
    "Clause",
    "SMTBackend",
    "SMTExpr",
    "SatResult",
    "ARITH_OPS",
    "COMPARE_OPS",
    "LOGIC_OPS",
    "SUPPORTED_OPS",
]
