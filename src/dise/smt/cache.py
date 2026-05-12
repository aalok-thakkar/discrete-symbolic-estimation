"""Caching wrapper around an :class:`SMTBackend`.

Memoizes deterministic queries (``is_satisfiable``, ``is_axis_aligned``,
``project_to_variable``, ``extract_var_bound``, ``free_vars``,
``top_level_conjuncts``, ``evaluate``) keyed by the canonical
``repr_expr`` of each input expression. This is a measurable speed-up
on workloads with repeated formulas — typical for the ASIP scheduler,
which queries the same path conditions many times during closure and
refinement.

Cache statistics (``hits``, ``misses``) are exposed for diagnostic
reporting; the dictionaries are bounded by ``max_entries`` (default
50 000) with FIFO eviction.

This wrapper is *purely additive*: writes (``make_int_var``, ``const``,
``op``, ``true``, ``false``, ``conjunction``, ``negation``) and
non-cacheable operations pass straight through to the inner backend.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from .base import SatResult, SMTBackend, SMTExpr


@dataclass
class CacheStats:
    """Diagnostic counters for the :class:`CachedBackend`."""

    is_satisfiable_hits: int = 0
    is_satisfiable_misses: int = 0
    is_axis_aligned_hits: int = 0
    is_axis_aligned_misses: int = 0
    evaluate_hits: int = 0
    evaluate_misses: int = 0
    other_hits: int = 0
    other_misses: int = 0

    @property
    def total_hits(self) -> int:
        return (
            self.is_satisfiable_hits
            + self.is_axis_aligned_hits
            + self.evaluate_hits
            + self.other_hits
        )

    @property
    def total_misses(self) -> int:
        return (
            self.is_satisfiable_misses
            + self.is_axis_aligned_misses
            + self.evaluate_misses
            + self.other_misses
        )

    @property
    def hit_rate(self) -> float:
        total = self.total_hits + self.total_misses
        return 0.0 if total == 0 else self.total_hits / total


class CachedBackend(SMTBackend):
    """Memoizing facade over another :class:`SMTBackend`."""

    def __init__(self, inner: SMTBackend, max_entries: int = 50_000) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._inner = inner
        self._max_entries = max_entries
        self.stats = CacheStats()
        self._cache_sat: OrderedDict[str, SatResult] = OrderedDict()
        self._cache_aa: OrderedDict[str, bool] = OrderedDict()
        self._cache_proj: OrderedDict[tuple[str, str], tuple[int, int] | None] = OrderedDict()
        self._cache_extract: OrderedDict[tuple[str, str], tuple[int | None, int | None] | None] = OrderedDict()
        self._cache_free: OrderedDict[str, frozenset[str]] = OrderedDict()
        self._cache_eval: OrderedDict[tuple[str, tuple], bool] = OrderedDict()

    @property
    def inner(self) -> SMTBackend:
        return self._inner

    def _bound(self, cache: OrderedDict) -> None:
        while len(cache) > self._max_entries:
            cache.popitem(last=False)

    # ----- Write/pass-through methods -----

    def make_int_var(self, name: str) -> SMTExpr:
        return self._inner.make_int_var(name)

    def const(self, value: int) -> SMTExpr:
        return self._inner.const(value)

    def op(self, op_name: str, *args: SMTExpr) -> SMTExpr:
        return self._inner.op(op_name, *args)

    def true(self) -> SMTExpr:
        return self._inner.true()

    def false(self) -> SMTExpr:
        return self._inner.false()

    def conjunction(self, *args: SMTExpr) -> SMTExpr:
        return self._inner.conjunction(*args)

    def negation(self, expr: SMTExpr) -> SMTExpr:
        return self._inner.negation(expr)

    def repr_expr(self, expr: SMTExpr) -> str:
        return self._inner.repr_expr(expr)

    # ----- Cached query methods -----

    def is_satisfiable(self, formula: SMTExpr, timeout_ms: int = 5000) -> SatResult:
        key = self._inner.repr_expr(formula)
        if key in self._cache_sat:
            self._cache_sat.move_to_end(key)
            self.stats.is_satisfiable_hits += 1
            return self._cache_sat[key]
        self.stats.is_satisfiable_misses += 1
        result = self._inner.is_satisfiable(formula, timeout_ms=timeout_ms)
        self._cache_sat[key] = result
        self._bound(self._cache_sat)
        return result

    def is_axis_aligned(self, formula: SMTExpr) -> bool:
        key = self._inner.repr_expr(formula)
        if key in self._cache_aa:
            self._cache_aa.move_to_end(key)
            self.stats.is_axis_aligned_hits += 1
            return self._cache_aa[key]
        self.stats.is_axis_aligned_misses += 1
        result = self._inner.is_axis_aligned(formula)
        self._cache_aa[key] = result
        self._bound(self._cache_aa)
        return result

    def project_to_variable(self, formula: SMTExpr, var: str) -> tuple[int, int] | None:
        key = (self._inner.repr_expr(formula), var)
        if key in self._cache_proj:
            self._cache_proj.move_to_end(key)
            self.stats.other_hits += 1
            return self._cache_proj[key]
        self.stats.other_misses += 1
        result = self._inner.project_to_variable(formula, var)
        self._cache_proj[key] = result
        self._bound(self._cache_proj)
        return result

    def extract_var_bound(
        self, clause: SMTExpr, var: str
    ) -> tuple[int | None, int | None] | None:
        key = (self._inner.repr_expr(clause), var)
        if key in self._cache_extract:
            self._cache_extract.move_to_end(key)
            self.stats.other_hits += 1
            return self._cache_extract[key]
        self.stats.other_misses += 1
        result = self._inner.extract_var_bound(clause, var)
        self._cache_extract[key] = result
        self._bound(self._cache_extract)
        return result

    def top_level_conjuncts(self, formula: SMTExpr) -> list[SMTExpr]:
        # Not cached: returned values reference live expressions.
        return self._inner.top_level_conjuncts(formula)

    def free_vars(self, formula: SMTExpr) -> frozenset[str]:
        key = self._inner.repr_expr(formula)
        if key in self._cache_free:
            self._cache_free.move_to_end(key)
            self.stats.other_hits += 1
            return self._cache_free[key]
        self.stats.other_misses += 1
        result = self._inner.free_vars(formula)
        self._cache_free[key] = result
        self._bound(self._cache_free)
        return result

    def evaluate(self, formula: SMTExpr, assignment: dict[str, int]) -> bool:
        key = (
            self._inner.repr_expr(formula),
            tuple(sorted(assignment.items())),
        )
        if key in self._cache_eval:
            self._cache_eval.move_to_end(key)
            self.stats.evaluate_hits += 1
            return self._cache_eval[key]
        self.stats.evaluate_misses += 1
        result = self._inner.evaluate(formula, assignment)
        self._cache_eval[key] = result
        self._bound(self._cache_eval)
        return result


__all__ = ["CachedBackend", "CacheStats"]
