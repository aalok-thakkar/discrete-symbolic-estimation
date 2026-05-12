"""Concrete region types: Empty, Unconstrained, AxisAlignedBox, GeneralRegion.

Plus :func:`build_region`, the dispatcher that turns an arbitrary SMT
formula into the most-specific region for it.
"""

from __future__ import annotations

import numpy as np

from ..distributions import ProductDistribution
from ..smt import SMTBackend, SMTExpr
from ._base import Region, SampleBatch

# -----------------------------------------------------------------------------
# EmptyRegion
# -----------------------------------------------------------------------------


class EmptyRegion(Region):
    """A region SMT-proved to be empty. Mass is identically 0."""

    def __init__(self, formula: SMTExpr) -> None:
        self._formula = formula

    @property
    def formula(self) -> SMTExpr:
        return self._formula

    @property
    def is_axis_aligned(self) -> bool:
        return True

    def mass(self, distribution, smt, rng, n_mc=1000):
        return (0.0, 0.0)

    def sample(self, distribution, smt, rng, n):
        return SampleBatch(
            inputs={v: np.empty(0, dtype=np.int64) for v in distribution.variables},
            n=0,
            rejection_ratio=0.0,
        )

    def contains(self, x: dict[str, int]) -> bool:
        return False

    def __repr__(self) -> str:
        return "EmptyRegion()"


# -----------------------------------------------------------------------------
# UnconstrainedRegion
# -----------------------------------------------------------------------------


class UnconstrainedRegion(Region):
    """The full input space under ``D``. Mass is 1; sampling delegates to ``D``."""

    def __init__(self, formula: SMTExpr) -> None:
        self._formula = formula

    @property
    def formula(self) -> SMTExpr:
        return self._formula

    @property
    def is_axis_aligned(self) -> bool:
        return True

    def mass(self, distribution, smt, rng, n_mc=1000):
        return (1.0, 0.0)

    def sample(self, distribution, smt, rng, n):
        inputs = distribution.sample(rng, n)
        return SampleBatch(inputs=inputs, n=n, rejection_ratio=None)

    def contains(self, x: dict[str, int]) -> bool:
        return True

    def __repr__(self) -> str:
        return "UnconstrainedRegion()"


# -----------------------------------------------------------------------------
# AxisAlignedBox
# -----------------------------------------------------------------------------


class AxisAlignedBox(Region):
    """Conjunction of per-variable closed intervals.

    Mass is closed-form (product of marginal masses) with variance 0.
    Sampling uses each factor's ``sample_truncated``. Instances should
    be treated as immutable.
    """

    def __init__(
        self,
        bounds: dict[str, tuple[int, int]],
        formula: SMTExpr,
    ) -> None:
        self._bounds: dict[str, tuple[int, int]] = dict(bounds)
        self._formula = formula

    @property
    def bounds(self) -> dict[str, tuple[int, int]]:
        return self._bounds

    @property
    def formula(self) -> SMTExpr:
        return self._formula

    @property
    def is_axis_aligned(self) -> bool:
        return True

    def mass(self, distribution, smt, rng, n_mc=1000):
        w = 1.0
        for v, (lo, hi) in self._bounds.items():
            if v not in distribution.factors:
                continue
            w *= distribution.factors[v].mass(lo, hi)
        return (w, 0.0)

    def sample(self, distribution, smt, rng, n):
        if n <= 0:
            return SampleBatch(
                inputs={v: np.empty(0, dtype=np.int64) for v in distribution.variables},
                n=0,
                rejection_ratio=None,
            )
        inputs: dict[str, np.ndarray] = {}
        for v in distribution.variables:
            lo, hi = self._bounds[v]
            inputs[v] = distribution.factors[v].sample_truncated(rng, lo, hi, n)
        return SampleBatch(inputs=inputs, n=n, rejection_ratio=None)

    def contains(self, x: dict[str, int]) -> bool:
        for v, (lo, hi) in self._bounds.items():
            if v not in x:
                return False
            if not (lo <= x[v] <= hi):
                return False
        return True

    def __repr__(self) -> str:
        parts = [f"{v}:[{lo},{hi}]" for v, (lo, hi) in self._bounds.items()]
        return f"AxisAlignedBox({', '.join(parts)})"


# -----------------------------------------------------------------------------
# GeneralRegion
# -----------------------------------------------------------------------------


class GeneralRegion(Region):
    """A region defined by a base axis-aligned envelope plus an SMT predicate.

    The base is an over-approximation; the predicate filters samples
    that fall inside it. Mass is estimated by importance sampling from
    the base.
    """

    def __init__(
        self,
        base: AxisAlignedBox,
        formula: SMTExpr,
        smt: SMTBackend,
    ) -> None:
        self._base = base
        self._formula = formula
        self._smt = smt

    @property
    def base(self) -> AxisAlignedBox:
        return self._base

    @property
    def formula(self) -> SMTExpr:
        return self._formula

    @property
    def is_axis_aligned(self) -> bool:
        return False

    def mass(self, distribution, smt, rng, n_mc=1000):
        w_base, _ = self._base.mass(distribution, smt, rng, n_mc)
        if w_base <= 0.0:
            return (0.0, 0.0)
        proposal = self._base.sample(distribution, smt, rng, n_mc)
        hits = 0
        for x in proposal.iter_assignments():
            try:
                if smt.evaluate(self._formula, x):
                    hits += 1
            except (ValueError, KeyError, ZeroDivisionError, ArithmeticError):
                continue
        n = proposal.n
        if n == 0:
            return (0.0, w_base * w_base * 0.25)  # uninformative
        p_hat = hits / n
        w_hat = w_base * p_hat
        # Wilson-smoothed Bernoulli variance plug-in:
        p_tilde = (hits + 1) / (n + 2)
        bern_var = p_tilde * (1.0 - p_tilde)
        # Variance of the sample-mean p_hat:
        var_p = bern_var / n
        w_var = w_base * w_base * var_p
        return (w_hat, w_var)

    def sample(self, distribution, smt, rng, n):
        if n <= 0:
            return SampleBatch(
                inputs={v: np.empty(0, dtype=np.int64) for v in distribution.variables},
                n=0,
                rejection_ratio=None,
            )
        accepted: dict[str, list[int]] = {v: [] for v in distribution.variables}
        attempts = 0
        accepts = 0
        max_attempts = max(1000, 200 * n)
        while accepts < n and attempts < max_attempts:
            need = max(8, n - accepts)
            batch = self._base.sample(distribution, smt, rng, need)
            for x in batch.iter_assignments():
                attempts += 1
                try:
                    ok = smt.evaluate(self._formula, x)
                except (ValueError, KeyError, ZeroDivisionError, ArithmeticError):
                    ok = False
                if ok:
                    for v in distribution.variables:
                        accepted[v].append(x[v])
                    accepts += 1
                    if accepts >= n:
                        break
        inputs = {
            v: np.array(accepted[v], dtype=np.int64) for v in distribution.variables
        }
        rej_ratio = accepts / attempts if attempts > 0 else 0.0
        return SampleBatch(inputs=inputs, n=accepts, rejection_ratio=rej_ratio)

    def contains(self, x: dict[str, int]) -> bool:
        if not self._base.contains(x):
            return False
        try:
            return self._smt.evaluate(self._formula, x)
        except (ValueError, KeyError, ZeroDivisionError, ArithmeticError):
            return False

    def __repr__(self) -> str:
        return f"GeneralRegion(base={self._base}, formula={self._smt.repr_expr(self._formula)})"


# -----------------------------------------------------------------------------
# Build region from formula
# -----------------------------------------------------------------------------


def _compute_box_bounds(
    formula: SMTExpr,
    distribution: ProductDistribution,
    smt: SMTBackend,
) -> dict[str, tuple[int, int]] | None:
    """Compute axis-aligned bounds from ``formula`` intersected with
    distribution support. Returns None if the formula is not reducible
    to a box (some clause is multi-variable, or arithmetic on a variable),
    or an empty dict marker if the bounds are unsatisfiable.
    """
    if not smt.is_axis_aligned(formula):
        return None
    bounds: dict[str, list[int]] = {
        v: list(distribution.factors[v].support_bounds())
        for v in distribution.variables
    }
    for clause in smt.top_level_conjuncts(formula):
        fv = smt.free_vars(clause)
        if len(fv) == 0:
            try:
                if not smt.evaluate(clause, {}):
                    return {}
                continue
            except (ValueError, KeyError):
                return None
        if len(fv) > 1:
            return None
        (v,) = fv
        if v not in bounds:
            return None  # variable not declared in distribution
        extracted = smt.extract_var_bound(clause, v)
        if extracted is None:
            return None
        lo_c, hi_c = extracted
        if lo_c is not None:
            bounds[v][0] = max(bounds[v][0], lo_c)
        if hi_c is not None:
            bounds[v][1] = min(bounds[v][1], hi_c)
        if bounds[v][0] > bounds[v][1]:
            return {}
    return {v: (lo, hi) for v, (lo, hi) in bounds.items()}


def _make_box_formula(
    bounds: dict[str, tuple[int, int]], smt: SMTBackend
) -> SMTExpr:
    parts = []
    for v, (lo, hi) in bounds.items():
        var = smt.make_int_var(v)
        parts.append(smt.op(">=", var, smt.const(lo)))
        parts.append(smt.op("<=", var, smt.const(hi)))
    return smt.conjunction(*parts)


def build_region(
    formula: SMTExpr,
    distribution: ProductDistribution,
    smt: SMTBackend,
) -> Region:
    """Build the most-specific :class:`Region` representing ``formula``."""
    sat = smt.is_satisfiable(formula)
    if sat == "unsat":
        return EmptyRegion(formula)

    box = _compute_box_bounds(formula, distribution, smt)
    if box == {}:
        return EmptyRegion(formula)
    if box is not None:
        # Axis-aligned + reducible to bounds — use closed-form box.
        if any(lo > hi for lo, hi in box.values()):
            return EmptyRegion(formula)
        return AxisAlignedBox(bounds=box, formula=formula)

    # Not reducible: build an axis-aligned envelope as the IS proposal.
    envelope: dict[str, tuple[int, int]] = {}
    for v in distribution.variables:
        envelope[v] = distribution.factors[v].support_bounds()
    # Refine envelope per-variable using projection where possible.
    for v in distribution.variables:
        proj = smt.project_to_variable(formula, v)
        if proj is not None:
            lo, hi = proj
            envelope[v] = (max(envelope[v][0], lo), min(envelope[v][1], hi))
    if any(lo > hi for lo, hi in envelope.values()):
        return EmptyRegion(formula)
    base = AxisAlignedBox(bounds=envelope, formula=_make_box_formula(envelope, smt))
    return GeneralRegion(base=base, formula=formula, smt=smt)


__all__ = [
    "AxisAlignedBox",
    "EmptyRegion",
    "GeneralRegion",
    "UnconstrainedRegion",
    "build_region",
]
