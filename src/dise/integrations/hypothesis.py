"""DiSE ↔ Hypothesis bridge: distribution-aware property-based testing.

`hypothesis <https://hypothesis.readthedocs.io>`_ is the Python community's
de-facto property-based-testing (PBT) library. Hypothesis strategies are
edge-case-biased generators designed to *find counterexamples fast*.
DiSE, by contrast, certifies the **operational** failure rate of a
program under an **operational** distribution.

This module is the bridge: it converts a Hypothesis ``SearchStrategy``
into a DiSE :class:`~dise.distributions.Distribution`, then runs
:func:`dise.estimate` against an arbitrary property. The result is a
*certified* answer to questions like:

    "Under my actual workload, what is the probability my code raises
    ``AssertionError`` — with provable confidence ``1 - delta``?"

— a different question from PBT's "does there exist any input that
fails?".

Quickstart::

    import hypothesis.strategies as st
    from dise.integrations.hypothesis import estimate_from_strategies

    result = estimate_from_strategies(
        strategies={"a": st.integers(min_value=1, max_value=31),
                    "b": st.integers(min_value=1, max_value=31)},
        property_fn=lambda a, b: a * b < (1 << 8),  # 8-bit overflow check
        epsilon=0.05, delta=0.05,
    )
    print(result.mu_hat, result.interval)

Tier 1 (analytic mass):
    * ``st.integers(min_value=L, max_value=H)`` → :class:`Uniform` ``(L, H)``.
    * ``st.sampled_from([v1, ..., vn])`` (integer values) →
      :class:`Categorical` over indices, with the program receiving the
      mapped value.

Tier 2 (sample-based; future work): arbitrary strategies via the
``hypothesis.strategies`` internal sampling machinery, with mass
estimated by importance sampling. Not yet implemented; the adapter
raises a helpful ``NotImplementedError`` for unsupported strategies.

Soft dependency: this module imports ``hypothesis`` lazily so the
absence of hypothesis at install time does not prevent ``dise`` from
loading.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from ..distributions import Distribution, Uniform
from ..estimator.api import EstimationResult, estimate
from ..smt import SMTBackend

if TYPE_CHECKING:  # pragma: no cover
    from hypothesis.strategies import SearchStrategy  # noqa: F401


def _require_hypothesis() -> Any:
    """Soft-import: raise a helpful error if `hypothesis` is missing."""
    try:
        import hypothesis
    except ImportError as e:
        raise ImportError(
            "dise.integrations.hypothesis requires the `hypothesis` package. "
            "Install it via `pip install hypothesis` or `uv pip install hypothesis`."
        ) from e
    return hypothesis


def _strategy_type_name(strategy: Any) -> str:
    return type(strategy).__name__


def _unwrap_lazy(strategy: Any) -> Any:
    """Hypothesis often wraps strategy constructors in ``LazyStrategy``.
    Walk through ``wrapped_strategy`` (and ``.strategy`` for older
    versions) until we reach a concrete strategy class.

    Uses ``is None`` checks (rather than truthiness) because
    ``bool(<SearchStrategy>)`` emits a Hypothesis warning.
    """
    seen: set[int] = set()
    while id(strategy) not in seen:
        seen.add(id(strategy))
        name = _strategy_type_name(strategy)
        if name != "LazyStrategy":
            break
        inner = getattr(strategy, "wrapped_strategy", None)
        if inner is None:
            inner = getattr(strategy, "strategy", None)
        if inner is None:
            break
        strategy = inner
    return strategy


def from_integers(low: int, high: int) -> Uniform:
    """Explicit constructor: a DiSE :class:`Uniform` matching
    ``st.integers(min_value=low, max_value=high)``."""
    if high < low:
        raise ValueError(f"from_integers: high {high} < low {low}")
    return Uniform(lo=low, hi=high)


def from_sampled_from(values: Sequence[int]) -> Uniform:
    """Explicit constructor for ``st.sampled_from(values)`` when the
    values are *consecutive* integers.

    For non-consecutive integer values, use :func:`auto_from_strategy`
    with a manual remap (Tier 2 — TODO).
    """
    if not values:
        raise ValueError("from_sampled_from: empty values")
    sorted_vs = sorted(int(v) for v in values)
    lo, hi = sorted_vs[0], sorted_vs[-1]
    if sorted_vs == list(range(lo, hi + 1)):
        return Uniform(lo=lo, hi=hi)
    raise NotImplementedError(
        "from_sampled_from currently supports only consecutive integer values "
        f"(got {sorted_vs!r}). Construct dise.distributions.Categorical "
        "manually for arbitrary supports."
    )


def auto_from_strategy(strategy: Any) -> Distribution:
    """Best-effort conversion of a Hypothesis strategy to a DiSE
    :class:`~dise.distributions.Distribution`.

    Tier-1 cases (supported):
      * ``st.integers(min_value=L, max_value=H)`` → :class:`Uniform`.
      * ``st.sampled_from([L, L+1, ..., H])`` over consecutive integers →
        :class:`Uniform`.

    For unsupported strategies, raises ``NotImplementedError`` with a
    pointer to the explicit constructors.
    """
    _require_hypothesis()
    strategy = _unwrap_lazy(strategy)
    name = _strategy_type_name(strategy)

    # st.integers(...) — internal class is hypothesis.strategies._internal.numbers.IntegersStrategy
    if name == "IntegersStrategy":
        start = getattr(strategy, "start", None) or getattr(strategy, "min_value", None)
        end = getattr(strategy, "end", None) or getattr(strategy, "max_value", None)
        if start is None or end is None:
            raise NotImplementedError(
                "st.integers() with unbounded support is not yet handled; pass "
                "explicit min_value and max_value."
            )
        return from_integers(int(start), int(end))

    # st.sampled_from(values) — internal class is SampledFromStrategy
    if name == "SampledFromStrategy":
        elements = getattr(strategy, "elements", None)
        if elements is None:
            raise NotImplementedError(
                "auto_from_strategy: cannot introspect SampledFromStrategy "
                "(missing `.elements`). Use from_sampled_from(values) explicitly."
            )
        return from_sampled_from(list(elements))

    # Tier 2 placeholders
    raise NotImplementedError(
        f"auto_from_strategy: strategy {name!r} is not yet supported. Supported "
        f"in Tier 1: st.integers(min_value, max_value), "
        f"st.sampled_from(<consecutive ints>). For other strategies, construct "
        f"a `dise.distributions.Distribution` manually and call `dise.estimate` "
        f"directly."
    )


def estimate_from_strategy(
    strategy: Any,
    property_fn: Callable[[int], bool],
    *,
    program: Callable[[int], Any] | None = None,
    epsilon: float = 0.05,
    delta: float = 0.05,
    budget: int | None = None,
    seed: int = 0,
    backend: SMTBackend | None = None,
    **estimate_kwargs: Any,
) -> EstimationResult:
    """Single-strategy convenience entry point.

    Equivalent to :func:`dise.estimate` with one input variable named
    ``"x"`` whose distribution is :func:`auto_from_strategy` (``strategy``).
    If ``program`` is ``None`` (default), the strategy's output is fed
    directly to ``property_fn``; otherwise ``program(x)`` is computed
    first.

    Examples
    --------

    >>> import hypothesis.strategies as st
    >>> from dise.integrations.hypothesis import estimate_from_strategy
    >>> result = estimate_from_strategy(
    ...     st.integers(min_value=1, max_value=100),
    ...     property_fn=lambda x: x > 50,
    ...     budget=2000,
    ... )                                                    # doctest: +SKIP
    >>> result.mu_hat                                         # doctest: +SKIP
    0.495...
    """
    dist = auto_from_strategy(strategy)
    actual_program = program if program is not None else (lambda x: x)
    return estimate(
        program=actual_program,
        distribution={"x": dist},
        property_fn=property_fn,
        epsilon=epsilon,
        delta=delta,
        budget=budget,
        seed=seed,
        backend=backend,
        **estimate_kwargs,
    )


def estimate_from_strategies(
    strategies: Mapping[str, Any],
    property_fn: Callable[..., bool],
    *,
    program: Callable[..., Any] | None = None,
    epsilon: float = 0.05,
    delta: float = 0.05,
    budget: int | None = None,
    seed: int = 0,
    backend: SMTBackend | None = None,
    **estimate_kwargs: Any,
) -> EstimationResult:
    """Multi-strategy entry point.

    ``strategies`` maps variable names to Hypothesis strategies; each is
    converted via :func:`auto_from_strategy`. The resulting joint is a
    product distribution. The default ``program`` returns its kwargs
    dict; ``property_fn(**kwargs)`` is called on the program's output.

    Examples
    --------

    >>> import hypothesis.strategies as st
    >>> result = estimate_from_strategies(
    ...     strategies={
    ...         "a": st.integers(min_value=1, max_value=31),
    ...         "b": st.integers(min_value=1, max_value=31),
    ...     },
    ...     property_fn=lambda a, b: a * b < (1 << 8),
    ...     budget=2000,
    ... )                                                    # doctest: +SKIP
    """
    distribution = {name: auto_from_strategy(s) for name, s in strategies.items()}
    if program is None:
        def actual_program(**kw: int) -> dict[str, int]:
            return kw
        def wrapped_property(out: dict[str, int]) -> bool:
            return bool(property_fn(**out))
        return estimate(
            program=actual_program,
            distribution=distribution,
            property_fn=wrapped_property,
            epsilon=epsilon,
            delta=delta,
            budget=budget,
            seed=seed,
            backend=backend,
            **estimate_kwargs,
        )
    return estimate(
        program=program,
        distribution=distribution,
        property_fn=property_fn,
        epsilon=epsilon,
        delta=delta,
        budget=budget,
        seed=seed,
        backend=backend,
        **estimate_kwargs,
    )


__all__ = [
    "auto_from_strategy",
    "estimate_from_strategy",
    "estimate_from_strategies",
    "from_integers",
    "from_sampled_from",
]
