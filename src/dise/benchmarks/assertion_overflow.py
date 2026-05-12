"""Assertion-violation benchmark: integer-multiplication overflow.

The canonical formal-verification framing: given a program with an
``assert``, estimate the probability that the assertion is violated
under an operational distribution over inputs.

Program (``safe_mul``):

.. code-block:: python

    def safe_mul(a: int, b: int, w: int) -> int:
        s = a * b
        assert s < (1 << w), "overflow"
        return s

The property is *assertion violation*; the estimand is
:math:`\\mu = \\Pr_D[\\text{assertion fires}]` — the program's
failure probability with respect to its overflow invariant. DiSE
certifies a two-sided interval on :math:`\\mu` at confidence
:math:`1 - \\delta`.

This is :func:`dise.failure_probability` applied to a tiny but
concrete arithmetic kernel — a clean exemplar of the assert-style
reliability question that probabilistic-verification artifacts in the
ATVA / SAS / NFM communities target.

CLI::

    python -m dise.benchmarks.assertion_overflow --M 31 --w 8 --budget 5000

Or via the top-level CLI::

    dise run "assertion_overflow_mul_w=8_U(1,31)" --budget 5000
"""

from __future__ import annotations

from dise.distributions import Uniform

from ._base import Benchmark, register
from ._common import common_argparser, run_and_print


def safe_mul(a: int, b: int, w: int) -> int:
    """Multiply with an overflow assertion: ``a * b < 2**w``."""
    s = a * b
    assert s < (1 << w), f"overflow: {a} * {b} = {s} >= 2^{w}"
    return s


def _build(M: int, w: int) -> Benchmark:
    """Build a benchmark for ``safe_mul`` with the given parameters.

    The wrapped ``program`` returns 1 iff the overflow assertion fires
    and 0 otherwise; the property checks for the failure marker. This
    is exactly what :func:`dise.failure_probability` does internally —
    we inline it here so the benchmark is a plain ``Benchmark``
    (uniform interface with the rest of the suite).
    """

    def program(a: int, b: int) -> int:
        try:
            safe_mul(a, b, w=w)
            return 0  # assertion held
        except AssertionError:
            return 1  # assertion violated

    return Benchmark(
        name=f"assertion_overflow_mul_w={w}_U(1,{M})",
        description=(
            f"Pr[a*b overflows {w}-bit unsigned] with a, b ~ Uniform(1, {M})"
        ),
        program=program,
        distribution={
            "a": Uniform(lo=1, hi=M),
            "b": Uniform(lo=1, hi=M),
        },
        # property = "assertion was violated"
        property_fn=lambda v: v == 1,
        suggested_budget=5000,
        notes=(
            "Canonical assertion-violation benchmark. The estimand is "
            "the failure probability — DiSE certifies a two-sided "
            "interval on the assertion-violation rate."
        ),
        metadata={"M": M, "w": w, "kind": "assertion_violation"},
    )


@register
def assertion_overflow_mul() -> Benchmark:
    return _build(M=31, w=8)


def main() -> None:
    p = common_argparser(_build(31, 8).description)
    p.add_argument("--M", type=int, default=31, help="upper bound on a, b")
    p.add_argument("--w", type=int, default=8, help="bit-width for the overflow check")
    args = p.parse_args()
    run_and_print(_build(args.M, args.w), args)


if __name__ == "__main__":
    main()
