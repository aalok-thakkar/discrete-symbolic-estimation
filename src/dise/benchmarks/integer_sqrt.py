"""Integer square root benchmark.

Program: Newton-style integer square root ``isqrt(n) = floor(sqrt(n))``.
Property: ``isqrt(n)**2 <= n < (isqrt(n) + 1)**2`` — the integer-sqrt
specification. Should be 1 (correct) for every input.

Distribution: uniform over ``[1, N]``.

This is a *correctness* benchmark — DiSE should certify
``mu = 1`` for an axis-aligned-friendly arithmetic kernel.

CLI::

    python -m benchmarks.integer_sqrt --N 1023 --budget 5000
"""

from __future__ import annotations

from dise.distributions import Uniform

from ._base import Benchmark, register
from ._common import common_argparser, run_and_print


def isqrt(n: int) -> int:
    """Floor of sqrt(n) for n >= 0, via Newton iteration on integers."""
    if n < 2:
        return n
    # Initial estimate.
    x = n
    y = (x + 1) // 2
    while y < x:
        x = y
        y = (x + n // x) // 2
    return x


def isqrt_correct(n: int) -> int:
    """1 iff isqrt(n) satisfies the integer-sqrt specification."""
    r = isqrt(n)
    if r < 0:
        return 0
    if r * r > n:
        return 0
    if (r + 1) * (r + 1) <= n:
        return 0
    return 1


def _build(N: int) -> Benchmark:
    return Benchmark(
        name=f"integer_sqrt_correct_U(1,{N})",
        description=f"isqrt specification holds on n ~ Uniform(1, {N})",
        program=lambda n: isqrt_correct(n),
        distribution={"n": Uniform(lo=1, hi=N)},
        property_fn=lambda out: out == 1,
        suggested_budget=2000,
        closed_form_mu=1.0,
        notes=(
            "Correctness benchmark: the specification is universally "
            "true, so DiSE should drive mu_hat to exactly 1.0 and the "
            "interval to [1, 1]."
        ),
    )


@register
def integer_sqrt() -> Benchmark:
    return _build(N=1023)


def main() -> None:
    p = common_argparser(_build(1023).description)
    p.add_argument("--N", type=int, default=1023)
    args = p.parse_args()
    bench = _build(args.N)
    run_and_print(bench, args)


if __name__ == "__main__":
    main()
