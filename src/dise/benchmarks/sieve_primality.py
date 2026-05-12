"""Trial-division primality benchmark.

Program: trial division up to ``sqrt(n)``; returns 1 iff ``n`` is prime.
Property: ``output == 1``.
Distribution: uniform integer.

A deterministic, branch-heavy primality test — every divisor probed
emits one branch. The frontier has to interleave refinement on the
``i*i <= n`` loop guard and the ``n % i == 0`` test.

CLI::

    python -m benchmarks.sieve_primality --N 200 --budget 5000
"""

from __future__ import annotations

from dise.distributions import Uniform

from ._base import Benchmark, register
from ._common import common_argparser, run_and_print


def trial_division_is_prime(n: int) -> int:
    """Return 1 iff ``n`` is prime; 0 otherwise."""
    if n < 2:
        return 0
    if n == 2:
        return 1
    if n % 2 == 0:
        return 0
    i = 3
    while i * i <= n:
        if n % i == 0:
            return 0
        i = i + 2
    return 1


def _build(N: int) -> Benchmark:
    return Benchmark(
        name=f"sieve_primality_U(2,{N})",
        description=f"trial-division primality on n ~ Uniform(2, {N})",
        program=lambda n: trial_division_is_prime(n),
        distribution={"n": Uniform(lo=2, hi=N)},
        property_fn=lambda out: out == 1,
        suggested_budget=5000,
        notes=(
            "Reliability = fraction of primes in [2, N]; closed-form via "
            "the prime-counting function pi(N) for verification."
        ),
        metadata={"N": N},
    )


@register
def sieve_primality() -> Benchmark:
    return _build(N=200)


def main() -> None:
    p = common_argparser(_build(200).description)
    p.add_argument("--N", type=int, default=200)
    args = p.parse_args()
    bench = _build(args.N)
    run_and_print(bench, args)


if __name__ == "__main__":
    main()
