"""Miller-Rabin primality benchmark.

Program: Miller-Rabin primality test with a single fixed witness ``a``.
Property: the test returns "probably prime" for the input ``n``.
Distribution: ``BoundedGeometric`` over ``n``.

CLI::

    python -m benchmarks.miller_rabin --witness 2 --budget 5000
"""

from __future__ import annotations

from dise.distributions import BoundedGeometric

from ._base import Benchmark, register
from ._common import common_argparser, run_and_print


def miller_rabin_single(n: int, a: int) -> int:
    """Return 1 if ``a`` witnesses ``n`` as 'probably prime', else 0.

    Trivial edges: n < 2 → 0; n == 2 → 1; even n > 2 → 0.
    """
    if n < 2:
        return 0
    if n == 2:
        return 1
    if n % 2 == 0:
        return 0
    d = n - 1
    s = 0
    while d % 2 == 0:
        d = d // 2
        s = s + 1
    x = 1
    base = a % n
    e = d
    while e != 0:
        if e % 2 == 1:
            x = (x * base) % n
        base = (base * base) % n
        e = e // 2
    if x == 1 or x == n - 1:
        return 1
    i = 0
    while i < s - 1:
        x = (x * x) % n
        if x == n - 1:
            return 1
        i = i + 1
    return 0


def _build(witness: int, p_geom: float, N: int) -> Benchmark:
    def program(n: int) -> int:
        return miller_rabin_single(n, witness)

    return Benchmark(
        name=f"miller_rabin_w={witness}_BG(p={p_geom},N={N})",
        description=f"Miller-Rabin (witness={witness}) accepts n",
        program=program,
        distribution={"n": BoundedGeometric(p=p_geom, N=N)},
        property_fn=lambda result: result == 1,
        suggested_budget=5000,
        notes=(
            "Probability that a single Miller-Rabin witness `a` deems `n` "
            "probably prime, with `n` drawn from a bounded geometric."
        ),
    )


@register
def miller_rabin() -> Benchmark:
    return _build(witness=2, p_geom=0.05, N=200)


def main() -> None:
    p = common_argparser(_build(2, 0.05, 200).description)
    p.add_argument("--witness", type=int, default=2)
    p.add_argument("--p-geom", type=float, default=0.05)
    p.add_argument("--N", type=int, default=200)
    args = p.parse_args()
    bench = _build(args.witness, args.p_geom, args.N)
    run_and_print(bench, args)


if __name__ == "__main__":
    main()
