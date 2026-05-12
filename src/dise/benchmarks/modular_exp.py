"""Modular exponentiation benchmark.

Program: ``modpow(a, b, m) = a**b mod m`` via repeated squaring.
Property: the result fits in ``w`` bits (i.e. ``result < 2**w``).
Distribution: uniform integers.

CLI::

    python -m benchmarks.modular_exp --w 4 --budget 5000
"""

from __future__ import annotations

from dise.distributions import Uniform

from ._base import Benchmark, register
from ._common import common_argparser, run_and_print


def modpow(a: int, b: int, m: int) -> int:
    """Repeated-squaring modular exponentiation."""
    result = 1
    base = a % m
    exp = b
    while exp != 0:
        if exp % 2 == 1:
            result = (result * base) % m
        base = (base * base) % m
        exp = exp // 2
    return result


def _build(a_hi: int, b_hi: int, m: int, w_bits: int) -> Benchmark:
    threshold = 1 << w_bits

    def program(a: int, b: int) -> int:
        return modpow(a, b, m)

    return Benchmark(
        name=f"modpow_fits_in_{w_bits}b_m={m}",
        description=f"modpow(a, b, m={m}) fits in {w_bits} bits",
        program=program,
        distribution={
            "a": Uniform(lo=1, hi=a_hi),
            "b": Uniform(lo=0, hi=b_hi),
        },
        property_fn=lambda result: result < threshold,
        suggested_budget=5000,
        notes=(
            "Repeated-squaring modular exponentiation. Branches on the bits "
            "of `b`; mu depends on how often a^b mod m falls below 2^w."
        ),
    )


@register
def modular_exp() -> Benchmark:
    return _build(a_hi=15, b_hi=15, m=37, w_bits=4)


def main() -> None:
    p = common_argparser(_build(15, 15, 37, 4).description)
    p.add_argument("--w", dest="w_bits", type=int, default=4)
    p.add_argument("--a-hi", type=int, default=15)
    p.add_argument("--b-hi", type=int, default=15)
    p.add_argument("--m", type=int, default=37)
    args = p.parse_args()
    bench = _build(args.a_hi, args.b_hi, args.m, args.w_bits)
    run_and_print(bench, args)


if __name__ == "__main__":
    main()
