"""Bitvector-kernel benchmarks (popcount, parity, log2).

For each kernel, the property checks the result lies in its valid
output range. Distribution: uniform over ``{0, ..., 2**w - 1}``.

CLI::

    python -m benchmarks.bitvector_kernels --kernel popcount --w 8 --budget 5000
"""

from __future__ import annotations

from dise.distributions import Uniform

from ._base import Benchmark, register
from ._common import common_argparser, run_and_print


def popcount(x: int) -> int:
    """Number of 1-bits in ``x``."""
    count = 0
    v = x
    while v != 0:
        count = count + (v % 2)
        v = v // 2
    return count


def parity(x: int) -> int:
    """1 iff the number of 1-bits in ``x`` is odd."""
    p = 0
    v = x
    while v != 0:
        p = p + (v % 2)
        v = v // 2
    return p % 2


def floor_log2(x: int) -> int:
    """Floor of log2(x) for x >= 1; -1 for x = 0."""
    if x == 0:
        return -1
    out = 0
    v = x
    while v > 1:
        v = v // 2
        out = out + 1
    return out


KERNELS = {
    "popcount": popcount,
    "parity": parity,
    "log2": floor_log2,
}


def _build(kernel_name: str, w: int) -> Benchmark:
    program = KERNELS[kernel_name]
    if kernel_name == "popcount":
        prop = lambda out: out >= 0  # noqa: E731
        descr = f"popcount(x) ≥ 0 for x in [0, 2^{w})"
    elif kernel_name == "parity":
        prop = lambda out: out == 0 or out == 1  # noqa: E731
        descr = f"parity(x) ∈ {{0, 1}} for x in [0, 2^{w})"
    elif kernel_name == "log2":
        prop = lambda out: out >= -1  # noqa: E731
        descr = f"log2(x) ≥ -1 for x in [0, 2^{w})"
    else:
        raise ValueError(kernel_name)
    return Benchmark(
        name=f"{kernel_name}_w{w}",
        description=descr,
        program=program,
        distribution={"x": Uniform(lo=0, hi=(1 << w) - 1)},
        property_fn=prop,
        suggested_budget=2000,
        closed_form_mu=1.0,
        notes=(
            "Property is always True by construction; tests whether the "
            "frontier collapses cleanly to mu_hat = 1."
        ),
    )


@register
def bitvector_popcount() -> Benchmark:
    return _build("popcount", w=6)


@register
def bitvector_parity() -> Benchmark:
    return _build("parity", w=6)


@register
def bitvector_log2() -> Benchmark:
    return _build("log2", w=6)


def main() -> None:
    p = common_argparser("Bitvector-kernel benchmarks under uniform inputs")
    p.add_argument(
        "--kernel", choices=list(KERNELS.keys()), default="popcount"
    )
    p.add_argument("--w", type=int, default=6, help="bitwidth")
    args = p.parse_args()
    bench = _build(args.kernel, args.w)
    run_and_print(bench, args)


if __name__ == "__main__":
    main()
