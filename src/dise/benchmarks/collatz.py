"""Collatz step-count benchmark.

Program: the Collatz iteration ``n -> n//2 if even else 3n + 1``, count
the number of steps until reaching 1.
Property: ``steps <= k``.
Distribution: ``BoundedGeometric`` over the starting value.

The Collatz trajectory length is famously irregular and not yet known
to terminate for all positive integers; for practical inputs we cap the
iteration at ``max_steps`` and treat non-termination as ``phi = 0``.

CLI::

    python -m benchmarks.collatz --k 30 --N 200 --budget 5000
"""

from __future__ import annotations

from dise.distributions import BoundedGeometric

from ._base import Benchmark, register
from ._common import common_argparser, run_and_print


def collatz_steps(n: int, max_steps: int = 1000) -> int:
    """Steps for ``n`` to reach 1 under the Collatz iteration (-1 on overflow)."""
    if n < 1:
        return -1
    steps = 0
    while n != 1:
        if steps >= max_steps:
            return -1
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3 * n + 1
        steps = steps + 1
    return steps


def _build(p_geom: float, N: int, k: int) -> Benchmark:
    def program(n: int) -> int:
        return collatz_steps(n)

    return Benchmark(
        name=f"collatz_le_{k}_BG(p={p_geom},N={N})",
        description=f"Collatz steps ≤ {k} from n ~ BoundedGeometric(p={p_geom}, N={N})",
        program=program,
        distribution={"n": BoundedGeometric(p=p_geom, N=N)},
        property_fn=lambda steps: 0 <= steps <= k,
        suggested_budget=5000,
        notes=(
            "Irregular control flow; trajectory length is a notoriously "
            "non-smooth function of n. A good stress test for the "
            "refinement heuristic."
        ),
    )


@register
def collatz() -> Benchmark:
    return _build(p_geom=0.05, N=200, k=30)


def main() -> None:
    p = common_argparser(_build(0.05, 200, 30).description)
    p.add_argument("--p", dest="p_geom", type=float, default=0.05)
    p.add_argument("--N", type=int, default=200)
    p.add_argument("--k", type=int, default=30)
    args = p.parse_args()
    bench = _build(args.p_geom, args.N, args.k)
    run_and_print(bench, args)


if __name__ == "__main__":
    main()
