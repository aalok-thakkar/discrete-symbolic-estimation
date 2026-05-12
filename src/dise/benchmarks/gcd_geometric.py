"""GCD step-count benchmark.

Program: Euclidean GCD with a step counter.
Property: ``steps <= k`` (the GCD terminates within ``k`` mod-iterations).
Distribution: ``BoundedGeometric(p, N)`` per input.

CLI::

    python -m benchmarks.gcd_geometric --p 0.1 --k 5 --N 100 --budget 5000
"""

from __future__ import annotations

from dise.distributions import BoundedGeometric

from ._base import Benchmark, register
from ._common import common_argparser, run_and_print


def gcd_steps(a: int, b: int) -> int:
    """Number of mod-iterations the Euclidean GCD performs on ``(a, b)``."""
    steps = 0
    while b != 0:
        a, b = b, a % b
        steps = steps + 1
    return steps


def _build(p: float, N: int, k: int) -> Benchmark:
    return Benchmark(
        name=f"gcd_steps_le_{k}_BG(p={p},N={N})",
        description=f"GCD steps ≤ {k}, inputs BoundedGeometric(p={p}, N={N})",
        program=gcd_steps,
        distribution={
            "a": BoundedGeometric(p=p, N=N),
            "b": BoundedGeometric(p=p, N=N),
        },
        property_fn=lambda steps: steps <= k,
        suggested_budget=5000,
        notes=(
            "Headline running example from the brief. With (p=0.1, N=100), "
            "mu_MC ≈ 0.989 at k=5 and ≈ 1.0 at k=10."
        ),
    )


# Canonical (default-parameter) instance for the registry.
@register
def gcd_geometric() -> Benchmark:
    return _build(p=0.1, N=100, k=5)


def main() -> None:
    p = common_argparser(_build(0.1, 100, 5).description)
    p.add_argument("--p", dest="p_geom", type=float, default=0.1)
    p.add_argument("--N", dest="N", type=int, default=100)
    p.add_argument("--k", dest="k_threshold", type=int, default=5)
    args = p.parse_args()
    bench = _build(args.p_geom, args.N, args.k_threshold)
    run_and_print(bench, args)


if __name__ == "__main__":
    main()
