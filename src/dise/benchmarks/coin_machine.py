"""Pedagogical intro benchmark: a tiny branching program with a rare bug.

The program partitions its input into three "obvious" regions and
contains a thin slice of inputs where it misbehaves:

.. code-block:: python

    def coin_machine(x: int) -> int:
        if x < 10:           return 0     # region A: safe
        if x < 100:           return 1    # region B: always-flags
        if x % 1000 == 0:     return 1    # region C-bug: rare bug (mass ≈ 0.1%)
        return 0                          # region C-ok: safe again

Under ``x ~ Uniform(1, 9999)``, the true probability that the program
returns ``1`` is

.. math::

    \\mu \\;=\\; \\frac{|B| + |C\\text{-bug}|}{|\\mathcal{X}|}
    \\;=\\; \\frac{90 + 9}{9999} \\;\\approx\\; 0.0099.

This is the **canonical pedagogical example** for ASIP:

* Three visible regions partition the input space — readers can see
  the partition refinement happen.
* A rare event (``C-bug``) is concealed inside region C; plain MC
  needs :math:`\\Theta(1/\\mu) \\approx 10^4` samples to certify it
  to half-width ``epsilon=0.005``.
* DiSE refines into the four partitions, certifies region A and C-ok
  as :math:`\\mu_\\pi = 0`, region B as :math:`\\mu_\\pi = 1`, and
  zooms in on C-bug — typically in hundreds of concolic runs.

Run::

    python -m dise.benchmarks.coin_machine --N 9999 --budget 5000

Or via the top-level CLI::

    dise run 'coin_machine_U(1,9999)' --budget 5000
"""

from __future__ import annotations

from dise.distributions import Uniform

from ._base import Benchmark, register
from ._common import common_argparser, run_and_print


def coin_machine(x: int) -> int:
    """A three-region branching program with a thin rare-bug slice."""
    if x < 10:
        return 0  # region A: safe
    if x < 100:
        return 1  # region B: always-flags (the noisy middle)
    if x % 1000 == 0:
        return 1  # region C-bug: rare slice triggering the bug
    return 0      # region C-ok: safe


def _build(N: int) -> Benchmark:
    return Benchmark(
        name=f"coin_machine_U(1,{N})",
        description=(
            f"Pedagogical 3-region program with a rare bug; x ~ Uniform(1, {N})"
        ),
        program=coin_machine,
        distribution={"x": Uniform(lo=1, hi=N)},
        property_fn=lambda y: y == 1,
        suggested_budget=2000,
        notes=(
            "Intro benchmark for ASIP. Region structure is obvious from the "
            "source; DiSE should refine into 4 leaves (A, B, C-bug, C-ok), "
            "close each, and certify the true mu to half-width 0 in a few "
            "hundred concolic runs. Plain MC needs Theta(1/mu) samples for "
            "the same half-width."
        ),
        metadata={"N": N, "kind": "branching_toy"},
    )


@register
def coin_machine_intro() -> Benchmark:
    return _build(N=9999)


def main() -> None:
    p = common_argparser(_build(9999).description)
    p.add_argument("--N", type=int, default=9999)
    args = p.parse_args()
    run_and_print(_build(args.N), args)


if __name__ == "__main__":
    main()
