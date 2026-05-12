"""Sparse-trie depth benchmark.

Program: insert two values into a 4-ary trie (mapping integer keys to
their base-4 digit sequences) and return the maximum depth of any
descent. This is a small surrogate for the depth-of-traversal property
that arises in routing / index lookups.

Property: ``max_depth <= k`` for a fixed cap ``k``.

Distribution: two independent ``Uniform`` integers.

CLI::

    python -m benchmarks.sparse_trie_depth --N 63 --k 3 --budget 5000
"""

from __future__ import annotations

from dise.distributions import Uniform

from ._base import Benchmark, register
from ._common import common_argparser, run_and_print


def trie_max_depth(a: int, b: int, max_depth_cap: int = 16) -> int:
    """Depth of the deepest descent when inserting ``a`` then ``b`` into a 4-ary trie.

    Both values are decomposed into base-4 digits; we walk the trie
    until either a new branch is created or a shared prefix ends. The
    deepest descent depth across the two insertions is returned. We
    cap the loop at ``max_depth_cap`` to keep concolic execution bounded.
    """
    # Determine the digits of `a` and `b` from least- to most-significant.
    if a < 0 or b < 0:
        return 0
    da = []
    v = a
    while v != 0:
        da.append(v % 4)
        v = v // 4
        if len(da) >= max_depth_cap:
            break
    db = []
    v = b
    while v != 0:
        db.append(v % 4)
        v = v // 4
        if len(db) >= max_depth_cap:
            break
    # First insertion: descend from root (depth 0) following digits of `a`.
    # Each step extends the trie by 1, so depth_a = len(da).
    depth_a = len(da)
    # Second insertion: shared prefix length L with `a`, plus one for
    # the divergence step (capped by len(db) + 1).
    L = 0
    while L < len(da) and L < len(db) and da[L] == db[L]:
        L = L + 1
    if L == len(db):
        depth_b = L  # b is a prefix of a (or equal); ends at depth L
    else:
        depth_b = L + 1
    if depth_a >= depth_b:
        return depth_a
    return depth_b


def _build(N: int, k: int) -> Benchmark:
    return Benchmark(
        name=f"sparse_trie_depth_le_{k}_U(0,{N})",
        description=f"4-ary trie max-depth ≤ {k} on (a, b) ~ Uniform(0, {N})^2",
        program=lambda a, b: trie_max_depth(a, b),
        distribution={
            "a": Uniform(lo=0, hi=N),
            "b": Uniform(lo=0, hi=N),
        },
        property_fn=lambda depth: depth <= k,
        suggested_budget=5000,
        notes=(
            "Surrogate for routing-depth properties. Branches on each "
            "base-4 digit; the property correlates with how long the "
            "two values share a common prefix."
        ),
    )


@register
def sparse_trie_depth() -> Benchmark:
    return _build(N=63, k=3)


def main() -> None:
    p = common_argparser(_build(63, 3).description)
    p.add_argument("--N", type=int, default=63)
    p.add_argument("--k", type=int, default=3)
    args = p.parse_args()
    bench = _build(args.N, args.k)
    run_and_print(bench, args)


if __name__ == "__main__":
    main()
