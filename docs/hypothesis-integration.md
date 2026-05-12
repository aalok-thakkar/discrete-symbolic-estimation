# DiSE × Hypothesis: distribution-aware property-based testing

> "Property-based testing" (PBT) and "operational reliability
> estimation" ask different questions. This document explains why
> bridging them is interesting, and how DiSE does it.

## 1. Two questions, two distributions

**Hypothesis** asks: *Does there exist any input that violates the
property?* It uses *edge-case-biased* strategies — small integers,
boundary cases, deliberately-pathological structures — to find
counterexamples fast. Its distributions are tuned for **bug-finding
coverage**, not realism.

**DiSE** asks: *What fraction of inputs from my operational
distribution violates the property?* It uses *operational*
distributions — production-realistic priors over inputs — and returns
a certified two-sided interval on the violation probability.

These are complementary. PBT-shrinking gives you *one* failing
example. DiSE gives you the *rate* at which failures happen under
the workload you care about.

## 2. The bridge

The adapter [`dise.integrations.hypothesis`](../src/dise/integrations/hypothesis.py)
turns a Hypothesis ``SearchStrategy`` into a DiSE ``Distribution`` and
runs ``estimate`` against the property.

```python
import hypothesis.strategies as st
from dise.integrations.hypothesis import estimate_from_strategies

result = estimate_from_strategies(
    strategies={"a": st.integers(min_value=1, max_value=31),
                "b": st.integers(min_value=1, max_value=31)},
    property_fn=lambda a, b: a * b < (1 << 8),   # 8-bit overflow
    epsilon=0.05, delta=0.05,
)
print(result.mu_hat, result.interval)
# 0.57±... — the certified probability that a*b fits in 8 bits.
```

The result extends a Hypothesis test with:

* **A certified satisfaction probability** $\hat\mu$, not just
  pass/fail.
* **A symbolic uncertainty mass** $W_{\text{open}}$ — the part of
  input space we haven't fully resolved.
* **Dominant failure partitions** — the leaves of the frontier with
  highest contribution to the failure side. Read off
  ``result.iterations`` for the trajectory of the refinement.

## 3. Tier-1 support

The adapter currently handles strategies whose mass admits a closed
form under DiSE's distribution interface:

| Hypothesis strategy                                       | DiSE distribution                       |
|-----------------------------------------------------------|-----------------------------------------|
| ``st.integers(min_value=L, max_value=H)``                 | ``Uniform(L, H)``                       |
| ``st.sampled_from(values)`` (consecutive integers)        | ``Uniform(min(values), max(values))``    |

Composite strategies — ``st.tuples``, ``st.lists``, ``st.builds`` —
are converted *only* if every leaf strategy is Tier-1 supported. For
unsupported strategies, the adapter raises ``NotImplementedError``
with a pointer to manual construction.

## 4. Tier-2 (future)

Two natural extensions:

1. **Sample-based mass.** For an arbitrary Hypothesis strategy, draw
   $N$ samples to estimate the support and per-region acceptance,
   then treat the distribution as a black box. DiSE's `GeneralRegion`
   already supports this style; the missing piece is a `BlackBoxDistribution`
   adapter that wraps a generator.
2. **Strategy ↔ symbolic-region coupling.** Hypothesis strategies are
   compositional (``st.lists(st.integers())``); DiSE's path-condition
   regions are also compositional. The interesting research question
   is: can a DiSE refinement step ``Refine(π, b)`` translate to a
   *constrained* Hypothesis strategy that only generates inputs
   satisfying $\pi \wedge b$? If yes, DiSE could *guide Hypothesis*
   into the high-information regions of input space — adaptive PBT
   that shrinks toward the operational tail rather than toward
   syntactic minima.

This second direction is the most exciting one for a follow-up
paper. The Tier-1 adapter is the scaffold.

## 5. Caveats

* Hypothesis's internal random source is a complex bytestring-based
  encoder ("ConjectureData"). The Tier-1 adapter bypasses it entirely
  — we read off the strategy's *declared* support and sample
  directly. The two never have to agree; if your strategy uses
  ``.filter(...)`` or ``.flatmap(...)`` in a way that changes the
  effective distribution, the Tier-1 adapter will under-count.
* Hypothesis is **soft-imported**: importing
  ``dise.integrations.hypothesis`` raises a friendly ``ImportError``
  if the library is missing. The rest of DiSE has no Hypothesis
  dependency.

## 6. When you should still use plain Hypothesis

PBT is the right tool when you want to know *whether* a bug exists,
not how often. DiSE complements it by certifying the *rate* of bugs
under realistic workloads — the two should be used together, not as
substitutes.
