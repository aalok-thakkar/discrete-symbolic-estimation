# Tutorial — a step-by-step walkthrough

This tutorial introduces DiSE through a single small example,
``coin_machine``, and works through every reportable quantity.
Readers comfortable with the algorithm description in
[`algorithm.md`](algorithm.md) may want to skim §3–§5 here for
the concrete artefacts.

## 1. The program

`coin_machine` is a three-region branching program with a
deliberately rare bug:

```python
def coin_machine(x: int) -> int:
    if x < 10:           return 0     # region A — safe
    if x < 100:           return 1    # region B — always flags
    if x % 1000 == 0:     return 1    # rare-bug slice (mass ≈ 0.09 %)
    return 0                          # region C-ok — safe
```

Under the uniform operational distribution
:math:`x \sim \mathrm{Uniform}(1, 9999)`, the true reliability is

$$
\mu \;=\; \Pr[x < 100\ \text{or}\ x \% 1000 = 0] \;=\; \frac{90 + 9}{9999} \;=\; \frac{99}{9999} \;\approx\; 0.0099.
$$

The 99 hits are: 90 in region B (``x ∈ {10, …, 99}``) plus 9 in the
rare-bug slice (``x ∈ {1000, 2000, …, 9000}``). This is the
*operational reliability* we want DiSE to certify.

## 2. Why plain Monte Carlo is expensive here

Plain MC at a fixed budget :math:`n`:

$$
\hat\mu_{\mathrm{MC}} \;=\; \frac{1}{n}\sum_{i=1}^n \mathbf{1}[\varphi(P(x_i)) = 1], \qquad x_i \sim D.
$$

The standard error :math:`\sqrt{\mu (1 - \mu) / n}` is about
:math:`0.001` at :math:`n = 10^4` and :math:`\mu \approx 0.01`. To
hit a half-width of, say, :math:`0.001`, you need
:math:`n \;\approx\; z^2 \mu(1-\mu) / \varepsilon^2 \approx 4 \times 10^4`
samples. The rare-bug slice's 9 inputs are easy to miss entirely on
small budgets.

DiSE wins because the program's control flow exposes the structure:
once the algorithm refines on ``x < 10``, ``x < 100``, and
``x % 1000 == 0``, the three "easy" regions become axis-aligned
boxes whose mass is the *closed-form product of marginal masses* —
zero estimator variance.

## 3. Running DiSE

```python
from dise import estimate, Uniform

result = estimate(
    program=coin_machine,
    distribution={"x": Uniform(1, 9999)},
    property_fn=lambda y: y == 1,
    epsilon=0.02,           # target half-width
    delta=0.05,             # confidence 1 - delta
    budget=2000,            # safety cap
    method="wilson",        # certified-interval construction
    seed=0,
)
print(result)
```

A typical run prints:

```
EstimationResult(mu_hat=0.0099, interval=[0.0099, 0.0099],
                 eps_stat=0, W_open=0,
                 samples=220, refinements=1,
                 terminated='epsilon_reached')
```

DiSE drove the certified interval to a *point* — half-width 0,
covering the true :math:`\mu = 99/9999` exactly — in 220 concolic
runs. The plain-MC cost for the same precision is two orders of
magnitude higher.

## 4. Reading the output

Each field of :class:`EstimationResult`:

| Field                | Meaning                                                                                   |
|----------------------|-------------------------------------------------------------------------------------------|
| ``mu_hat``           | Point estimate of :math:`\mu`. Equal to the truth here.                                   |
| ``interval``         | Certified two-sided interval at confidence :math:`1 - \delta`.                            |
| ``eps_stat``         | :math:`\varepsilon_{\text{stat}}` — the statistical half-width. 0 once all leaves close.  |
| ``W_open``           | :math:`W_{\text{open}}` — mass of open (unresolved) leaves.                               |
| ``samples_used``     | Concolic runs actually executed (≤ ``budget``).                                           |
| ``refinements_done`` | SMT refinements performed.                                                                |
| ``n_leaves``         | Final partition size.                                                                     |
| ``n_open_leaves``    | Leaves still un-closed at termination — should be 0 here.                                 |
| ``n_closed_leaves``  | Leaves closed (true/false) at termination.                                                |
| ``terminated_reason``| One of ``epsilon_reached``, ``budget_exhausted``, ``time_exhausted``, ``no_actions_available``. |

The half-width is :math:`(\text{hi} - \text{lo})/2`, accessible as
``result.half_width``.

## 5. Visualizing the refinement

After the run, ``result.iterations`` records the algorithm's
trajectory — one entry per scheduler iteration, with the
:math:`\hat\mu`, :math:`\varepsilon_{\text{stat}}`,
:math:`W_{\text{open}}`, and interval at that point. For the
``coin_machine`` example the trajectory is short:

```python
for log in result.iterations:
    print(f"iter={log.iter_idx:>2}  action={log.action_kind:<8}  "
          f"leaves={log.n_leaves}  open={log.n_open_leaves}  "
          f"mu_hat={log.mu_hat:.4f}  eps={log.eps_stat:.4g}  W={log.W_open:.4g}")
```

The action sequence is roughly:

* Bootstrap (one ``allocate`` at the root, 200 samples).
* One ``refine`` event splits the input space on the first
  observed divergence (``x < 100``).
* Subsequent refines on ``x < 10`` and ``x % 1000 == 0`` close each
  axis-aligned child via the SMT path-determinism shortcut
  (Theorem 3 in [`algorithm.md`](algorithm.md) §8).

By the time the loop exits, every leaf is ``CLOSED_TRUE`` or
``CLOSED_FALSE`` and the interval has collapsed.

## 6. Assertion-violation framing

The same machinery answers the classical "what's the failure
probability of this assertion?" question:

```python
from dise import failure_probability, Uniform

def safe_mul(a: int, b: int) -> int:
    s = a * b
    assert s < (1 << 8), "8-bit overflow"
    return s

result = failure_probability(
    program=safe_mul,
    distribution={"a": Uniform(1, 31), "b": Uniform(1, 31)},
    epsilon=0.05,
    budget=2000,
)
print(result.mu_hat)        # ≈ 0.40 — overflow probability
```

Internally :func:`dise.failure_probability` wraps ``safe_mul`` in a
``try/except`` that converts ``AssertionError`` to a Boolean failure
marker, then calls :func:`dise.estimate`. See
:doc:`/docs/algorithm` §12 for the formal derivation.

## 7. Anytime-valid certificates

ASIP is adaptive: per-leaf sample sizes are chosen based on observed
variance, and the run stops as soon as
:math:`\varepsilon_{\text{stat}} + W_{\text{open}} \le \varepsilon`.
Classical Wilson intervals are valid only at *fixed* :math:`n` — to
get certificates that survive ASIP's data-dependent stopping rule,
use ``method="anytime"``:

```python
result = estimate(
    program=coin_machine,
    distribution={"x": Uniform(1, 9999)},
    property_fn=lambda y: y == 1,
    epsilon=0.02, delta=0.05,
    method="anytime",     # time-uniform Wilson via Bonferroni-in-time
    budget=None,          # no sample cap — terminate at epsilon_reached
    budget_seconds=60.0,  # but cap wall-clock to 60 s
    seed=0,
)
```

This is the ATVA-grade recommended setting; see
[`algorithm.md`](algorithm.md) §13 for the four adaptive-bias risks
the anytime bound resolves.

## 8. Using a Hypothesis strategy

If you already have a Hypothesis ``SearchStrategy`` describing your
inputs:

```python
import hypothesis.strategies as st
from dise.integrations.hypothesis import estimate_from_strategies

result = estimate_from_strategies(
    strategies={
        "a": st.integers(min_value=1, max_value=31),
        "b": st.integers(min_value=1, max_value=31),
    },
    property_fn=lambda a, b: a * b < (1 << 8),
    epsilon=0.05, delta=0.05,
)
```

This is "operational property-based testing" — Hypothesis answers
"does any input fail?", DiSE answers "what fraction of operational
inputs fail, with certified confidence?". See
[`hypothesis-integration.md`](hypothesis-integration.md).

## 9. Comparing against baselines

To compare DiSE against plain MC and stratified random MC on the
same problem at the same budget:

```bash
dise compare 'coin_machine_U(1,9999)' --budget 2000 --n-seeds 5
```

The output is a table of median ``mu_hat / half_width / samples /
coverage / wall(s)`` per method. The ``coverage`` column is the
fraction of seeds whose interval covered the MC ground truth —
expected to be :math:`\ge 1 - \delta` for sound methods.

## 10. Where to go next

* [`algorithm.md`](algorithm.md) — ASIP, theorems, proofs.
* [`api-reference.md`](api-reference.md) — module-by-module API.
* [`cli-reference.md`](cli-reference.md) — every ``dise`` subcommand.
* [`evaluation.md`](evaluation.md) — comparator methodology and
  benchmark suite.
* [`hypothesis-integration.md`](hypothesis-integration.md) — PBT bridge.
* [`limitations.md`](limitations.md) — what DiSE does *not* support.
* [`related-work.md`](related-work.md) — bibliography and
  positioning.
