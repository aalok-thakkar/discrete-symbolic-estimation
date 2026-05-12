# DiSE: algorithm, theorems, proofs

> A reading guide. Each section is cross-referenced to the implementation.
> Proofs are sketches calibrated to be checkable against the code, not to
> stand alone in a journal.

## 1. The reliability-estimation question

Let $P : \mathcal{X} \to \mathcal{Y}$ be a deterministic program operating
on integer inputs $\mathcal{X} \subseteq \mathbb{Z}^d$. Let $D$ be a
discrete distribution over $\mathcal{X}$ with closed-form factor PMFs:

$$
D(x_1, \ldots, x_d) \;=\; \prod_{i=1}^d D_i(x_i).
$$

Let $\varphi : \mathcal{Y} \to \{0, 1\}$ be a Boolean property of the
program's output. The **operational reliability** of $P$ under $D$ is

$$
\mu \;=\; \Pr_{x \sim D}\big[\varphi(P(x)) = 1\big].
$$

DiSE outputs an estimator $\hat\mu$ and a **certified two-sided
half-width** $\varepsilon_{\text{stat}} + W_{\text{open}}$ such that

$$
\Pr\big[\,|\hat\mu - \mu| \le \varepsilon_{\text{stat}} + W_{\text{open}}\,\big]
\;\ge\; 1 - \delta.
$$

The reported interval is
$\big[\max(0, \hat\mu - \varepsilon_{\text{stat}} - W_{\text{open}}),
 \min(1, \hat\mu + \varepsilon_{\text{stat}} + W_{\text{open}})\big]$,
clipped to $[0, 1]$ since $\mu \in [0, 1]$ a priori.

The prototype's headline regime is class **(D1)** — product-form
distributions over integer programs. Classes (D2) Bayes-net-structured
joints and (D3) general discrete distributions are deferred (see
[`limitations.md`](limitations.md)).

### 1.1 Two framings: property-on-output vs. assertion-violation

The general entry point [`dise.estimate(program, distribution, property_fn, ...)`](../src/dise/estimator/api.py)
takes a Boolean predicate on the program's output. The classical
formal-verification setting — given a program with an `assert` somewhere,
estimate the **failure probability** $\Pr_D[P \text{ violates the
assertion}]$ — is a special case, surfaced via the convenience wrapper
[`dise.failure_probability`](../src/dise/estimator/api.py).
Concretely:

```python
from dise import failure_probability, Uniform

def safe_mul(a: int, b: int) -> int:
    s = a * b
    assert s < (1 << 8), "8-bit overflow"
    return s

result = failure_probability(
    program=safe_mul,
    distribution={"a": Uniform(1, 31), "b": Uniform(1, 31)},
    epsilon=0.05, delta=0.05,
)
# result.mu_hat ≈ 0.40 — the certified overflow probability.
```

Internally the wrapper installs a `try / except` around `program` that
converts the targeted exception class (default `AssertionError`, but
configurable) into a Boolean output, then calls `estimate` with a
property that asks "did the program fail?". The benchmark
[`assertion_overflow_mul_w=8_U(1,31)`](../src/dise/benchmarks/assertion_overflow.py)
is exactly this kernel. The output-property framing strictly subsumes
the assert framing — properties like "result $\le k$" or "result fits
in $w$ bits" don't naturally fit a single assertion site.

## 2. The frontier

ASIP maintains a **frontier**: a tree of path-condition regions
$\{R_\pi\}_{\pi \in \Pi}$ whose leaves partition $\mathcal{X}$. Each leaf
is in one of five lifecycle states (see [`Status`](../src/dise/regions/_base.py)):

| State          | Meaning                                                                   |
|----------------|---------------------------------------------------------------------------|
| `OPEN`         | sampled but not yet resolved                                              |
| `CLOSED_TRUE`  | proven $\varphi(P(x)) = 1$ for every $x \in R_\pi$ (path-deterministic)   |
| `CLOSED_FALSE` | proven $\varphi(P(x)) = 0$ for every $x \in R_\pi$                        |
| `EMPTY`        | SMT-proved unsat — mass exactly $0$                                       |
| `DIVERGED`     | concolic exceeded `max_concolic_branches` on every observed sample        |

The **stratified estimator** of $\mu$ is

$$
\hat\mu \;=\; \sum_{\pi \in \Pi}\, \hat w_\pi \cdot \hat\mu_\pi,
\qquad
\hat w_\pi \approx \Pr_D[X \in R_\pi],\;\;
\hat\mu_\pi \approx \Pr_D[\varphi(P(X)) = 1 \mid X \in R_\pi].
$$

For an axis-aligned region $R_\pi = \prod_i [a_i, b_i]$, the mass admits
a **closed form** with zero variance:

$$
\hat w_\pi \;=\; \prod_{i=1}^d D_i(\{a_i, \ldots, b_i\}), \qquad
\mathrm{Var}(\hat w_\pi) = 0.
$$

This is the structural variance-reduction lever that distinguishes DiSE
from plain Monte Carlo.

## 3. Wilson-smoothed within-leaf variance

Within an open leaf $\pi$, $\hat\mu_\pi = h_\pi / n_\pi$ is the sample
mean over $n_\pi$ concolic runs. For variance reasoning we use the
**Wilson-smoothed per-sample plug-in**:

$$
\widehat{\mathrm{Var}}\big(\varphi(P(X)) \mid X \in R_\pi\big) \;=\;
\tilde p_\pi(1 - \tilde p_\pi),\qquad
\tilde p_\pi = \frac{h_\pi + 1}{n_\pi + 2}.
$$

The sample-mean variance is therefore
$\widehat{\mathrm{Var}}(\hat\mu_\pi) = \tilde p_\pi(1-\tilde p_\pi)/n_\pi$.
**This never collapses to zero** on all-hits / all-miss batches — a
specific failure mode of the MLE plug-in that the brief flagged.

In code: [`FrontierNode.mu_var`](../src/dise/regions/_frontier.py).

## 4. Theorem 1 (variance identity)

For an *independent* mass estimator $\hat w_\pi$ and sample mean
$\hat\mu_\pi$,

$$
\mathrm{Var}(\hat w_\pi \hat\mu_\pi)
\;=\; w_\pi^2 \mathrm{Var}(\hat\mu_\pi)
\;+\; \mu_\pi^2 \mathrm{Var}(\hat w_\pi)
\;+\; \mathrm{Var}(\hat w_\pi)\mathrm{Var}(\hat\mu_\pi).
$$

*Proof.* Let $X = \hat w_\pi$, $Y = \hat\mu_\pi$, $\mathbb{E}[X] = w_\pi$,
$\mathbb{E}[Y] = \mu_\pi$, $X \perp\!\!\!\perp Y$.
Then $\mathbb{E}[XY] = w_\pi \mu_\pi$, so

$$
\mathrm{Var}(XY) = \mathbb{E}[X^2 Y^2] - (w_\pi \mu_\pi)^2
= \mathbb{E}[X^2]\mathbb{E}[Y^2] - w_\pi^2 \mu_\pi^2.
$$

Substituting $\mathbb{E}[X^2] = \mathrm{Var}(X) + w_\pi^2$ and the
analogue for $Y$, then expanding, gives the identity. $\square$

Cross-leaf independence (mass and sample-mean estimators draw from
disjoint randomness pools and use distinct concolic runs) yields

$$
\mathrm{Var}(\hat\mu) \;=\; \sum_{\pi \in \Pi}\,\mathrm{Var}(\hat w_\pi \hat\mu_\pi).
$$

For axis-aligned leaves, $\mathrm{Var}(\hat w_\pi) = 0$ and the
contribution collapses to $w_\pi^2 \mathrm{Var}(\hat\mu_\pi)$.

Implementation: [`FrontierNode.variance_contribution`](../src/dise/regions/_frontier.py).

## 5. Mass-conservative refinement

When DiSE refines leaf $\pi$ on a clause $b$, the children have path
conditions $\pi \wedge b$ and $\pi \wedge \neg b$. If both children
reduce to axis-aligned boxes, their closed-form masses partition $w_\pi$
exactly. If at least one child requires a `GeneralRegion`,
*independent* IS mass estimates would not sum to $w_\pi$ — IS noise
would leak into the partition invariant $\sum_\pi w_\pi = 1$.

Instead DiSE draws **one** batch of $N$ samples from
$D \mid R_\pi$, partitions by $b$, and proportionally splits:

$$
\hat p_{b \mid \pi}
\;=\; \frac{1}{N}\sum_{i=1}^N \mathbf{1}[b(x^{(i)})],
\quad
\hat w_{\pi \wedge b} = \hat w_\pi \cdot \hat p_{b \mid \pi},
\quad
\hat w_{\pi \wedge \neg b} = \hat w_\pi \cdot (1 - \hat p_{b \mid \pi}).
$$

The Wilson-smoothed split variance
$\tilde p_b(1 - \tilde p_b)/N$ scaled by $\hat w_\pi^2$ is added to each
child's `w_var`. This restores the conservation invariant
$\sum_\pi w_\pi = w_{\text{parent}}$ exactly modulo IS noise concentrated
at the parent's mass (the root has mass $1$).

Implementation: [`Frontier._proportional_split_mass`](../src/dise/regions/_frontier.py).

## 6. Closure rule

A leaf $\pi$ is *closed* if every input it admits leads to the same
path through $P$ and hence — by determinism of $P$ — the same
$\varphi$-value.

### 6.1 Sample-based closure (heuristic)

A leaf $\pi$ is *eligible* for closure when:

1. $n_\pi \ge n_{\min}$ (`closure_min_samples`), and
2. every observed branch sequence at $\pi$ is identical, and
3. every observed $\varphi$-value agrees.

### 6.2 Symbolic shortcut (sound)

If the SMT backend additionally proves

$$
F_\pi \wedge \neg \big(b_1 \wedge \cdots \wedge b_k\big) \;\models\; \bot,
$$

where $b_1, \ldots, b_k$ are the *observed* clauses beyond
$\mathrm{depth}(\pi)$, then *every* $x \in R_\pi$ traverses the
observed path. By determinism, $\varphi(P(\cdot))$ is constant on
$R_\pi$ — closure is **certified**.

DiSE invokes the shortcut whenever raw path clauses are available
(the scheduler always provides them). If the backend returns `unknown`
— typical for `MockBackend` on non-axis-aligned arithmetic — closure
falls back to the sample-based criterion per the brief.

Implementation: [`Frontier.try_close`](../src/dise/regions/_frontier.py).

## 7. Theorem 2 (certified two-sided interval)

**Setup.** Decompose the leaves into three disjoint sets:
$\Pi_{\text{closed}}$ (CLOSED_TRUE ∪ CLOSED_FALSE) with exact
$\mu_\pi \in \{0, 1\}$; $\Pi_{\text{open}}$ with sampled $\hat\mu_\pi$;
$\Pi_{\text{empty}}$ with $w_\pi = 0$. Let
$W_{\text{open}} = \sum_{\pi \in \Pi_{\text{open}}} w_\pi$ and let
$\varepsilon_{\text{stat}}$ be a $(1 - \delta)$-valid half-width on
$\sum_{\pi \in \Pi_{\text{open}}} \hat w_\pi \hat\mu_\pi$.

**Claim.** With probability $\ge 1 - \delta$,

$$
\hat\mu - \varepsilon_{\text{stat}} - W_{\text{open}}
\;\le\; \mu \;\le\;
\hat\mu + \varepsilon_{\text{stat}} + W_{\text{open}}.
$$

*Proof sketch.* Write

$$
\mu - \hat\mu \;=\;
\underbrace{\sum_{\pi \in \Pi_{\text{open}}} \big(w_\pi \mu_\pi - \hat w_\pi \hat\mu_\pi\big)}_{\text{open-region error}}
\;+\; \underbrace{\sum_{\pi \in \Pi_{\text{closed}}} \big(w_\pi - \hat w_\pi\big)\mu_\pi}_{\text{mass error on closed leaves}}.
$$

For axis-aligned closed leaves $\hat w_\pi = w_\pi$ exactly, so the
closed-mass term is zero. For each open term decompose
$w_\pi \mu_\pi - \hat w_\pi \hat\mu_\pi
= (w_\pi - \hat w_\pi)\mu_\pi + \hat w_\pi(\mu_\pi - \hat\mu_\pi)$.
The first summand is bounded in absolute value by $|w_\pi - \hat w_\pi|$
(since $\mu_\pi \in [0, 1]$); the second by $\hat w_\pi \cdot
|\mu_\pi - \hat\mu_\pi| \le \hat w_\pi$. Summing and using
$\sum_{\pi \text{ open}} \hat w_\pi \le W_{\text{open}}$ bounds the
deterministic component by $W_{\text{open}}$. The random fluctuation
$\sum_\pi \hat w_\pi(\hat\mu_\pi - \mu_\pi)$ is controlled by
$\varepsilon_{\text{stat}}$. The union of the two events giving the
two-sided bound holds with probability $\ge 1 - \delta$. $\square$

**Three half-width methods.** All three are sound (union-bound-correct
at level $1 - \delta$); they differ in tightness.

* **Wilson + Bonferroni** (default; `method="wilson"`).
  Each open leaf gets confidence $1 - \delta/K$ via Wilson's score
  interval:

  $$
  \varepsilon_{\text{stat}} \;=\; \sum_{\pi \in \Pi_{\text{open}}} \hat w_\pi \cdot
  \mathrm{Wilson}(n_\pi, h_\pi, \delta/K).
  $$

* **Classical Bernstein** (`method="bernstein"`).
  Use the total estimator variance $\hat V$ and a per-contribution
  bound $B = \max_\pi w_\pi$:

  $$
  \varepsilon_{\text{stat}} \;=\;
  \sqrt{2 \hat V \log(2/\delta)} + \tfrac{B}{3} \log(2/\delta).
  $$

* **Empirical-Bernstein (Maurer–Pontil 2009)** (`method="empirical-bernstein"`).
  Per-leaf, with $\hat V_\pi = \tilde p_\pi(1-\tilde p_\pi)$ the
  Wilson-smoothed empirical variance and $M = 1$ the per-sample range:

  $$
  \varepsilon_{\text{stat}}
  \;=\; \sum_{\pi \in \Pi_{\text{open}}} \hat w_\pi \left[
  \sqrt{\tfrac{2 \hat V_\pi \log(2/\delta_K)}{n_\pi}} + \tfrac{7 \log(2/\delta_K)}{3 (n_\pi - 1)}
  \right],
  $$

  with Bonferroni $\delta_K = \delta / K$.

Implementation: [`compute_estimator_state`](../src/dise/estimator/__init__.py).
In practice the Wilson sum is tightest on Bernoulli leaves and is the
default for the main results.

## 8. Theorem 3 (closure correctness)

If the SMT backend returns `unsat` for
$F_\pi \wedge \neg \mathrm{path}_\pi$ (where $\mathrm{path}_\pi$ is the
observed clause-conjunction beyond $\pi$'s depth), then every
$x \in R_\pi$ traverses $\mathrm{path}_\pi$.

*Proof.* Suppose $\exists x \in R_\pi$ taking a path differing from
$\mathrm{path}_\pi$. Then some clause $b_j$ of $\mathrm{path}_\pi$
satisfies $x \not\models b_j$, hence $x \models \neg \mathrm{path}_\pi$.
Combined with $x \in R_\pi \Leftrightarrow x \models F_\pi$, we have
$F_\pi \wedge \neg \mathrm{path}_\pi$ satisfiable — contradicting the
backend's `unsat` (which we trust by **soundness** of the SMT oracle).
Since $P$ is deterministic, every $x \in R_\pi$ produces the same
output and hence the same $\varphi$-value. $\square$

This is what makes Z3-backed runs strictly more accurate than
`MockBackend`-backed runs on path-non-deterministic regions.

## 9. ASIP — Adaptive Symbolic Importance Partitioning

The driver loop (in [`scheduler/__init__.py`](../src/dise/scheduler/__init__.py)):

```
ASIPScheduler.run():
    bootstrap(n_bootstrap)                # initial samples at root
    try_close_all()
    while not should_terminate():
        actions = candidate_actions()
        a* = argmax_a (expected_gain(a) / cost(a))
        if gain_per_cost(a*) <= min_gain_per_cost:  return
        execute(a*)
        try_close_all()
    return EstimatorState(frontier, delta)
```

### 9.1 Action types and their expected gains

* **Allocate$(\pi, k)$.** Add $k$ samples to open leaf $\pi$. Expected
  variance reduction (Theorem 1 + Wilson plug-in):

  $$
  \mathrm{gain}(\pi, k) \;=\; \frac{\hat w_\pi^2 \cdot \tilde p_\pi(1-\tilde p_\pi)\cdot k}{n_\pi \cdot (n_\pi + k)},
  \qquad \mathrm{cost} = k.
  $$

* **Refine$(\pi, b)$.** Split $\pi$ on clause $b$. For each candidate
  clause $b$ appearing in *some* observed path beyond depth $\pi$, the
  expected variance reduction is estimated by partitioning the current
  samples at $\pi$ according to whether they took $b$:

  $$
  G(b) \;=\; V_\pi \;-\;\big(V_{\pi \wedge b} + V_{\pi \wedge \neg b}\big),
  $$

  with per-side variances using Wilson-smoothed $\tilde p$ and the
  empirical split fraction. Cost is `refinement_cost_in_samples`
  (default $1$ — refinement involves two SMT calls). The clause
  maximizing $G$ is chosen; ties fall back to "first divergent
  position" (see [`_best_refinement_clause`](../src/dise/scheduler/__init__.py)).

### 9.2 Termination

| `terminated_reason`       | condition                                                              |
|---------------------------|------------------------------------------------------------------------|
| `epsilon_reached`         | $\varepsilon_{\text{stat}} + W_{\text{open}} \le \varepsilon$ (**primary** stopping condition) |
| `budget_exhausted`        | `samples_used >= budget_samples` (optional; pass `budget_samples=None` to disable) |
| `time_exhausted`          | wall-clock exceeded `budget_seconds` (optional; default `None`)        |
| `no_actions_available`    | every candidate has gain-per-cost $\le$ `min_gain_per_cost`            |

The algorithm is **budget-neutral by design**: the gain/cost rule at
every step picks the most-efficient action toward reducing
$\varepsilon_{\text{stat}}$, and the loop halts as soon as the target
$\varepsilon$ is reached. The sample budget exists as a *practical
safety net* for hard targets (e.g. $\varepsilon = 10^{-9}$ on a
high-entropy program where SMT can't refine fast enough). Soundness-
mode runs pass `budget=None`; the algorithm then runs until the target
$\varepsilon$ is reached or — via `min_gain_per_cost` — diminishing
returns are detected. A wall-clock cap (`budget_seconds`) is also
available.

## 10. Complexity

Let $n_B$ = total sample budget, $K$ = final number of open leaves,
$d$ = maximum refinement depth, $S$ = average SMT cost per query.

* **Per concolic run:** $O(\text{program steps})$ Python overhead plus
  at most one SMT-expression node per branch.
* **Action selection per iteration:** $O(K)$ for leaf-level scans;
  $O(K \cdot \text{paths per leaf})$ for variance-aware refinement
  scoring.
* **Refinement:** two SMT `is_satisfiable` calls, one IS batch
  ($N$ samples) of `evaluate` calls. With `CachedBackend`, repeated
  formulas avoid re-querying the underlying SMT solver.
* **Closure:** one SMT call (the path-determinism check) per closure
  attempt; cached for the same $(F_\pi, \mathrm{path}_\pi)$.

Dominant cost: concolic execution ($O(n_B)$) and SMT solves
($O(d \cdot K \cdot S)$). The Wilson interval contributes $O(K)$ per
estimator-state computation.

## 11. Anytime semantics

Because $\hat\mu$, $\varepsilon_{\text{stat}}$, and $W_{\text{open}}$
are maintained incrementally, every iteration's :class:`EstimatorState`
is a valid certified interval. The reported `terminated_reason`
distinguishes the four exit conditions above; in every case the
returned interval is *honest* about residual uncertainty — wide when
the budget runs out, exact `[mu, mu]` when the frontier fully closes.

Honesty about uncertainty is the verification claim.

## 12. Failure probability = special case of $\varphi$

The classical assertion-violation setting

$$
\mu_{\text{fail}} \;=\; \Pr_D[P \text{ raises } \texttt{AssertionError}]
$$

is recovered by

$$
P_{\mathrm{wrapped}}(x) \;:=\;
\begin{cases}
1 & \text{if } P(x) \text{ raises an exception in } \mathcal{E}, \\
0 & \text{otherwise};
\end{cases}
\qquad
\varphi(y) := (y = 1).
$$

with $\mathcal{E}$ a user-specified set of exception classes (default
$\{\texttt{AssertionError}\}$). The wrapper
[`dise.failure_probability`](../src/dise/estimator/api.py) does this
instrumentation automatically and delegates to `estimate`. Theorems 1–3
apply verbatim; the certified interval covers $\mu_{\text{fail}}$ with
probability $\ge 1 - \delta$.

A canonical illustration is the [`assertion_overflow`](../src/dise/benchmarks/assertion_overflow.py)
benchmark: a tiny program that asserts the result of an integer
multiplication fits in `w` bits, with $a, b \sim \text{Uniform}(1, M)$.
DiSE certifies the overflow probability to within $\varepsilon$ in a
few hundred concolic runs.

The output-property framing strictly subsumes assertion-violation:
properties like "result $\le k$" or "result fits in $w$ bits" can't be
expressed as a single assertion site, while every assert can be lifted
into an output property via the wrapper.
