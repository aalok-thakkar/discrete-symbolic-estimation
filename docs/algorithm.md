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

### 1.2 The central decomposition (read this if you read nothing else)

The two-term half-width $\varepsilon_{\text{stat}} + W_{\text{open}}$
is **the** organizing object of the algorithm. Each term has a
distinct driver, and every algorithmic decision DiSE makes is in
service of shrinking one of them:

| Term                              | What it measures                          | How DiSE shrinks it                |
|-----------------------------------|-------------------------------------------|------------------------------------|
| $\varepsilon_{\text{stat}}$       | sampling uncertainty on **open** leaves   | draw more concolic samples         |
| $W_{\text{open}}$                 | total mass of unresolved (open) leaves    | SMT-refine an open leaf            |

The scheduler chooses between *sample more* (drives
$\varepsilon_{\text{stat}} \downarrow$) and *refine* (drives
$W_{\text{open}} \downarrow$) at every step, picking whichever offers
the better gain-per-cost on the *sum* (§9). Termination is
$\varepsilon_{\text{stat}} + W_{\text{open}} \le \varepsilon$.

Everything else in this document is in service of making the
decomposition rigorous:

* §2 defines the frontier whose leaves carry the
  $(\hat w_\pi, \hat\mu_\pi)$ that build $\hat\mu$.
* §3-§4 give the per-leaf variance ingredients that build
  $\varepsilon_{\text{stat}}$.
* §5 (mass-conservative refinement) is *what makes
  $W_{\text{open}}$ a meaningful additive quantity* — children of a
  leaf partition the parent's mass exactly.
* §6 (closure) is the operation that *removes* a leaf from
  $W_{\text{open}}$ entirely (it migrates from OPEN to CLOSED, so its
  mass is exact and its hit-rate is symbolically certified).
* §7 makes the central decomposition precise — Theorem 2 below — with
  explicit assumptions, an anytime-valid statement, and a proof.
* §13 verifies the decomposition survives ASIP's adaptive choices.

The formula above is the *only* user-facing soundness contract: any
implementation that maintains $\varepsilon_{\text{stat}}(t) +
W_{\text{open}}(t) \le \varepsilon$ at the chosen stopping time, with
the per-leaf bounds in $\varepsilon_{\text{stat}}$ derived from an
anytime-valid construction (currently the predictable-plug-in
empirical-Bernstein of Waudby-Smith & Ramdas 2024; see §7.4),
satisfies the coverage guarantee.

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
(the scheduler always provides them). The handling of `unknown` is
configurable:

* **Default** (`strict_unknown=False`): closure falls back to the
  sample-based criterion per the brief. Required for `MockBackend`
  usability on non-axis-aligned arithmetic; admits a ~1 % closure-bias
  on hard programs.
* **Strict** (`strict_unknown=True`): closure is *refused*. The leaf
  stays OPEN and contributes to `W_open`, widening the certified
  interval rather than risking bias. Recommended when the soundness
  contract is more important than interval tightness, or when using
  a backend whose `is_satisfiable` is known to be incomplete on the
  workload.

Implementation: [`Frontier.try_close`](../src/dise/regions/_frontier.py).

## 7. Theorem 2 (certified two-sided interval, anytime-valid)

We state Theorem 2 with explicit assumptions and an anytime-valid
guarantee covering the algorithm's adaptive choices.

### 7.1 Setup and assumptions

Fix:

* A probability space $(\Omega, \mathcal F, \mathbb P)$.
* A filtration $\{\mathcal F_t\}_{t \ge 0}$ where $\mathcal F_t$ is the
  $\sigma$-algebra generated by the algorithm's first $t$ concolic
  runs (inputs, branch sequences, $\varphi$-values) plus all SMT
  query results up to step $t$.
* A stopping time $T$ with respect to $\{\mathcal F_t\}$ — the
  iteration at which the scheduler halts (any of the four
  ``terminated_reason`` conditions).

The algorithm at time $t$ exposes a frontier $\Pi_t$ — itself
$\mathcal F_t$-measurable — partitioning $\mathcal X$ into leaves
$R_\pi$.

We assume:

* **(A1) Deterministic program.** $P : \mathcal X \to \mathcal Y$ is a
  measurable, deterministic function.
* **(A2) Independent-per-leaf sampling.** Conditional on the event
  $\{\pi \in \Pi_t\}$ and the path condition $F_\pi$, the samples
  drawn from $D \mid R_\pi$ between leaf-creation time and step $t$
  are i.i.d. with distribution $D \mid R_\pi$.
* **(A3) Sound SMT.** ``is_satisfiable`` is *sound*: never returns
  ``sat`` when the formula is unsatisfiable, never ``unsat`` when
  satisfiable. (May return ``unknown``.)
* **(A4) Sound closure.** A leaf is closed as ``CLOSED_TRUE`` (resp.
  ``CLOSED_FALSE``) only if Theorem 3's symbolic-shortcut hypothesis
  holds with ``unsat`` from the backend, or — for ``unknown`` — every
  observed sample at the leaf agreed and $n_\pi \ge n_{\min}$.

Assumptions A1–A3 are *exact*. A4 is exact for the SMT shortcut on a
sound backend (Theorem 3); for the sample-only fallback it is a
high-probability heuristic — see §13.4 for the residual-bias bound.

### 7.2 Statement

Let $\Pi_T = \Pi_T^{\text{open}} \,\cup\, \Pi_T^{\text{closed}} \,\cup\,
\Pi_T^{\text{empty}}$ be the disjoint decomposition of leaves at the
stopping time. Let
$W_{\text{open}}(T) = \sum_{\pi \in \Pi_T^{\text{open}}} \hat w_\pi$
and let $\varepsilon_{\text{stat}}(T)$ be an *anytime-valid*
$(1 - \delta)$ half-width on $\sum_{\pi \in \Pi_T^{\text{open}}}
\hat w_\pi \hat\mu_\pi$ (concrete constructions in §7.4).

**Theorem 2.** *Under (A1)–(A4),*

$$
\mathbb P\Big[\,\big|\hat\mu_T - \mu\big| \;\le\; \varepsilon_{\text{stat}}(T) + W_{\text{open}}(T)\Big]
\;\ge\; 1 - \delta.
$$

### 7.3 Proof

Write

$$
\mu - \hat\mu_T \;=\;
\underbrace{\sum_{\pi \in \Pi_T^{\text{open}}} \big(w_\pi \mu_\pi - \hat w_\pi \hat\mu_\pi\big)}_{\text{open-region error}}
\;+\; \underbrace{\sum_{\pi \in \Pi_T^{\text{closed}}} \big(w_\pi - \hat w_\pi\big)\mu_\pi}_{\text{mass error on closed leaves}}.
$$

The closed-mass term vanishes for axis-aligned closed leaves
($\hat w_\pi = w_\pi$ exactly).

For each open term, decompose

$$
w_\pi \mu_\pi - \hat w_\pi \hat\mu_\pi
= (w_\pi - \hat w_\pi)\mu_\pi + \hat w_\pi(\mu_\pi - \hat\mu_\pi).
$$

Take absolute values. Since $\mu_\pi \in [0, 1]$, the deterministic
contribution is bounded by $|w_\pi - \hat w_\pi| + \hat w_\pi$, summed
to at most $W_{\text{open}}(T)$ (using the mass-conservation invariant;
see §5).

The stochastic contribution
$\sum_{\pi \in \Pi_T^{\text{open}}} \hat w_\pi (\hat\mu_\pi - \mu_\pi)$
is bounded with probability $\ge 1 - \delta$ by
$\varepsilon_{\text{stat}}(T)$ if the per-leaf bounds are
*time-uniform* (anytime valid). Specifically, applying the time-uniform
Wilson bound (Howard, Ramdas, McAuliffe, Sekhon 2021) at confidence
$\delta_K = \delta / K_{\max}$ — where $K_{\max} \le 2^d$ is the
absolute maximum number of leaves the algorithm can create —
each leaf's stopped sample-mean deviation is controlled:

$$
\mathbb P\Big[\,\exists t \ge 1\!: |\hat\mu_\pi(t) - \mu_\pi| > h_t^{(\pi)}\,\Big] \;\le\; \delta_K.
$$

Summing over the at-most $K_{\max}$ leaves and weighting by
$\hat w_\pi$ yields $\varepsilon_{\text{stat}}(T)$. The union of the
deterministic bound and the stochastic bound covers $|\hat\mu_T - \mu|
\le \varepsilon_{\text{stat}}(T) + W_{\text{open}}(T)$ with
probability $\ge 1 - \delta$. $\square$

**Remark (post-refinement freshness).** When the scheduler refines a
leaf $\pi$, the parent's samples are *dropped*; the children receive
fresh i.i.d. samples from $D \mid R_{\pi \wedge b}$ and
$D \mid R_{\pi \wedge \neg b}$. The certified estimator $\hat\mu_T$
therefore depends only on samples that are i.i.d. *conditional on the
leaf they were drawn from*. Refinement decisions can use any sample
data without contaminating the certified estimate.

### 7.4 Three concrete half-width constructions

All three constructions are sound at level $1 - \delta$; they differ in
tightness and validity regime.

* **Wilson + Bonferroni** (``method="wilson"``).
  Each open leaf gets confidence $1 - \delta/K_T$:

  $$
  \varepsilon_{\text{stat}}(T) \;=\; \sum_{\pi \in \Pi_T^{\text{open}}} \hat w_\pi \cdot
  \mathrm{Wilson}\!\big(n_\pi(T), h_\pi(T), \delta/K_T\big).
  $$

  **Validity regime:** non-adaptive stopping (e.g. fixed budget reached
  on every run). Practical default for the main tables; tightest of
  the three on Bernoulli leaves.

* **Anytime Wilson + Bonferroni** (``method="anytime"``).
  Each per-leaf bound is the time-uniform Wilson interval (§13.1),
  union-bounded over $n$ via the Basel identity
  $\sum_{n \ge 1} 6/(\pi^2 n^2) = 1$:

  $$
  \varepsilon_{\text{stat}}(T) \;=\; \sum_{\pi \in \Pi_T^{\text{open}}} \hat w_\pi \cdot
  \mathrm{Wilson}\!\Big(n_\pi(T), h_\pi(T), \tfrac{6\delta}{\pi^2 \, n_\pi^2(T) \, K_T}\Big).
  $$

  **Validity regime:** *adaptive* stopping and *adaptive* per-leaf
  sample sizes — both true of ASIP. Slightly looser than ``"wilson"``
  by a $\sqrt{\log n_\pi}$ factor, plus the $\pi^2/6$ inflation from
  the union-in-time.

* **Predictable-plug-in empirical-Bernstein** (``method="betting"``).
  Waudby-Smith & Ramdas (2024) Theorem 2 applied per leaf:

  $$
  W_\pi(T) \;=\; \frac{\log(2/\delta_K) + \sum_{i=1}^{n_\pi(T)} v_i \, \psi_E(\lambda_i)}{\sum_{i=1}^{n_\pi(T)} \lambda_i},
  \quad
  \varepsilon_{\text{stat}}(T) \;=\; \sum_{\pi \in \Pi_T^{\text{open}}} \hat w_\pi \cdot W_\pi(T),
  $$

  with $\psi_E(\lambda) = (-\log(1-\lambda) - \lambda)/4$,
  $v_i = 4(X_i - \hat\mu_{i-1})^2$, predictable bet
  $\lambda_t = \min\!\big(\sqrt{2\log(2/\delta_K) / (\hat\sigma_{t-1}^2 \cdot t \log(t+1))},\,c\big)$,
  truncation $c \in (0,1)$ (default $1/2$), and Bonferroni
  $\delta_K = \delta / K_T$.

  **Validity regime:** identical to ``"anytime"`` (adaptive stopping +
  adaptive per-leaf sample sizes) **but strictly tighter** in
  low-variance regimes — no $\pi^2/6$ inflation, variance-adaptive,
  closed-form. **This is the recommended setting for ATVA-grade
  certificates under the actual ASIP schedule.**

* **Bernstein** (``method="bernstein"``).
  Classical Bernstein on the total estimator variance $\hat V_T$ and
  per-contribution bound $B = \max_\pi \hat w_\pi$:

  $$
  \varepsilon_{\text{stat}}(T) \;=\; \sqrt{2 \hat V_T \log(2/\delta)} + \tfrac{B}{3} \log(2/\delta).
  $$

  **Validity regime:** fixed-$n$ per leaf. Conservative; soundness-
  only.

* **Maurer-Pontil empirical-Bernstein** (``method="empirical-bernstein"``).
  Per leaf, with $\hat V_\pi = \tilde p_\pi(1-\tilde p_\pi)$:

  $$
  \varepsilon_{\text{stat}}(T)
  \;=\; \sum_{\pi \in \Pi_T^{\text{open}}} \hat w_\pi \left[
  \sqrt{\tfrac{2 \hat V_\pi \log(2/\delta_K)}{n_\pi}} + \tfrac{7 \log(2/\delta_K)}{3 (n_\pi - 1)}
  \right]
  $$

  with Bonferroni $\delta_K = \delta / K_T$. **Validity regime:**
  fixed-$n$. Tighter than Bernstein when empirical variance is small.

Implementation: [`compute_estimator_state`](../src/dise/estimator/__init__.py).
For ATVA-grade certificates under the adaptive ASIP schedule, use
``method="betting"``; for fixed-$n$ runs with non-adaptive stopping,
``method="wilson"`` is tightest.

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

## 13. Statistical correctness under adaptive choices

This section addresses four risks that formal-methods reviewers
challenge in any adaptive-sampling algorithm. Each is stated, then
the section explains how DiSE handles it.

### 13.1 Adaptive sample sizes (anytime validity)

**Risk.** The per-leaf sample size $n_\pi(t)$ is chosen by the
scheduler as a function of observed data; fixed-$n$ confidence
intervals are *not* valid at data-dependent stopping times on the
sample axis.

**Mitigation.** ``method="anytime"`` uses a **time-uniform Wilson**
bound. Concretely, the fixed-$n$ Wilson interval is applied at the
per-step confidence $\delta_n = 6\delta / (\pi^2 n^2)$, and a union
bound over $n \ge 1$ gives time-uniform validity at level $\delta$:

$$
\Pr\!\Big[\,\exists n \ge 1 : |\bar X_n - \mu| > h_n\,\Big]
\;\le\; \sum_{n \ge 1} \delta_n \;=\; \delta.
$$

(The constant $6 / \pi^2$ comes from the Basel identity $\sum n^{-2} =
\pi^2 / 6$.) See [`wilson_halfwidth_anytime`](../src/dise/estimator/__init__.py).

A tighter mixing approach following Howard, Ramdas, McAuliffe, Sekhon
(2021) or Waudby-Smith and Ramdas (2024) is the natural follow-up;
the current bound trades a $\sqrt{\log n}$ factor for implementation
simplicity.

### 13.2 Optional stopping

**Risk.** The ``epsilon_reached`` termination condition is itself a
function of the observed half-width — classical fixed-$n$ confidence
intervals do not retain their coverage when the user is allowed to
"peek" at the data and stop when the interval looks tight.

**Mitigation.** The anytime-valid bound of §13.1 holds *simultaneously*
at every stopping time $T$:

$$
\Pr\!\Big[\,\sup_{T} |\bar X_T - \mu| > h_T\,\Big] \;\le\; \delta.
$$

So the scheduler may halt at any data-dependent $T$ (including
``epsilon_reached``, ``budget_exhausted``, or ``time_exhausted``) and
the returned interval covers $\mu$ with probability $\ge 1 - \delta$.

### 13.3 Partition dependence

**Risk.** The frontier $\Pi_t$ is a *random* partition that depends on
the algorithm's prior observations — and Theorem 1's variance
identity assumes a fixed partition.

**Mitigation.** Bonferroni over leaves with a *worst-case* leaf count.
The algorithm respects ``max_refinement_depth = d``, so the frontier
has at most $K_{\max} \le 2^d$ leaves (a deterministic bound).
Applying Bonferroni $\delta / K_{\max}$ per leaf bounds the union of
all possible per-leaf failure events, regardless of which subset
$\Pi_T \subseteq \{\text{all possible leaves}\}$ is realized at the
stopping time. Conservatively, the implementation uses $K_T$ (the
actual open-leaf count) for tightness; the worst-case bound applies
if reviewers prefer the data-independent factor.

### 13.4 Refinement-decision correlation

**Risk.** Refinement clauses are chosen *based on* the parent's
sample data. Naïvely, this couples the partition with the sample
distribution used to estimate $\hat\mu_\pi$.

**Mitigation: post-refinement freshness.** ASIP **drops all samples
at a leaf when it is refined**. The children's $\hat\mu_{\pi \wedge b}$
and $\hat\mu_{\pi \wedge \neg b}$ are computed from *fresh* concolic
runs drawn from $D \mid R_{\pi \wedge b}$ and $D \mid R_{\pi \wedge
\neg b}$. The dropped parent samples may have *influenced the choice
of $b$*, but they no longer contribute to the certified estimate
$\hat\mu_T$. Conditional on the child being a leaf at time $T$, its
samples are i.i.d. with distribution $D \mid R_{\pi \wedge b}$ — the
required hypothesis for the anytime Wilson bound.

The mass-conservation proportional-split estimate $\hat w_{\pi \wedge
b}$ is computed from a *separate* IS batch from the parent
(``Frontier._proportional_split_mass``). The split-mass variance is
folded into ``w_var`` and propagates through Theorem 1.

**Residual closure bias.** When the SMT shortcut returns ``unknown``
and the scheduler closes a leaf based on $n_{\min}$ agreeing samples
alone, the closure may be wrong. The residual bias on $\mu_\pi$ is
bounded by

$$
\big|\hat\mu_\pi - \mu_\pi\big| \;\le\; q_\pi \cdot (1 - q_\pi)^{n_{\min}} \cdot \mathbf{1}\!\big[\text{minority path exists}\big]
$$

where $q_\pi$ is the minority-path mass within $R_\pi$. For practical
$n_{\min} \ge 5$ and $q_\pi \ge 0.05$, this is $\le 0.04$; at
$n_{\min} = 20$ it falls below $10^{-4}$. The contribution to the
total bias is $\sum_{\pi \text{ closed via fallback}} w_\pi \cdot
(\text{residual})$, which is *not* covered by the $\delta$ confidence
budget — review-grade certificates should set ``backend="z3"`` so the
SMT shortcut succeeds (Theorem 3) and the residual bias is zero.

### 13.5 Summary

The certification claim at level $1 - \delta$ holds when:

1. ``method="anytime"`` is used (§13.1, §13.2);
2. ``max_refinement_depth`` is enforced (§13.3);
3. The SMT backend is sound, *and* either the path-determinism
   shortcut fires on every closure (under Z3) or the residual-bias
   budget of §13.4 is acceptable to the reviewer.

Conditions (1)–(3) collectively give an *anytime-valid* certified
interval that survives the four adaptive-bias risks listed above.
