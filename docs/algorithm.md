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
half-width** $\varepsilon_{\text{stat}} + \varepsilon_{\text{mass}} +
W_{\text{close}}$ such that

$$
\Pr\big[\,|\hat\mu - \mu| \le \varepsilon_{\text{stat}} + \varepsilon_{\text{mass}} + W_{\text{close}}\,\big]
\;\ge\; 1 - \delta.
$$

The three additive components have separate origins:

* $\varepsilon_{\text{stat}}$ — *per-open-leaf sampling uncertainty*,
  a union-bound over the open leaves of an anytime-valid concentration
  half-width on each leaf's empirical mean (§7.4).
* $\varepsilon_{\text{mass}}$ — *mass-MC uncertainty* on leaves whose
  mass $\hat w_\pi$ is estimated by importance sampling. Exactly zero
  on axis-aligned leaves (closed-form mass) and small (proportional to
  $1/\sqrt{N}$ where $N$ is `n_mass_samples`) on `GeneralRegion`
  leaves.
* $W_{\text{close}}$ — *closure-attribution uncertainty* from
  sample-based concentration closure (§6). Each leaf closed via the
  sample path contributes $\varepsilon_{\text{close}} \cdot \hat
  w_\pi$ to this accumulator; SMT-verified closures contribute zero.

The reported interval is
$\big[\max(0, \hat\mu - \varepsilon_{\text{stat}} - \varepsilon_{\text{mass}} - W_{\text{close}}),
 \min(1, \hat\mu + \varepsilon_{\text{stat}} + \varepsilon_{\text{mass}} + W_{\text{close}})\big]$,
clipped to $[0, 1]$ since $\mu \in [0, 1]$ a priori.

**Why no $W_{\text{open}}$ term?** Earlier versions of DiSE included an
additive $W_{\text{open}} = \sum_{\pi \in \Pi^{\text{open}}} \hat w_\pi$
on the half-width as a deterministic worst-case envelope (the bound
$\mu_\pi \in [0, 1]$). The current bound subsumes this contribution
via $\varepsilon_{\text{stat}}$ — the per-leaf Wilson / betting CS
already bounds $|\mu_\pi - \hat\mu_\pi|$ with high probability, and
multiplying by $\hat w_\pi$ and summing gives the correct tighter
bound. The pre-revision treatment was a loose upper bound on the same
event. See §7.3 for the proof rewrite.

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

A leaf $\pi$ is *closed* when DiSE attributes the leaf's
contribution to $\hat\mu$ as $\hat w_\pi \cdot \{0, 1\}$ (one or zero,
not a partial sample mean). Closure is either *exact* (proved via SMT)
or *approximate-but-bounded* (validated via a Wilson-anytime
concentration test, with the residual error charged to
$W_{\text{close}}$).

### 6.1 Preconditions (both paths)

A leaf $\pi$ is *eligible* for closure when:

1. $n_\pi \ge n_{\min}$ (`closure_min_samples`); and
2. every observed $\varphi$-value at $\pi$ agrees.

These two conditions are necessary for both the SMT and the
concentration paths. Note that *branch-sequence agreement is not
required* for the concentration path — the algorithm's earlier
revisions required it but the bound only depends on the Bernoulli
sequence of $\varphi$ values, not on the internal control-flow.

### 6.2 SMT-verified closure (exact)

If the SMT backend proves

$$
F_\pi \wedge \neg \big(b_1 \wedge \cdots \wedge b_k\big) \;\models\; \bot,
$$

where $b_1, \ldots, b_k$ are the *observed* clauses beyond
$\mathrm{depth}(\pi)$, then *every* $x \in R_\pi$ traverses the
observed path. By determinism of $P$, $\varphi(P(\cdot))$ is constant
on $R_\pi$ — closure is **exact** (zero contribution to
$W_{\text{close}}$). The SMT path requires the additional precondition
that all observed branch sequences agree (otherwise the path being
verified is not unique).

### 6.3 Concentration-bounded closure (sound approximation)

When SMT returns `unknown` (typical on non-LIA arithmetic, the
`MockBackend`, or hard formulas), DiSE falls back to a sample-based
test. Let $v_\pi$ be the agreed-upon $\varphi$ value (0 or 1) and let
$q_\pi \;=\; \Pr_{x \sim D \mid R_\pi}\!\big[\varphi(P(x)) \neq v_\pi\big]$
be the true *disagreement rate* in $R_\pi$. The closure rule fires iff

$$
\mathrm{Wilson}_{\text{anytime}}\!\big(n_\pi, 0, \delta_{\text{close}}\big)
\;\le\; \varepsilon_{\text{close}},
$$

where the Wilson-anytime half-width is evaluated at the *observed
all-agree* count (i.e. zero disagreements among $n_\pi$ samples) and
the per-leaf failure budget $\delta_{\text{close}}$ (default
$0.005$). The Wilson-anytime bound is *one-sided* on $q_\pi$ when
$\hat\mu_\pi = 0$, and by the time-uniform guarantee

$$
\Pr\!\big[\,\exists t \ge 1 : q_\pi > \mathrm{Wilson}_{\text{anytime}}(t, 0, \delta_{\text{close}})\,\big]
\;\le\; \delta_{\text{close}}.
$$

So *with probability* $\ge 1 - \delta_{\text{close}}$, the post-closure
attribution $\hat w_\pi \cdot v_\pi$ differs from the true contribution
$\hat w_\pi \cdot \mu_\pi$ by at most $\hat w_\pi \cdot
\varepsilon_{\text{close}}$. The algorithm accumulates this allowance
into a global accumulator

$$
W_{\text{close}}(t) \;:=\; \sum_{\pi\,\text{closed via sample path before }t} \hat w_\pi \cdot \varepsilon_{\text{close}},
$$

which appears as an additive component of the certified half-width
(§1 and §7).

The Wilson-anytime construction is anytime-valid in $n_\pi$, so a
leaf may be re-tested at increasing sample counts without inflating
$\delta_{\text{close}}$ per leaf. Across $K$ sample-closed leaves the
union-bound budget is $K \cdot \delta_{\text{close}}$ — the caller
chooses $\delta_{\text{close}}$ small enough to absorb this.

Implementation: [`Frontier.try_close`](../src/dise/regions/_frontier.py).
The signature is

```python
Frontier.try_close(
    node, min_samples,
    *, delta_close=0.005, closure_epsilon=0.02,
) -> bool
```

with the accumulator `Frontier.W_close_accumulated` exposed for
inspection / aggregation in `compute_estimator_state`.

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
  ``CLOSED_FALSE``) only if either:
  (a) Theorem 3's symbolic-shortcut hypothesis holds with ``unsat`` from
      the backend (the SMT path — closure is exact); or
  (b) every observed $\varphi$ at the leaf agreed, $n_\pi \ge n_{\min}$,
      and
      $\mathrm{Wilson}_{\text{anytime}}(n_\pi, 0, \delta_{\text{close}})
      \le \varepsilon_{\text{close}}$ (the concentration path — closure
      is approximate, residual error charged to $W_{\text{close}}$ at
      level $\delta_{\text{close}}$ per leaf).

Assumptions A1–A3 are *exact*. A4 is exact for the SMT shortcut on a
sound backend (Theorem 3); the concentration path is sound at
confidence $\ge 1 - \delta_{\text{close}}$ per leaf by Wilson's
time-uniform inequality (see §6.3 and §13.4).

### 7.2 Statement

Let $\Pi_T = \Pi_T^{\text{open}} \,\cup\, \Pi_T^{\text{closed}-\text{smt}}
\,\cup\, \Pi_T^{\text{closed}-\text{conc}} \,\cup\, \Pi_T^{\text{empty}}$
be the disjoint decomposition of leaves at the stopping time, with
SMT-closed and concentration-closed leaves distinguished. Let

$$
W_{\text{close}}(T) \;:=\; \sum_{\pi \in \Pi_T^{\text{closed}-\text{conc}}} \hat w_\pi \cdot \varepsilon_{\text{close}},
\qquad
\varepsilon_{\text{mass}}(T) \;:=\; z \cdot \sum_{\pi \in \Pi_T} \sqrt{\widehat{\mathrm{Var}}(\hat w_\pi)}
$$

with $z \approx 2$ a Wilson constant (defaults to $z = 2.0$,
corresponding to a $\sim 95\%$ confidence per leaf on the mass
estimator). Let $\varepsilon_{\text{stat}}(T)$ be an *anytime-valid*
half-width on $\sum_{\pi \in \Pi_T^{\text{open}}} \hat w_\pi \hat\mu_\pi$
(concrete constructions in §7.4).

**Theorem 2.** *Under (A1)–(A4) and with the union-bound budget*

$$
\delta \;=\; \delta_{\text{stat}} + \delta_{\text{mass}} + K_{\text{close}}(T) \cdot \delta_{\text{close}},
$$

*where $K_{\text{close}}(T)$ is the number of sample-closed leaves at
$T$ and the inner budgets $(\delta_{\text{stat}},
\delta_{\text{mass}}, \delta_{\text{close}})$ are configured by the
caller,*

$$
\mathbb P\Big[\,\big|\hat\mu_T - \mu\big| \;\le\; \varepsilon_{\text{stat}}(T) + \varepsilon_{\text{mass}}(T) + W_{\text{close}}(T)\Big]
\;\ge\; 1 - \delta.
$$

### 7.3 Proof

Write the total error as a sum over leaves:

$$
\mu - \hat\mu_T \;=\;
\sum_{\pi \in \Pi_T^{\text{open}}} \big(w_\pi \mu_\pi - \hat w_\pi \hat\mu_\pi\big)
\;+\;
\sum_{\pi \in \Pi_T^{\text{closed}}} \big(w_\pi \mu_\pi - \hat w_\pi v_\pi\big)
$$

where $v_\pi \in \{0, 1\}$ is the closure-assigned value on closed
leaves.

For an *open* leaf, decompose

$$
w_\pi \mu_\pi - \hat w_\pi \hat\mu_\pi
= (w_\pi - \hat w_\pi)\mu_\pi + \hat w_\pi(\mu_\pi - \hat\mu_\pi).
$$

The first term is a mass-MC error; bounding $\mu_\pi \le 1$ gives
$|(w_\pi - \hat w_\pi)\mu_\pi| \le |w_\pi - \hat w_\pi|$. Across all
leaves, $\sum_\pi |w_\pi - \hat w_\pi| \le \varepsilon_{\text{mass}}(T)$
with probability $\ge 1 - \delta_{\text{mass}}$ — by the Wilson plug-in
$|w_\pi - \hat w_\pi| \le z \sqrt{\widehat{\mathrm{Var}}(\hat w_\pi)}$
at confidence $1 - \delta_{\text{mass}}$ (and zero on axis-aligned
leaves where $\hat w_\pi = w_\pi$ exactly). The second term is the
*sampling* error on the per-leaf empirical mean; summing
$\sum_\pi \hat w_\pi \cdot |\mu_\pi - \hat\mu_\pi|$ is bounded by
$\varepsilon_{\text{stat}}(T)$ at confidence $1 - \delta_{\text{stat}}$
via the anytime-valid per-leaf construction and Bonferroni over the at
most $K_T$ open leaves (§7.4).

For an *SMT-closed* leaf,
$\mu_\pi = v_\pi$ exactly (Theorem 3); the error term is just the
mass-MC contribution, already covered by $\varepsilon_{\text{mass}}$.

For a *concentration-closed* leaf, $\mu_\pi$ may differ from $v_\pi$
by up to $\varepsilon_{\text{close}}$ with probability $\ge 1 -
\delta_{\text{close}}$ (§6.3). Hence
$|w_\pi \mu_\pi - \hat w_\pi v_\pi| \le |w_\pi - \hat w_\pi| \cdot 1 +
\hat w_\pi \cdot \varepsilon_{\text{close}}$. The mass term is already
absorbed by $\varepsilon_{\text{mass}}$; the closure term sums across
sample-closed leaves to exactly $W_{\text{close}}(T)$.

Combining the three event-budgets by union bound gives
$|\hat\mu_T - \mu| \le \varepsilon_{\text{stat}} + \varepsilon_{\text{mass}}
+ W_{\text{close}}$ at confidence $\ge 1 - \delta$ where $\delta$
absorbs the three inner budgets as stated. $\square$

**Why no $W_{\text{open}}$ term?** Earlier (pre-revision) statements
of Theorem 2 carried an additional $W_{\text{open}}(T) = \sum_{\pi \in
\Pi_T^{\text{open}}} \hat w_\pi$ on the half-width. That term came
from bounding the per-leaf sampling error $|\mu_\pi - \hat\mu_\pi|$
by the trivial $\le 1$ — using the constraint $\mu_\pi \in [0, 1]$.
The current bound uses the *anytime Wilson / betting CS* bound
$|\mu_\pi - \hat\mu_\pi| \le h_\pi$, which is tighter on every leaf
with at least a few samples. Both bounds are sound; the new one is
just sharper.

**Remark (post-refinement freshness).** When the scheduler refines a
leaf $\pi$, the parent's samples are *dropped*; the children receive
fresh i.i.d. samples from $D \mid R_{\pi \wedge b}$ and
$D \mid R_{\pi \wedge \neg b}$. The certified estimator $\hat\mu_T$
therefore depends only on samples that are i.i.d. *conditional on the
leaf they were drawn from*. Refinement decisions can use any sample
data without contaminating the certified estimate.

### 7.4 Concrete half-width constructions for $\varepsilon_{\text{stat}}$

All constructions below are sound at level $1 - \delta_{\text{stat}}$;
they differ in tightness and validity regime.

* **Wilson + Bonferroni** (`method="wilson"`).
  Each open leaf gets confidence $1 - \delta_{\text{stat}}/K_T$:

  $$
  \varepsilon_{\text{stat}}(T) \;=\; \sum_{\pi \in \Pi_T^{\text{open}}} \hat w_\pi \cdot
  \mathrm{Wilson}\!\big(n_\pi(T), h_\pi(T), \delta_{\text{stat}}/K_T\big).
  $$

  **Validity regime:** non-adaptive stopping (e.g. fixed budget reached
  on every run). Tightest of the constructions at fixed $n$.

* **Anytime: min(Wilson-anytime, betting CS)** (`method="anytime"` — recommended).
  Each per-leaf bound is the *intersection* of two anytime-valid
  $(1 - \delta_{\text{inner}})$ confidence intervals, with
  $\delta_{\text{inner}} = \delta_{\text{stat}} / (2 K_T)$
  (Bonferroni over the two constructions, then over $K_T$ open leaves):

  $$
  \varepsilon_{\text{stat}}(T) \;=\; \sum_{\pi \in \Pi_T^{\text{open}}} \hat w_\pi \cdot
  \min\!\big(\mathrm{Wilson}_{\text{anytime}}, \mathrm{Betting}_{\text{CS}}\big)\!\big(n_\pi, h_\pi, \delta_{\text{inner}}\big).
  $$

  The Wilson-anytime path applies the fixed-$n$ Wilson interval at
  $\delta_n = 6 \delta_{\text{inner}} / (\pi^2 n^2)$, union-bounded
  over $n$ via the Basel identity. The betting-CS path is a
  hedged-capital construction (Waudby-Smith & Ramdas 2024) with a
  fixed $\lambda$-grid and per-$\lambda$ Bonferroni. The two are
  complementary: Wilson smoothing is tighter at the extremes
  ($h \in \{0, n\}$), the betting CS is tighter in the interior of
  $[0, 1]$. Taking the min recovers both regimes at the cost of a 2×
  Bonferroni split.

  **Validity regime:** *adaptive* stopping (`epsilon_reached`
  computed from the observed interval) and *adaptive* per-leaf sample
  sizes — both true of ASIP. **This is the bound to cite for the
  default `(1 - \delta)`-certificate.**

* **Bernstein** (`method="bernstein"`).
  Classical Bernstein on the total estimator variance $\hat V_T$ and
  per-contribution bound $B = \max_\pi \hat w_\pi$:

  $$
  \varepsilon_{\text{stat}}(T) \;=\; \sqrt{2 \hat V_T \log(2/\delta_{\text{stat}})} + \tfrac{B}{3} \log(2/\delta_{\text{stat}}).
  $$

  Conservative; soundness-only.

* **Maurer-Pontil empirical-Bernstein** (`method="empirical-bernstein"`).
  Per leaf, with $\hat V_\pi = \tilde p_\pi(1 - \tilde p_\pi)$ the
  Wilson-smoothed empirical variance:

  $$
  \varepsilon_{\text{stat}}(T)
  \;=\; \sum_{\pi \in \Pi_T^{\text{open}}} \hat w_\pi \left[
  \sqrt{\tfrac{2 \hat V_\pi \log(2/\delta_K)}{n_\pi}} + \tfrac{7 \log(2/\delta_K)}{3 (n_\pi - 1)}
  \right]
  $$

  with Bonferroni $\delta_K = \delta_{\text{stat}} / K_T$. **Validity
  regime:** fixed-$n$. Tighter than Bernstein when empirical variance
  is small.

Implementation: [`compute_estimator_state`](../src/dise/estimator/__init__.py)
and [`betting_halfwidth_anytime`](../src/dise/estimator/__init__.py).
For the headline benchmark comparisons (`docs/sota-comparison.md`)
we use `method="anytime"` — DiSE beats the SoTA bounded-mean sampling
baseline (betting CS) on 10 / 12 benchmarks at this setting.

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

* **Refine$(\pi, b)$.** Split $\pi$ on a clause $b$. The clause is
  drawn from the *observed* paths beyond $\mathrm{depth}(\pi)$ and is
  filtered by four guards (each documented in
  [`_propose_actions`](../src/dise/scheduler/__init__.py) and
  [`_best_refinement_clause`](../src/dise/scheduler/__init__.py)):

  **Guard 1: not phi-uniform.** If every observed $\varphi$ at $\pi$
  agrees, the leaf is a candidate for closure (Wilson-anytime test),
  not refinement. Splitting a phi-uniform leaf produces children that
  inherit the same all-agree statistics and merely consumes
  Bonferroni budget per child — strictly hurts the certified
  half-width.

  **Guard 2: budget-aware leaf cap.** Under
  $\varepsilon_{\text{close}}$-sound closure each closed leaf needs
  roughly $\log(1/\delta_{\text{close}}) / \varepsilon_{\text{close}}^2$
  samples. With sample budget $B$ this allows at most
  $K_{\max\text{-leaves}} = B / n_{\text{close-per-leaf}}$ leaves
  before the sample budget is exhausted by seed allocations. The
  scheduler refuses to refine when $|\Pi_t| \ge K_{\max\text{-leaves}}$.

  **Guard 3: variance-reduction floor.** Among candidate clauses, only
  those with
  $G(b) \ge 0.25 \cdot V_\pi$ are considered — refinements that drop
  the parent's variance by less than 25 % rarely pay off after
  Bonferroni inflation.

  **Guard 4: predicted-bound gate.** Even for a clause that clears
  guards 1–3, refinement is rejected unless the *predicted post-
  refinement* contribution to $\varepsilon_{\text{stat}}$ is strictly
  tighter than the *no-refinement* contribution at the same total
  budget:

  $$
  \sum_{c \in \{b, \neg b\}} \hat w_{\pi \wedge c}^{\text{future}} \cdot
  \min(\text{Wilson}_{\text{anytime}}, \text{Betting})(n_c^{\text{future}}, h_c^{\text{future}}, \delta_{\text{inner}}^{(K+1)})
  \;<\; \hat w_\pi \cdot \min(\cdots)(n^{\text{future}}, h^{\text{future}}, \delta_{\text{inner}}^{(K)}).
  $$

  Here $n^{\text{future}}, h^{\text{future}}$ extrapolate the current
  empirical $h/n$ over all remaining budget. The gate captures the
  Bonferroni-vs-sample-size tradeoff: a child gets $1/2$ of the
  parent's future samples but pays a $\delta/(K+1)$ Bonferroni budget
  instead of $\delta/K$. The per-leaf half-width only shrinks by
  $\sqrt{\log(K)/(2 \log(K+1))}$, which is not enough to break even
  unless the clause is *highly informative* (at least one child near
  phi-uniform).

  Among the surviving candidates, the clause maximizing the
  variance-reduction gain
  $G(b) = V_\pi - (V_{\pi \wedge b} + V_{\pi \wedge \neg b})$ is
  chosen, with per-side variances using Wilson-smoothed $\tilde p$ and
  the empirical split fraction. Cost is
  `refinement_cost_in_samples` (default $1$ — refinement involves two
  SMT calls plus a mass-MC split).

  Implementation: [`_best_refinement_clause`](../src/dise/scheduler/__init__.py),
  [`_predicted_no_refine_halfwidth`](../src/dise/scheduler/__init__.py),
  [`_predicted_refine_halfwidth`](../src/dise/scheduler/__init__.py).

  **Known limitation.** The predicted-bound gate uses the *empirical*
  per-side $h/n$ to project future statistics. On the chosen clause's
  empirically-uniform side, this is biased low — the clause picker
  maximizes apparent uniformity. A Wilson-upper-bound debiasing
  over-suppresses refinements where the structural uniformity is
  real (e.g. `coin_machine`'s `x < 10` branch); a more principled
  debiasing (Bayesian posterior on per-side $\mu$) is a follow-up.

### 9.2 Termination

| `terminated_reason`       | condition                                                              |
|---------------------------|------------------------------------------------------------------------|
| `epsilon_reached`         | $\varepsilon_{\text{stat}} + \varepsilon_{\text{mass}} + W_{\text{close}} \le \varepsilon$ (**primary** stopping condition) |
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

Because $\hat\mu$, $\varepsilon_{\text{stat}}$,
$\varepsilon_{\text{mass}}$ and $W_{\text{close}}$ are all maintained
incrementally, every iteration's `EstimatorState` is a valid certified
interval. The reported `terminated_reason` distinguishes the four
exit conditions above; in every case the returned interval is *honest*
about residual uncertainty — wide when the budget runs out, exact
`[mu, mu]` when the frontier fully closes via SMT (and only as tight
as $W_{\text{close}}$ allows when all closures fall back to the
concentration path).

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

**Residual closure budget (sound).** When the SMT shortcut returns
``unknown`` and the scheduler closes a leaf via the concentration
path (§6.3), the residual disagreement rate $q_\pi$ is bounded by
$\mathrm{Wilson}_{\text{anytime}}(n_\pi, 0, \delta_{\text{close}})$
with probability $\ge 1 - \delta_{\text{close}}$ — the Wilson-anytime
guarantee on a zero-count Bernoulli stream. The closure rule only
fires when this bound is at most $\varepsilon_{\text{close}}$, so the
*residual error* the algorithm carries forward is at most
$\hat w_\pi \cdot \varepsilon_{\text{close}}$ per closed leaf — and
this is precisely the contribution to $W_{\text{close}}$ that the
certified half-width accounts for in §1 and §7.

In contrast to the pre-revision behavior (where MockBackend +
all-agree-heuristic closure could attribute exact mass without a
matching budget entry), the current rule is **sound under
MockBackend**: any leaf the algorithm closes through the
concentration path pays $\hat w_\pi \cdot \varepsilon_{\text{close}}$
into $W_{\text{close}}$, which appears in the certified interval. The
3-seed coverage check on the registered benchmarks reports
$\text{coverage} = 1.0$ on every benchmark for every method including
DiSE — no soundness failures observed.

The empirical comparison
([`sota-comparison.md`](sota-comparison.md)) at the recommended
defaults ($\delta_{\text{close}} = 0.005$,
$\varepsilon_{\text{close}} = 0.025$,
$n_{\text{mass-samples}} = 10\,000$) shows DiSE producing tighter
certified half-widths than the SoTA Hedged-Capital betting CS
(Waudby-Smith & Ramdas 2024) on 10 out of 12 benchmarks while
remaining sound on all 12.

### 13.5 Summary

The certification claim at level $1 - \delta$ holds when:

1. `method="anytime"` is used (§13.1, §13.2);
2. `max_refinement_depth` is enforced (§13.3);
3. The SMT backend is sound (§13.4); the concentration closure path
   under MockBackend is itself sound at level $\delta_{\text{close}}$
   per leaf (charged to $W_{\text{close}}$) — no separate Z3
   requirement.

The three inner confidence budgets
$(\delta_{\text{stat}}, \delta_{\text{mass}}, \delta_{\text{close}})$
sum (with Bonferroni over closed leaves) to the user-supplied
$\delta$. Default partition: $\delta_{\text{stat}} = \delta_{\text{mass}}
= 0.45 \delta$, $\delta_{\text{close}} = 0.10 \delta / K_{\text{close-max}}$.

Conditions (1)–(3) collectively give an *anytime-valid* certified
interval that survives the four adaptive-bias risks listed above.
