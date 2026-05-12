# DiSE: algorithm and theorems

> A reading guide. The code modules are cross-referenced inline; the
> proofs below match the implementation in
> [`src/dise`](../src/dise) line-for-line.

## 1. Setting

Let $P : \mathcal{X} \to \mathcal{Y}$ be a deterministic program operating
on integer inputs $\mathcal{X} \subseteq \mathbb{Z}^d$. Let $D$ be a
discrete distribution over $\mathcal{X}$ with closed-form factor
PMFs:

$$
D(x_1, \ldots, x_d) = \prod_{i=1}^d D_i(x_i).
$$

Let $\varphi : \mathcal{Y} \to \{0, 1\}$ be a Boolean property of the
program's output. The *operational reliability* under $D$ is

$$
\mu \;=\; \Pr_{x \sim D}\big[\varphi(P(x)) = 1\big].
$$

Given a target accuracy $(\varepsilon, \delta) \in (0, 1)^2$ and a sample
budget $B$, DiSE outputs an estimator $\hat\mu$ and a *certified
two-sided half-width* $\varepsilon_{\text{total}} = \varepsilon_{\text{stat}} + W_{\text{open}}$ such that

$$
\Pr\big[\,|\hat\mu - \mu| \le \varepsilon_{\text{total}}\,\big] \;\ge\; 1 - \delta.
$$

In the implementation we *report* the interval
$\big[\max(0, \hat\mu - \varepsilon_{\text{total}}),\, \min(1, \hat\mu + \varepsilon_{\text{total}})\big]$,
clipped to $[0, 1]$ since $\mu \in [0, 1]$ a priori.

The prototype's headline regime is class **(D1)**: product-form
distributions over integer programs. Classes (D2) Bayes-net-structured
joints and (D3) general discrete are deferred; see
[`docs/limitations.md`](limitations.md).

## 2. The frontier

At any moment in the run, DiSE maintains a *frontier*: a tree of
path-condition regions $\{R_\pi\}_{\pi \in \Pi}$ whose leaves
partition $\mathcal{X}$. Each leaf is in one of five lifecycle states
(see [`Status`](../src/dise/regions/_base.py)):

| State          | Meaning                                                                   |
|----------------|---------------------------------------------------------------------------|
| `OPEN`         | sampled but not yet resolved                                              |
| `CLOSED_TRUE`  | proven $\varphi(P(x)) = 1$ for all $x \in R_\pi$ (path-deterministic)     |
| `CLOSED_FALSE` | proven $\varphi(P(x)) = 0$ for all $x \in R_\pi$                          |
| `EMPTY`        | SMT-proved unsat — mass exactly $0$                                       |
| `DIVERGED`     | concolic exceeded `max_concolic_branches` on every observed sample        |

The *stratified estimator* of $\mu$ is

$$
\hat\mu \;=\; \sum_{\pi \in \Pi}\, \hat w_\pi \cdot \hat\mu_\pi,
\quad\text{where}\quad
\hat w_\pi \approx \Pr_D[X \in R_\pi],\qquad
\hat\mu_\pi \approx \Pr_D[\varphi(P(X)) = 1 \mid X \in R_\pi].
$$

For an axis-aligned region $R_\pi = \{x : \forall i, a_i \le x_i \le b_i\}$,
the mass admits a *closed form*:

$$
\hat w_\pi \;=\; \prod_{i=1}^d D_i(\{a_i, a_i+1, \ldots, b_i\}) \quad\text{with}\quad \mathrm{Var}(\hat w_\pi) = 0.
$$

This is the structural variance-reduction lever that distinguishes DiSE
from plain Monte Carlo on bounded-variance test functions.

## 3. Sample-mean estimator and Wilson smoothing

Within an open leaf $\pi$, $\hat\mu_\pi$ is the sample mean over
$n_\pi$ concolic runs drawn from $D \mid R_\pi$:

$$
\hat\mu_\pi \;=\; \frac{h_\pi}{n_\pi}\quad(h_\pi = \text{hits}).
$$

For variance reasoning we use a *Wilson-smoothed* per-sample plug-in:

$$
\widehat{\mathrm{Var}}_{\text{per-sample}}(\varphi(P(X)) \mid X \in R_\pi) \;=\; \tilde p_\pi (1 - \tilde p_\pi),
\qquad
\tilde p_\pi = \frac{h_\pi + 1}{n_\pi + 2}.
$$

The sample-mean variance is thus
$\widehat{\mathrm{Var}}(\hat\mu_\pi) = \tilde p_\pi(1-\tilde p_\pi)/n_\pi$.
This **never collapses to zero** on all-hits / all-miss batches — a
specific failure mode of vanilla MLE plug-ins that the brief flagged.

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
$\mathbb{E}[Y] = \mu_\pi$, and $X \perp Y$.
Then $\mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$, so
$\mathrm{Var}(XY) = \mathbb{E}[X^2 Y^2] - (\mathbb{E}[X]\mathbb{E}[Y])^2
= \mathbb{E}[X^2]\mathbb{E}[Y^2] - \mathbb{E}[X]^2 \mathbb{E}[Y]^2$.
Substituting $\mathbb{E}[X^2] = \mathrm{Var}(X) + \mathbb{E}[X]^2$ and the
analogue for $Y$ yields the stated identity. $\square$

By independence across leaves (each leaf's samples are drawn from
$D \mid R_\pi$ with fresh randomness; mass estimators use disjoint
randomness pools when sampled),

$$
\mathrm{Var}(\hat\mu) \;=\; \sum_{\pi \in \Pi}\,\mathrm{Var}(\hat w_\pi \hat\mu_\pi).
$$

For axis-aligned leaves $\mathrm{Var}(\hat w_\pi) = 0$ and the
contribution simplifies to $w_\pi^2 \mathrm{Var}(\hat\mu_\pi)$.

Implementation: [`FrontierNode.variance_contribution`](../src/dise/regions/_frontier.py)
and [`Frontier.compute_mu_hat`](../src/dise/regions/_frontier.py).

## 5. Mass-conservative refinement

When DiSE refines leaf $\pi$ on a clause $b$, the two children's path
conditions are $\pi \wedge b$ and $\pi \wedge \neg b$. If both children
are reducible to axis-aligned boxes, their closed-form masses
*partition* $w_\pi$ exactly. If at least one child requires a
:class:`GeneralRegion` (the predicate cannot be reduced to interval
bounds), independent importance-sampling estimates of the two child
masses would *not* sum to $w_\pi$, introducing IS noise into the
partition invariant $\sum_\pi w_\pi = 1$.

DiSE instead draws **one** batch of $N$ samples from
$D \mid R_\pi$, partitions it by $b$, and proportionally splits the
parent's mass:

$$
\hat p_{b \mid \pi} \;=\; \frac{1}{N}\sum_{i=1}^N \mathbf{1}[b(x^{(i)})], \quad
\hat w_{\pi \wedge b} \;=\; \hat w_\pi \cdot \hat p_{b \mid \pi},
\quad
\hat w_{\pi \wedge \neg b} \;=\; \hat w_\pi \cdot (1 - \hat p_{b \mid \pi}).
$$

The Wilson-smoothed split variance
$\hat p \cdot (1 - \hat p) / N$ scaled by $\hat w_\pi^2$ is added to
each child's `w_var`. This restores the conservation invariant exactly
modulo IS noise concentrated at the parent's mass (the root has mass
$1$).

Implementation: [`Frontier._proportional_split_mass`](../src/dise/regions/_frontier.py).

## 6. Closure rule

A leaf $\pi$ is *closed* if every input it admits leads to the same
path through $P$ and hence the same $\varphi$ value.

### Sample-based closure (heuristic)

A leaf $\pi$ is *eligible* for closure when:

1. $n_\pi \ge n_{\min}$ (`closure_min_samples`), and
2. every observed branch sequence at $\pi$ is identical (path
   determinism observed), and
3. every observed $\varphi$-value agrees.

### Symbolic shortcut (sound)

If the SMT backend can additionally prove

$$
F_\pi \wedge \neg \big(b_1 \wedge \cdots \wedge b_k\big) \;\models\; \bot,
$$

where $b_1, \ldots, b_k$ is the *observed* path beyond depth
$\mathrm{depth}(\pi)$, then *every* $x \in R_\pi$ traverses the
observed path. Since $P$ is deterministic, $\varphi(P(\cdot))$ is
constant on $R_\pi$ — closure is *certified*.

DiSE invokes the shortcut whenever raw path clauses are available
(scheduler always provides them). If the backend returns `unknown` —
typical for `MockBackend` on non-axis-aligned arithmetic — closure
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
\underbrace{\sum_{\pi \in \Pi_{\text{open}}} (w_\pi \mu_\pi - \hat w_\pi \hat\mu_\pi)}_{\text{open-region error}}
\;+\; \underbrace{\sum_{\pi \in \Pi_{\text{closed}}} (w_\pi - \hat w_\pi)\cdot\mu_\pi}_{\text{mass error on closed}}.
$$

For axis-aligned closed leaves, $\hat w_\pi = w_\pi$ exactly, so the
second sum is zero. The first sum decomposes further as
$(w_\pi - \hat w_\pi)\mu_\pi + \hat w_\pi(\mu_\pi - \hat\mu_\pi)$. The
absolute value is bounded by $|w_\pi - \hat w_\pi| + \hat w_\pi \cdot |\mu_\pi - \hat\mu_\pi|$
$\le w_\pi + |w_\pi - \hat w_\pi|$, and summing gives at most
$W_{\text{open}}$ once $\hat\mu_\pi \in [0, 1]$. The random
fluctuation around the deterministic component is
$\sum_\pi \hat w_\pi(\hat\mu_\pi - \mu_\pi)$, controlled by
$\varepsilon_{\text{stat}}$. $\square$

**Three half-width methods.** All three implementations are sound
(union-bound-correct at level $1 - \delta$); they differ in tightness.

* **Wilson + Bonferroni** (default; `method="wilson"`).
  Each open leaf gets confidence $1 - \delta/K$ via Wilson's score
  interval; the total half-width is

  $$
  \varepsilon_{\text{stat}} = \sum_{\pi \in \Pi_{\text{open}}} \hat w_\pi \cdot
  \mathrm{Wilson}(n_\pi, h_\pi, \delta/K).
  $$

* **Classical Bernstein** (`method="bernstein"`).
  Use the total estimator variance $\hat V$ and a per-contribution
  bound $B = \max_\pi w_\pi$:

  $$
  \varepsilon_{\text{stat}} = \sqrt{2 \hat V \log(2/\delta)} + \tfrac{B}{3} \log(2/\delta).
  $$

* **Empirical-Bernstein, Maurer–Pontil** (`method="empirical-bernstein"`).
  Per-leaf, with $\hat V_\pi = \tilde p_\pi(1-\tilde p_\pi)$ the
  Wilson-smoothed empirical variance and $M = 1$ the per-sample range:

  $$
  \varepsilon_{\text{stat}}
  = \sum_{\pi \in \Pi_{\text{open}}} \hat w_\pi \left[
  \sqrt{\tfrac{2 \hat V_\pi \log(2/\delta_K)}{n_\pi}} + \tfrac{7 \log(2/\delta_K)}{3 (n_\pi - 1)}
  \right],
  $$
  with Bonferroni $\delta_K = \delta / K$.

Implementation: [`compute_estimator_state`](../src/dise/estimator/__init__.py).
In practice the Wilson sum is the tightest on Bernoulli leaves and is
the default for the main results.

## 8. Theorem 3 (closure correctness)

If the SMT backend returns `unsat` for $F_\pi \wedge \neg \mathrm{path}_\pi$
where $\mathrm{path}_\pi$ is the observed clause-conjunction beyond
$\pi$'s depth, then every $x \in R_\pi$ traverses $\mathrm{path}_\pi$.

*Proof.* Suppose $\exists x \in R_\pi$ taking a path differing from
$\mathrm{path}_\pi$. Then there is some clause $b_j$ in $\mathrm{path}_\pi$
such that $x \not\models b_j$, hence $x \models \neg \mathrm{path}_\pi$.
Combined with $x \in R_\pi \Leftrightarrow x \models F_\pi$, we have
$F_\pi \wedge \neg \mathrm{path}_\pi$ satisfiable — contradicting the
backend's `unsat` (which we trusted by *soundness* of the SMT
oracle). Since $P$ is deterministic, all $x \in R_\pi$ share the same
output and hence the same $\varphi$-value. $\square$

This is what makes Z3-backed runs strictly more accurate than
`MockBackend`-backed runs on path-non-deterministic regions.

## 9. ASIP — adaptive symbolic importance partitioning

The driver loop (in [`scheduler/__init__.py`](../src/dise/scheduler/__init__.py)):

```
ASIPScheduler.run():
    bootstrap(n_bootstrap)             # initial samples at root
    try_close_all()
    while not should_terminate():
        actions = candidate_actions()
        a* = argmax_a (expected_gain(a) / cost(a))
        execute(a*)
        try_close_all()
    return EstimatorState(frontier, delta)
```

### Action types and their expected gains

* **Allocate$(\pi, k)$.** Add $k$ samples to open leaf $\pi$. Expected
  variance reduction (Theorem 1 + Wilson plug-in):

  $$
  \mathrm{gain} \;=\; \frac{\hat w_\pi^2 \cdot \tilde p_\pi(1-\tilde p_\pi)\cdot k}{n_\pi \cdot (n_\pi + k)},
  \qquad \mathrm{cost} = k.
  $$

* **Refine$(\pi, b)$.** Split $\pi$ on clause $b$. For each candidate
  clause $b$ appearing in *some* observed path beyond depth $\pi$, we
  estimate the expected variance reduction by partitioning the
  *current samples at $\pi$* according to whether they took $b$:

  $$
  G(b) \;=\; V_\pi \;-\;\Big(V_{\pi \wedge b} + V_{\pi \wedge \neg b}\Big)
  $$

  with per-side variances using Wilson-smoothed $\tilde p$ and the
  empirical split fraction. Cost is `refinement_cost_in_samples`
  (default $1$ — refinement involves two SMT calls). The clause
  maximizing $G$ is chosen; ties fall back to "first divergent
  position" (see [`_best_refinement_clause`](../src/dise/scheduler/__init__.py)).

* **Termination** at the first iteration where:

  | reason                  | condition                                              |
  |-------------------------|--------------------------------------------------------|
  | `epsilon_reached`       | $\varepsilon_{\text{stat}} + W_{\text{open}} \le \varepsilon$  |
  | `budget_exhausted`      | `samples_used >= budget_samples`                       |
  | `no_actions_available`  | every candidate's gain is non-positive                 |

## 10. Complexity

Let $n_B$ = total sample budget, $K$ = final number of open leaves,
$d$ = maximum refinement depth, $S$ = average SMT cost per query.

* **Per concolic run:** $O(\text{program steps})$ Python overhead + at
  most one SMT-expression node per branch.
* **Action selection per iteration:** $O(K)$ for leaf-level scans;
  $O(K \cdot \text{paths per leaf})$ for variance-aware refinement
  scoring. With path budget $\le$ batch size, this is $O(K \cdot n_B / K)$
  amortized.
* **Refinement:** two SMT `is_satisfiable` calls, one IS batch
  ($N$ samples) of `evaluate` calls. With `CachedBackend`, repeated
  formulas avoid re-querying the underlying SMT solver.
* **Closure:** one SMT call (the path-determinism check) per
  closure attempt; cached for the same $(F_\pi, \mathrm{path}_\pi)$.

Overall, the dominant cost is concolic execution ($O(n_B)$) and SMT
solves ($O(d \cdot K \cdot S)$ in the worst case). The Wilson interval
contributes $O(K)$ to each estimator-state computation.

## 11. Anytime semantics

Because $\hat\mu$, $\varepsilon_{\text{stat}}$, $W_{\text{open}}$ are
*all* maintained incrementally, every iteration's
:class:`EstimatorState` is a valid certified interval. The reported
``terminated_reason`` distinguishes:

* `epsilon_reached` — the target accuracy was hit before exhausting
  the budget.
* `budget_exhausted` — the budget ran out; the returned interval is
  *honest* about the residual uncertainty (often wide, but always
  valid).
* `no_actions_available` — the gain heuristic concluded no remaining
  action is positive-gain. In a well-tuned configuration this
  indicates the frontier has been fully refined.

Honesty about uncertainty is the verification claim.
