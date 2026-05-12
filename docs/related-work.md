# Related work

DiSE sits at the intersection of three well-studied lines of work:
**probabilistic / quantitative program analysis**, **statistical
estimation with concentration bounds**, and **stratified sampling**.
This page positions the prototype relative to each.

## 1. Probabilistic symbolic execution (PSE)

* **Geldenhuys, Dwyer, Visser (2012).** *Probabilistic symbolic
  execution.* ISSTA 2012. Foundational PSE: enumerate symbolic paths,
  use model counting (LattE / Barvinok) to compute exact path
  probabilities for affine path conditions over uniform inputs.
* **Filieri, Frias, Pasareanu, Visser (2015).** *Model counting for
  complex data structures.* TACAS 2015. Extends PSE to structured
  inputs.
* **Borges, Filieri, d'Amorim, Pasareanu, Visser (2014).**
  *Compositional solution space quantification for probabilistic
  software analysis.* PLDI 2014. Scaling PSE via compositional model
  counting.

**Positioning.** PSE is *exact* on every fully-enumerated path; DiSE is
*statistical* — it gives certified intervals with high probability.
PSE struggles with (i) unbounded loops, (ii) non-LIA arithmetic, (iii)
non-uniform input distributions. DiSE handles all three: bounded-depth
concolic execution for (i), `GeneralRegion` + MCMC sampling for (ii),
arbitrary product-form factor PMFs for (iii). DiSE could be viewed as
"PSE with sampling-based mass" — when every region admits a closed-form
mass (axis-aligned + uniform), DiSE and PSE converge to the same exact
answer.

## 2. Probabilistic model checking

* **Kwiatkowska, Norman, Parker.** *PRISM*. Decades of work on
  symbolic model checking for Markov chains, MDPs, CTMCs.
* **STORM (Dehnert et al., 2017).** Successor with stronger
  performance.
* **PMC papers** approach probabilistic verification through
  *model-based* abstractions (states + transitions) rather than
  program-source-level concolic.

**Positioning.** PMC tools require a model in their input language
(PRISM, JANI, Markov chain). DiSE operates directly on the Python
program. The trade-off is that DiSE's estimates are *statistical* —
no exact answer — but the front end accepts arbitrary deterministic
integer programs.

## 3. Sampling-based reliability estimation

* **Sankaranarayanan, Chakarov, Gulwani (2013).** *Static analysis for
  probabilistic programs.* PLDI 2013. Direct analysis of probabilistic
  programs.
* **Sampson, Panchekha, Mytkowicz, McKinley, Grossman, Ceze (2014).**
  *Expressing and verifying probabilistic assertions.* PLDI 2014.
  Closest in spirit: probabilistic assertions inside C programs,
  verified by sampling with Hoeffding bounds.
* **Albarghouthi, Hsu (2019).** *Synthesizing coupling proofs of
  differential privacy.* POPL 2019. Symbolic synthesis side.

**Positioning vs. Sampson et al.** DiSE's `failure_probability`
directly addresses the same assertion-violation question; the key
difference is *adaptive stratification*: Sampson et al. use a uniform
Hoeffding bound across iid samples, while DiSE refines symbolic regions
to concentrate samples in informative leaves. For a target half-width
$\varepsilon$ DiSE typically uses an order of magnitude fewer samples
when there is meaningful path-structural variance.

## 4. Concolic and symbolic execution

* **Sen, Marinov, Agha (2005).** *CUTE: a concolic unit testing
  engine for C.* FSE 2005. The original "concolic" coinage.
* **Godefroid, Klarlund, Sen (2005).** *DART: directed automated
  random testing.* PLDI 2005.
* **Cadar, Dunbar, Engler (2008).** *KLEE: unassisted and automatic
  generation of high-coverage tests for complex systems programs.*
  OSDI 2008.

**Positioning.** DiSE's concolic component is a small Python
implementation, far simpler than KLEE / DART. The novelty is the
coupling between concolic path conditions and a *quantitative*
estimator over an input distribution — the concolic part is a means to
an end (capturing path conditions for refinement), not a coverage
target.

## 5. Concentration bounds and stratified sampling

* **Wilson (1927).** *Probable inference, the law of succession, and
  statistical inference.* JASA 1927. The Wilson score interval — the
  default `eps_stat` method in DiSE.
* **Bernstein (1924/1946).** Bernstein's inequality.
* **Maurer, Pontil (2009).** *Empirical Bernstein bounds and sample
  variance penalization.* COLT 2009. The MP-EB bound implemented as
  `method="empirical-bernstein"`.
* **Owen (2013).** *Monte Carlo theory, methods, and examples*
  (informally "McBook"). Reference text on stratified sampling and
  variance reduction.
* **Acklam (2003).** *An algorithm for computing the inverse normal
  cumulative distribution function*. Web-published rational
  approximation used internally by :func:`wilson_halfwidth_for_leaf`
  to avoid a ``scipy.special`` dependency.

### 5.1 Anytime-valid concentration (the adaptive case)

DiSE's adaptive sampling and optional-stopping rule require *anytime-
valid* bounds — fixed-$n$ Wilson is not enough. Implemented as
``method="anytime"``:

* **Robbins (1970).** *Statistical methods related to the laws of the
  iterated logarithm.* Annals of Mathematical Statistics. The
  foundational mixture-martingale argument.
* **Darling, Robbins (1967).** Iterated-logarithm concentration that
  spawned the mixing literature.
* **Howard, Ramdas, McAuliffe, Sekhon (2021).** *Time-uniform Chernoff
  bounds via nonnegative supermartingales.* Probability Surveys. The
  modern reference; provides time-uniform Bernstein, Hoeffding, etc.
* **Howard, Ramdas (2022).** *Sequential estimation of quantiles with
  applications to A/B testing and best-arm identification.*
  Bernoulli. Anytime-valid intervals for proportions, directly
  applicable to per-leaf Bernoulli estimation.
* **Waudby-Smith, Ramdas (2024).** *Estimating means of bounded random
  variables by betting.* JRSS-B. State-of-the-art tight anytime-valid
  intervals for bounded means; the natural follow-up to the current
  union-bound-in-time construction.

DiSE applies these classical bounds *per leaf*, with Bonferroni
correction over the open-leaf count $K$. The structural-variance
contribution from axis-aligned closed-form mass is what reduces the
effective $K$ in practice. See [`algorithm.md`](algorithm.md) §13 for
how the bounds are stitched together to survive the four adaptive-bias
risks (adaptive sample sizes, optional stopping, partition dependence,
refinement-decision correlation).

## 6. Standalone artifacts and tooling references

* **Z3** (de Moura, Bjørner 2008). The default SMT backend.
* **scipy.stats** for distribution primitives.
* **numpy** for IS sampling.

## 7. Property-based testing (PBT)

* **MacIver et al., *Hypothesis*.** The reference Python PBT library;
  edge-case-biased generators for finding counterexamples.
* **Claessen, Hughes (2000).** *QuickCheck: a lightweight tool for
  random testing of Haskell programs.* ICFP 2000. The progenitor.

**Positioning.** PBT asks *"is there any input that violates the
property?"* — a one-counterexample question. DiSE asks *"what fraction
of inputs from my operational distribution violates the property?"*
— a quantitative question with a certified interval. The integration
in [`docs/hypothesis-integration.md`](hypothesis-integration.md)
converts Hypothesis strategies into DiSE distributions; the
research direction it points to (operational PBT, adaptive
strategy-region coupling) is a natural follow-on paper.

## 8. What DiSE adds, in one line

> **DiSE is the first prototype to combine adaptive SMT-driven
> stratification, closed-form axis-aligned mass on LIA regions, and
> anytime-valid certified intervals into a single algorithm for
> reliability estimation under structured discrete input
> distributions.**
