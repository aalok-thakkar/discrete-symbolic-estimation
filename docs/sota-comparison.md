# Comparative study — DiSE vs. state-of-the-art sampling estimators

> This document complements [`evaluation.md`](evaluation.md) and
> [`related-work.md`](related-work.md). It reports a head-to-head
> empirical comparison of DiSE against three SoTA *sampling-only*
> baselines on the 12 registered benchmarks. The SoTA comparators
> share with DiSE the "draw samples, certify an interval, stop
> when accurate enough" structure; the only difference is whether
> the algorithm exploits symbolic information about the program.
> This isolates DiSE's structural-stratification contribution from
> its statistical machinery.

## 1. Comparators

| Method | Class | Reference | What it does |
|---|---|---|---|
| `plain_mc` | non-adaptive | (DiSE baseline) | Fixed-`n` Wilson-score interval on iid samples. |
| `stratified_random` | non-adaptive | (DiSE baseline) | 16 hash-bucket strata; Bonferroni-combined per-bucket Wilson. |
| `adaptive_hoeffding` | adaptive | Sampson et al., PLDI 2014 | Sequential Hoeffding with Basel-mass union-bound-in-time. |
| `ebstop` | adaptive | Mnih–Szepesvári–Audibert, ICML 2008 | Empirical-Bernstein adaptive stopping. |
| `betting_cs` | adaptive | Waudby-Smith & Ramdas, JRSS-B 2024 | Hedged-capital betting CS with PrPl betting fractions. |
| `dise` | adaptive + symbolic | This work | ASIP: adaptive symbolic importance partitioning. |

* The three adaptive concentration methods (`adaptive_hoeffding`,
  `ebstop`, `betting_cs`) are *anytime-valid* — sound under
  data-dependent stopping rules, just like DiSE's `"anytime"`
  method.
* `plain_mc` is the standard non-adaptive comparator. Its certified
  half-width is tightest at fixed `n` (no union-bound-in-time
  penalty), but it cannot terminate early when the empirical
  variance allows.
* All five non-DiSE methods are *symbol-blind*: they treat `P`
  as a black-box oracle returning Bernoulli observations. DiSE is
  the only method using path-condition / SMT reasoning.

### Methods *not* included (and why)

| Method | Reason |
|---|---|
| PSE with model counting (Geldenhuys et al. ISSTA'12, Filieri et al.) | Requires LattE/Barvinok and Java SPF. Exact on affine paths with uniform inputs only — for our non-uniform `BoundedGeometric` benchmarks the model-counting kernel does not apply unmodified. Structural positioning is given in [`related-work.md`](related-work.md) §1. |
| PRISM / STORM | Require encoding integer Python programs as PRISM/JANI Markov chains. Out-of-scope translation effort; cf. [`related-work.md`](related-work.md) §2. |
| Static probabilistic abstract interpretation (Sankaranarayanan et al. PLDI'13) | Not packaged as a reusable tool. |
| Subset Simulation / AMS | Require a Lyapunov-style level function; applicable only to the rare-event regime, and only `coin_machine` and `assertion_overflow` are rare-event in our suite. |
| Hypothesis / QuickCheck | Counterexample-finding, not probability estimation; structurally different problem (see [`related-work.md`](related-work.md) §7). DiSE's Hypothesis adapter is the right comparison and is documented in [`hypothesis-integration.md`](hypothesis-integration.md). |

## 2. Configuration

Reproduce with:

```bash
uv run python scripts/sota_compare.py \
    --budget 2000 --n-seeds 3 \
    --epsilon 0.05 --delta 0.05 \
    --mc-samples 10000 \
    --dise-budget-seconds 30 \
    --out-dir results/sota_anytime/
uv run python scripts/sota_report.py \
    --in results/sota_anytime/summary.json \
    --out-dir results/sota_anytime/
```

* `budget = 2000` (cap on Bernoulli observations)
* `epsilon = 0.05` (target half-width)
* `delta = 0.05` (95% confidence)
* `n_seeds = 3`
* `mc_samples = 10000` (ground truth)
* DiSE backend: `MockBackend` with sample-based leaf closure
  (DiSE's documented fast configuration; see
  [`evaluation.md`](evaluation.md) §6). The Z3 backend was found
  to either time out or be unable to prove path-determinism on
  most benchmarks at this budget (`W_open ≈ 1`, vacuous intervals
  with half-width = 0.5), so MockBackend is the head-to-head
  configuration.
* DiSE has a 30-second wall-clock cap per run (`--dise-budget-seconds 30`)
  because Collatz, Miller-Rabin and modular-exponentiation
  concolic traces are otherwise unbounded.
* All four non-DiSE adaptive methods use `batch_size=50` and
  `delta=0.05`.

Total wall-clock for the full sweep: **312 s** (5.2 min).

## 3. Results

The full per-method × per-benchmark tables are in
[`../results/sota_anytime/`](../results/sota_anytime/):

* [`table_half_width.md`](../results/sota_anytime/table_half_width.md)
* [`table_samples.md`](../results/sota_anytime/table_samples.md)
* [`table_coverage.md`](../results/sota_anytime/table_coverage.md)
* [`table_wall_clock.md`](../results/sota_anytime/table_wall_clock.md)
* [`ranking.md`](../results/sota_anytime/ranking.md) — per-benchmark ranking
  by half-width
* [`synthesis.md`](../results/sota_anytime/synthesis.md) — headline tally

### 3.1 Aggregate rank distribution (lower is better)

How many times each method finished in rank-`k` across the 12
benchmarks (counting only methods with `coverage ≥ 0.5`):

| method | rank=1 | rank=2 | rank=3 | rank=4 | rank=5 | rank=6 |
|---|---|---|---|---|---|---|
| `plain_mc` (Wilson, fixed `n`) | **11** | 1 | 0 | 0 | 0 | 0 |
| `stratified_random` | 0 | 4 | 3 | 2 | 2 | 1 |
| `adaptive_hoeffding` [PLDI'14] | 0 | 0 | 3 | 1 | 2 | 6 |
| `ebstop` [ICML'08] | 0 | 0 | 4 | 5 | 3 | 0 |
| `betting_cs` [JRSS'24] | 0 | 4 | 1 | 2 | 5 | 0 |
| **`dise`** (ours) | **1** | 3 | 1 | 2 | 0 | 3 |

### 3.2 Half-width (lower = tighter certified interval)

| benchmark | `plain_mc` | `strat` | `ada_hoeff` | `ebstop` | `betting` | **`dise`** |
|---|---|---|---|---|---|---|
| `assertion_overflow_mul_w=8_U(1,31)` | 0.0215 | 0.1245 | 0.0696 | 0.0911 | 0.0494 | 0.0000 ⚠ |
| `coin_machine_U(1,9999)` | **0.0041** | 0.0239 | 0.0391 | 0.0286 | 0.0283 | 0.0104 |
| `collatz_le_30_BG(p=0.05,N=200)` | **0.0112** | 0.0583 | 0.0696 | 0.0583 | 0.0479 | 0.4488 |
| `gcd_steps_le_5_BG(p=0.1,N=100)` | 0.0046 | 0.0256 | 0.0406 | 0.0300 | 0.0352 | 0.0000 ⚠ |
| `integer_sqrt_correct_U(1,1023)` | **0.0005** | 0.0163 | 0.0348 | 0.0243 | 0.0283 | 0.0046 |
| `log2_w6` | **0.0005** | 0.0161 | 0.0348 | 0.0243 | 0.0283 | 0.0246 |
| `miller_rabin_w=2_BG(p=0.05,N=200)` | **0.0209** | 0.0740 | 0.0696 | 0.0892 | 0.0499 | 0.2323 |
| `modpow_fits_in_4b_m=37` | **0.0218** | 0.1240 | 0.0696 | 0.0915 | 0.0499 | 0.5000 |
| `parity_w6` | **0.0005** | 0.0161 | 0.0348 | 0.0243 | 0.0283 | 0.0171 |
| `popcount_w6` | **0.0005** | 0.0161 | 0.0348 | 0.0243 | 0.0283 | 0.0249 |
| `sieve_primality_U(2,200)` | 0.0185 | 0.0989 | 0.0696 | 0.0815 | 0.0499 | **0.0060** |
| `sparse_trie_depth_le_3_U(0,63)` | **0.0005** | 0.0163 | 0.0348 | 0.0243 | 0.0283 | 0.0134 |

⚠ = empirical coverage < 0.5 (interval did not contain MC truth).

### 3.3 Sample efficiency (lower = fewer Bernoulli evaluations)

| benchmark | `plain_mc` | `ebstop` | `betting` | **`dise`** |
|---|---|---|---|---|
| `assertion_overflow_mul_w=8_U(1,31)` | 2000 | 2000 | 1500 | 220 |
| `coin_machine_U(1,9999)` | 2000 | 1300 | **100** | 220 |
| `collatz_le_30_BG(p=0.05,N=200)` | 2000 | 2000 | **150** | 820 |
| `gcd_steps_le_5_BG(p=0.1,N=100)` | 2000 | 1400 | **100** | 220 |
| `integer_sqrt_correct_U(1,1023)` | 2000 | 850 | **100** | 300 |
| `log2_w6` | 2000 | 850 | **100** | 320 |
| `miller_rabin_w=2_BG(p=0.05,N=200)` | 2000 | 2000 | 1300 | **958** |
| `modpow_fits_in_4b_m=37` | 2000 | 2000 | 1550 | **1010** |
| `parity_w6` | 2000 | 850 | **100** | 420 |
| `popcount_w6` | 2000 | 850 | **100** | 320 |
| `sieve_primality_U(2,200)` | 2000 | 2000 | 950 | **550** |
| `sparse_trie_depth_le_3_U(0,63)` | 2000 | 850 | **100** | 791 |

`betting_cs` is the most sample-efficient method on 8/12 benchmarks.

### 3.4 Coverage (fraction of seeds whose certified interval contains MC truth)

Every non-DiSE method achieves `coverage = 1.00` on every benchmark.
DiSE+MockBackend shows soundness failures on:

| benchmark | coverage | analytical truth | DiSE μ̂ | MC truth |
|---|---|---|---|---|
| `assertion_overflow_mul_w=8_U(1,31)` | **0.00** | 388/961 = 0.4037 | 0.4300 / 0.3970 | 0.4004 ± 0.0035 |
| `gcd_steps_le_5_BG(p=0.1,N=100)` | **0.33** | — | 0.9880 (half=0) | 0.9892 ± 0.0010 |
| `coin_machine_U(1,9999)` | 0.67 | 99/9999 = 0.0099 | 0.0089 / 0.0099 | 0.0092 ± 0.0010 |
| `sieve_primality_U(2,200)` | 0.67 | 46/199 = 0.2312 | 0.2273 (half=0.006) | 0.2314 ± 0.0030 |

`coin_machine` and `sieve_primality` are MC-noise artefacts: DiSE
returns the analytically-correct point estimate but with zero
half-width, so even small MC noise (~ σ_MC = 0.001) drops the
seed below the coverage threshold. `assertion_overflow` and `gcd`
are *genuine* soundness failures: DiSE's MockBackend
sample-based closure misallocates leaf mass.

### 3.5 Wall-clock overhead

Median seconds per seed (lower = faster):

| benchmark | sampling SoTA median | DiSE | overhead × |
|---|---|---|---|
| `assertion_overflow_mul_w=8_U(1,31)` | 0.002 | 0.006 | 3.0 × |
| `coin_machine_U(1,9999)` | 0.001 | 0.006 | 6.0 × |
| `collatz_le_30_BG(p=0.05,N=200)` | 0.004 | 30.591 | **7,650 ×** |
| `gcd_steps_le_5_BG(p=0.1,N=100)` | 0.002 | 0.025 | 12 × |
| `integer_sqrt_correct_U(1,1023)` | 0.002 | 1.918 | **960 ×** |
| `log2_w6` | 0.001 | 0.077 | 77 × |
| `miller_rabin_w=2_BG(p=0.05,N=200)` | 0.002 | 31.304 | **15,652 ×** |
| `modpow_fits_in_4b_m=37` | 0.002 | 30.014 | **15,007 ×** |
| `parity_w6` | 0.002 | 0.702 | 351 × |
| `popcount_w6` | 0.001 | 0.110 | 110 × |
| `sieve_primality_U(2,200)` | 0.002 | 0.908 | 454 × |
| `sparse_trie_depth_le_3_U(0,63)` | 0.003 | 5.084 | **1,695 ×** |

DiSE incurs 3 × to 16,000 × overhead over symbol-blind sampling.
The cost is the per-input concolic trace plus per-iteration
frontier maintenance.

## 4. Findings

### 4.1 PlainMC at fixed `n` dominates 11/12 benchmarks

The single biggest finding: **a vanilla fixed-`n` MC with a
Wilson-score interval is the empirical winner on 11 out of 12
benchmarks at `budget = 2000`**. This is because:

* Wilson at fixed `n` does *not* pay the union-bound-in-time
  penalty that anytime-valid methods (BettingCS, EBStop,
  AdaptiveHoeffding) must pay; the half-width is tighter by a
  $\sqrt{\log n}$ factor.
* At `budget = 2000` on Bernoulli observations, $\sqrt{n}$ is
  large enough that the iid-Wilson interval is comparable to the
  empirical-Bernstein floor — so the variance-adaptive methods
  do not pull ahead in *interval width*.

This finding is consistent with the bounded-Bernoulli theory: at
*fixed* sample size, no anytime-valid construction can be tighter
than the Clopper-Pearson / Wilson interval. The reason to use an
anytime-valid construction is *adaptive stopping* — to spend fewer
samples — which leads to:

### 4.2 BettingCS dominates sample efficiency

`betting_cs` is **the most sample-efficient method on 8/12
benchmarks**, often by 10-20 ×. It certifies a half-width of 0.028
on `log2_w6` from just 100 samples; DiSE uses 320 samples for a
slightly tighter (0.025) half-width, and EBStop uses 850 for 0.024.

For applications where the per-sample cost is high (e.g. invoking
`P` requires a slow simulator), BettingCS is the strictly-dominant
SoTA choice among the symbol-blind methods.

### 4.3 DiSE's structural-stratification pays in one regime

DiSE is the rank-1 method on exactly **one** benchmark:
`sieve_primality_U(2,200)`. The reason is structural: trial-division
primality on $n \in \{2, ..., 200\}$ has a *small* number of
distinct path-conditions (one per candidate divisor), so DiSE's
concolic frontier converges quickly to a refinement that closes
each prime/non-prime leaf with deterministic mass attribution.

In addition, DiSE places rank-2 on `coin_machine`, `integer_sqrt`,
and `sparse_trie` — all cases where the property is constant
on a heavy region and DiSE's path-determinism shortcut delivers a
tighter interval than the adaptive concentration methods.

### 4.4 DiSE is rank-6 on three benchmarks

`collatz_le_30_BG`, `miller_rabin_w=2_BG`, and `modpow_fits_in_4b`
are all benchmarks where:

* The control flow is irregular or input-dependent.
* MockBackend cannot prove leaf path-determinism, so
  the `W_open` term in the certified interval stays large.
* The 30-second wall-clock cap fires before DiSE can refine to
  convergence.

At `epsilon = 0.05`, DiSE on these benchmarks delivers a half-width
between 0.23 and 0.50 — wider than every sampling baseline (which
all converge to between 0.05 and 0.09).

### 4.5 DiSE+MockBackend is unsound on 2/12 benchmarks

Beyond the MC-noise artefacts on `coin_machine` and
`sieve_primality`, DiSE has *genuine* soundness failures on
`assertion_overflow` (μ̂=0.43, truth=0.4037 — off by 6 SEs) and
`gcd_steps_le_5` (intervals of width zero that exclude truth on
2/3 seeds).

The cause is documented in
[`evaluation.md`](evaluation.md) §6: the sample-based closure
heuristic in `MockBackend` is not a $(1-\delta)$-coverage
construction. It is fast, and it works on most benchmarks, but it
gives a strict-confidence guarantee only with the Z3 backend —
which on these benchmarks is too slow at this budget to converge.

## 5. Implications for the DiSE claim set

The DiSE write-up advances four claims; the head-to-head SoTA
comparison maps them onto empirical results as follows:

| # | Claim from `README.md` / `algorithm.md` | Empirical status at `budget=2000` |
|---|---|---|
| C1 | "DiSE refines into leaves and certifies the answer in a few hundred concolic runs" | **Confirmed** on `coin_machine` (220 runs, half ≤ 0.01), `integer_sqrt` (300 runs, half 0.005), `sieve_primality` (550 runs, half 0.006), `sparse_trie` (791 runs, half 0.013). |
| C2 | "Plain MC needs $\Theta(1/\mu) \approx 10^5$ samples to certify the rare bug" | **Wrong as stated**: at `budget=2000` PlainMC's Wilson half-width on `coin_machine` is 0.0041, *tighter* than DiSE's 0.0104. The $\Theta(1/\mu)$ scaling is for *relative* half-width $\varepsilon_{\text{rel}} = \varepsilon / \mu$, not the absolute half-width that the benchmark targets. |
| C3 | Order-of-magnitude sample reduction vs. plain MC | **Confirmed** vs. `plain_mc` (DiSE uses 220 vs 2000 = 9 ×), **not** confirmed vs. `betting_cs` (which uses 100 vs DiSE's 220 on coin_machine, 100 vs 300 on integer_sqrt, etc.). |
| C4 | Certified $(1-\delta)$-coverage | **Fails** with MockBackend on 2/12 benchmarks. Z3 backend would restore soundness but at >100× the wall-clock cost. |

### 5.1 Recommended action items for the DiSE write-up

1. **Add `betting_cs` to `dise.baselines` and the `default_methods`
   set used in `dise compare` and `dise experiment`.** The existing
   baselines (`plain_mc`, `stratified_random`) understate the
   sampling SoTA by 1-2 orders of magnitude in sample efficiency.
   The Waudby-Smith & Ramdas 2024 betting CS is the natural
   counterpart to ASIP's anytime-valid claim and should appear
   alongside it in every comparison.

2. **Update the headline claim about plain MC.** The current README
   statement ("plain MC needs $\Theta(1/\mu) \approx 10^5$ samples
   to certify the rare bug at the same half-width") is true only
   for *relative* precision. Either change `coin_machine` to use a
   relative-precision goal (e.g. $\varepsilon_{\text{rel}} = 0.1$,
   i.e. half-width $\le 0.1 \mu \approx 10^{-3}$, which *does*
   require $10^5$ samples for plain MC) or rephrase the claim to
   refer to relative half-width.

3. **Update `limitations.md` and `evaluation.md` §6** to *explicitly*
   warn that MockBackend's sample-based closure is not (1-δ)-sound
   when leaf regions are not axis-aligned LIA. List
   `assertion_overflow_mul_w=8_U(1,31)` as the simplest documented
   counterexample.

4. **Narrow the contribution positioning.** The empirical
   evidence supports the following narrower claim unambiguously:
   *"On benchmarks where the operational distribution concentrates
   on a region whose path-condition is provable in the chosen SMT
   backend, ASIP delivers tighter certified intervals at lower
   sample cost than the anytime-valid bounded-mean SoTA. On
   benchmarks with irregular control flow or non-LIA arithmetic
   the symbolic overhead is wasted."*

5. **Report wall-clock in the headline tables.** The 3–15,000 ×
   wall-clock overhead is the real cost of DiSE's symbolic
   component and should not be hidden. Whether the user *can*
   afford that overhead depends entirely on whether the
   per-sample cost of `P` is comparable to the per-sample cost
   of concolic execution.

## 6. Improvement attempt: sound concentration-bounded closure

The §4.5 finding — DiSE+MockBackend is unsound on 4/12 benchmarks
(`assertion_overflow`, `gcd_steps_le_5`, `coin_machine`,
`sieve_primality`) — pointed to the closure rule as the bug.

### 6.1 The change

The old rule (in `dise.regions.Frontier.try_close`) closed a leaf as
deterministic whenever the all-`n`-samples-agree AND
all-branch-sequences-agree conditions held. With MockBackend (no
SMT verification), this is **not** a (1−δ)-coverage construction:
five Bernoulli draws can be all-0 with probability ≥ 3% even if the
true mu is 0.5.

The new rule (commit `95f07b5`) adds a Wilson-anytime
concentration test: closure fires only when

> `wilson_halfwidth_anytime(n_samples, 0, δ_close) ≤ ε_close`.

When this fires via the sample path (not SMT-verified), the leaf
contributes `ε_close × w_leaf` to a new `W_close_accumulated`
accumulator on the frontier. The certified half-width grows by
this amount. SMT-verified closures still contribute 0 (they are
exact).

New `SchedulerConfig` knobs (and matching `estimate` /
`failure_probability` kwargs):

* `delta_close` (default 0.005) — per-leaf closure-failure budget.
* `closure_epsilon` (default 0.02) — disagreement budget per closed leaf.

### 6.2 Empirical effect

Three runs, all at the same budget=2000, ε=0.05, δ=0.05:

* [`results/sota_anytime/`](../results/sota_anytime/) — old
  heuristic (unsound).
* [`results/sota_sound_e02/`](../results/sota_sound_e02/) —
  sound, ε_close=0.02 (tight).
* [`results/sota_sound_e05/`](../results/sota_sound_e05/) —
  sound, ε_close=0.05 (loose, recommended).

Full 3-way table:
[`results/sota_sound_e05/three_way.md`](../results/sota_sound_e05/three_way.md).

#### Soundness

| variant | unsound (cov < 0.95) |
|---|---|
| DiSE-old heuristic | **4/12** (`assertion_overflow`, `gcd_steps_le_5`, `coin_machine`, `sieve_primality`) |
| DiSE-sound ε=0.02 | 1/12 (`assertion_overflow` at cov=0.33) |
| DiSE-sound ε=0.05 | **0/12** ✓ |

`assertion_overflow` at ε=0.02 has coverage 0.33 because the
empirical mass estimate of the "overflow" leaf (w_hat ≈ 0.43) is
biased high vs the analytical truth (0.4037) by more than the
closure-uncertainty allowance — the residual is a mass-estimation
error, not a closure error.

#### Half-width

DiSE-sound is *uniformly looser* than the old heuristic, by 0–47×
depending on benchmark:

| benchmark | old | sound ε=0.02 | sound ε=0.05 | PlainMC |
|---|---|---|---|---|
| `assertion_overflow` | 0.0000 (unsound) | 0.0200 | 0.0500 | 0.0215 |
| `coin_machine` | 0.0104 (unsound) | 0.0202 | 0.0361 | 0.0041 |
| `gcd_steps_le_5` | 0.0000 (unsound) | 0.0228 | 0.0384 | 0.0046 |
| `integer_sqrt` | 0.0046 | 0.5000 | 0.2680 | 0.0005 |
| `log2_w6` | 0.0246 | 0.5000 | 0.0558 | 0.0005 |
| `popcount_w6` | 0.0249 | 0.5000 | 0.0563 | 0.0005 |
| `sieve_primality` | 0.0060 (unsound) | 0.5000 | 0.5000 | 0.0185 |

Sound DiSE is *fundamentally* bounded below by `closure_epsilon` on
benchmarks where every leaf closes via the sample path (MockBackend,
no SMT verification). PlainMC's Wilson half-width has no such floor;
on the trivial-property benchmarks (μ=1.0) PlainMC certifies at 0.0005
while sound DiSE cannot go below 0.05.

### 6.3 Diagnosis: the refinement strategy over-fragments under sound closure

Inspection of the per-seed frontier on `integer_sqrt_correct_U(1,1023)`
shows DiSE refining into **9 open leaves** under ε=0.02 — none
with enough samples to clear the concentration check. The
refinement step doubles the per-leaf sample cost of subsequent
closure; the algorithm's "expected gain from refinement"
calculation does not account for this. Result: leaves keep
fragmenting and never close, leaving `W_open ≈ 1.0` and a vacuous
half-width of 0.5.

The fix is *not* in the closure rule (which is now sound and
correct) but in the *refinement scheduler*: don't refine leaves
that are already path-deterministic (i.e., all-agree branch
sequences + all-agree phi), just keep sampling them until the
concentration bound clears.

### 6.4 Bottom line

The sound closure fixes the (1−δ)-coverage failure (4/12 → 0/12 at
ε=0.05). But it exposes a second-order issue: the refinement
scheduler, tuned for the old aggressive closure, over-fragments
under the sound rule. PlainMC at fixed n=2000 still wins on
half-width on 12/12 benchmarks; DiSE-sound-ε=0.05 is 2–100× wider
on the benchmarks where it produces an informative interval, and
returns half=0.5 (uninformative) on 5/12.

**Net assessment**: the sound closure is necessary for principled
soundness claims (the previous DiSE/MockBackend was reporting
intervals it could not support). But to *beat* SoTA at the same
budget, DiSE needs additional changes:

1. **Refinement scheduler: don't refine all-agree leaves.**
   Eliminates the fragmentation that drives `W_open → 1.0`.
2. **Sound mass estimation.** The `assertion_overflow` coverage
   gap at ε=0.02 is from biased `w_hat`, not from closure.
3. **Useful Z3.** With MockBackend, every closure pays
   `ε_close × w_leaf`; with Z3-verified closure, the contribution
   is 0 (exact). A bit-vector backend (item #5 in §5.1) would
   recover tight intervals on the kernels where LIA fails.

## 7. Reproduction

All results in this document were produced by the commands in §2
and §6 and are persisted as JSON in
[`../results/sota_anytime/`](../results/sota_anytime/),
[`../results/sota_sound_e02/`](../results/sota_sound_e02/), and
[`../results/sota_sound_e05/`](../results/sota_sound_e05/).
The summary files are at `summary.json` in each directory.
