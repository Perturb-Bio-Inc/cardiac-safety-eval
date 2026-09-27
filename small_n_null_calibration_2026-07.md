# Why the small-benchmark permutation null sits below 0.50, measured

2026-07-20. Scripts: `small_n_null_probe.py` (synthetic probe), `small_n_null_recheck.py`
(sensitivity re-check). Data: `small_n_null_probe.json`, `small_n_null_recheck.json`.
`eval_core.py` was not modified; both scripts call its
`permutation_null` / `grouped_oof_scores` unchanged.

## The claim under test

The README caveat said the small-benchmark permutation null "centers slightly below 0.50 (an
overfitting-under-shuffle effect, not leakage)"
(Source: read the July 2026 version of README.md → Caveats, first bullet, pre-edit text).
The shift is large, not slight: `multichannel_logreg` on CiPA has null_mean 0.3741 and every
fitted engine-v0 candidate sits between 0.355 and 0.399
(Source: read loop_summary.json 2026-07-20 → per-candidate permutation.null_mean).
The parenthetical was never measured.

**Result: the "overfitting" half of the explanation is wrong.** The shift is not driven by model
flexibility; it runs the other way. It is a leave-one-group-out artifact: out-of-fold scores carry
a per-fold offset that is anti-correlated with that fold's own label composition, so every
cross-fold comparison is biased against the truth. No leakage, and the p-values as computed are
still valid.

## What was run

Features drawn iid normal, independent of labels, so there is nothing to find and any departure
from 0.50 is a property of the estimator. Labels and fold structure taken from the real
benchmarks. Note: the permutation null runs on the **dev** split, so the real structures are
n=12 in 4 mechanism groups (CiPA-28, 4/4/4 ordinal) and n=22 in 5 groups (engine-v0, 11/11), not
the n=28 / n=33 full-benchmark sizes named in the task
(Source: ran benchmarks.CiPABenchmark().dev() / EngineV0Benchmark().dev() 2026-07-20 → (12,4) and (22,4)).
2,000 shuffles per cell; standard errors below are on the null mean.

### A. Flexibility, at the real fold structures

| model | fitted params | CiPA dev (n=12, k=4) | engine-v0 dev (n=22, k=5) |
|---|---|---|---|
| fit-free single column | 0 | 0.5012 ± 0.0042 | 0.4986 ± 0.0028 |
| fit-free 3-column sum | 0 | 0.4989 ± 0.0041 | 0.4976 ± 0.0029 |
| logreg, 1 feature | 2 | **0.3311** ± 0.0050 | **0.3544** ± 0.0039 |
| logreg, 2 features | 3 | 0.3494 ± 0.0053 | 0.3933 ± 0.0041 |
| logreg, 3 features | 4 | 0.3744 ± 0.0055 | 0.4228 ± 0.0040 |
| logreg, 4 features | 5 | 0.3825 ± 0.0054 | 0.4322 ± 0.0040 |
| logreg, 8 features | 9 | 0.4062 ± 0.0054 | 0.4510 ± 0.0038 |
| logreg, 16 features | 17 | 0.4087 ± 0.0052 | 0.4631 ± 0.0038 |

Two things fall out. A scorer that does not look at labels sits at 0.50 on the nose, which is what
the stated explanation predicts and is the part it gets right. But the shift is **deepest at the
simplest fitted model and shrinks monotonically as parameters are added**, the exact opposite of
an overfitting effect. The harness's own large benchmark makes the same point from the other end:
`fp_logreg` on DICTrank fits 1,025 parameters on ~700 dev molecules and its null mean is 0.4970
(Source: read loop_summary.json 2026-07-20 → dictrank_fp/fp_logreg_l2_3.0 permutation.null_mean 0.497).
A thousand parameters cost 0.003; two parameters at n=12 cost 0.17.

### B. Size of the shift vs n (k=4 folds, 1/3 prevalence, 4-param logreg)

| n | 12 | 16 | 22 | 28 | 40 | 60 | 100 | 200 |
|---|---|---|---|---|---|---|---|---|
| null mean | 0.3890 | 0.3988 | 0.4208 | 0.4208 | 0.4364 | 0.4539 | 0.4577 | 0.4710 |

Decays toward 0.50 roughly like 1/n, still 0.03 short at n=200 with only 4 folds.

### C. Size of the shift vs fold structure (n=28 fixed)

| folds k | 2 | 3 | 4 | 7 | 14 | 28 (leave-one-out) |
|---|---|---|---|---|---|---|
| fold size | 14 | 9.3 | 7 | 4 | 2 | 1 |
| null mean | 0.4475 | 0.4282 | 0.4140 | 0.3991 | 0.3886 | 0.3815 |

At fixed n the shift deepens as folds get smaller. Leave-one-out is the worst case.

### D. Where the shift lives: within-fold vs between-fold pairs

Split the AUC into positive/negative pairs scored by the *same* fold model and pairs scored by
*different* fold models, on noise features under 2,000 shuffles:

| benchmark structure | within-fold AUC | between-fold AUC |
|---|---|---|
| CiPA dev (n=12, k=4) | 0.5044 ± 0.0064 | **0.3565** ± 0.0054 |
| engine-v0 dev (n=22, k=5) | 0.4993 ± 0.0042 | **0.4034** ± 0.0039 |

This is the mechanism, isolated. Pairs compared inside one fold are unbiased at 0.50. The entire
deficit lives in cross-fold comparisons.

## The cause

Under leave-one-group-out, each fold's model is fit on the *complement* of that fold. The number
of positives is fixed, so a fold that happens to draw more positives leaves fewer for its own
training set, and the model it gets back is calibrated to a lower positive rate and returns
systematically lower scores for that fold. Positive-rich folds get pushed down, positive-poor
folds get pushed up, and every cross-fold pair inherits that anti-correlation. B and C are the
two knobs the mechanism predicts: the deviation of a fold's positive rate from the global rate
shrinks with n (B) and grows as folds get smaller (C). D confirms it directly, and A explains why
flexibility *helps*: more features means more variance in the fitted direction, which dilutes the
shared intercept-driven offset that carries the bias. A fit-free scorer has no offset at all, so
it sits at 0.50.

This is a known property of cross-validated AUC on small samples (the pooled-OOF, or "AUC of
pooled predictions", bias), not anything specific to this harness. It is not leakage: leakage
pushes the null *above* 0.50.

## Practical correction

**None to `perm_p` itself.** The observed statistic and every null draw are produced by the same
pipeline on the same folds, so the permutation test is exact under label exchangeability whatever
the null centres at. The shift is already absorbed by construction. Do not "correct" p-values.

**The gate becomes permissive relative to a naive 0.50 reading, and that reading is the error.**
Because the null shifts down, its 95th percentile shifts down with it, so a candidate can clear
the as-run gate while failing a 0.50-centred one. Arithmetic, from re-running each null at 2,000
shuffles and re-scoring against the distribution shifted up by (0.50 − null_mean)
(Source: ran small_n_null_recheck.py 2026-07-20 → small_n_null_recheck.json):

| benchmark | candidate | dev AUC | null mean | p / p95 as run | p / p95 re-centred | gate flips? |
|---|---|---|---|---|---|---|
| engine-v0 | `all_features` | 0.9752 | 0.3942 | 0.0005 / 0.661 | 0.0010 / 0.767 | no (pass → pass) |
| engine-v0 | `magnitude_only` | 0.8760 | 0.3625 | 0.0005 / 0.636 | 0.0095 / 0.774 | no (pass → pass) |
| engine-v0 | `panel_no_magnitude` | 0.9008 | 0.3777 | 0.0005 / 0.620 | 0.0020 / 0.742 | no (pass → pass) |
| engine-v0 | `panel_only` | 0.6942 | 0.3471 | 0.0025 / 0.562 | 0.0670 / 0.715 | **yes (pass → fail)** |
| CiPA | `herg_only_baseline` | 0.8438 | 0.5039 | 0.038 / 0.813 | 0.027 / 0.809 | no (pass → pass) |
| CiPA | `kernik_only` | 0.9062 | 0.5002 | 0.017 / 0.813 | 0.009 / 0.812 | no (pass → pass) |
| CiPA | `multichannel_logreg` | 0.5625 | 0.3765 | 0.236 / 0.781 | 0.382 / 0.905 | no (fail → fail) |
| CiPA | `multichannel_plus_kernik` | 0.7500 | 0.3884 | 0.088 / 0.781 | 0.167 / 0.893 | no (fail → fail) |

One candidate out of eight is reference-dependent: `panel_only` (dev 0.694, the transcriptomic
panel score with magnitude left in) clears the as-run gate at p=0.0025 and would not clear a
0.50-centred one at p=0.067. It is not a headline result: its locked AUC is 0.533, below the
0.70 magnitude baseline, so it fails certification on the baseline gate regardless
(Source: read loop_summary.json 2026-07-20 → enginev0/panel_only locked_auc 0.533). Worth
knowing that the dev-side evidence for a magnitude-free panel signal is thinner than p=0.0025
suggests to a reader who assumes chance is 0.5.

**Reading rule that follows: on the small benchmarks, 0.50 is not the chance level for a fitted
model's OOF AUC.** Chance is `null_mean`. A dev AUC of 0.60 against a null centred at 0.38 is a
real separation from chance; the same 0.60 read against 0.50 is a different and wrong story. Any
prose that quotes a small-benchmark dev AUC should quote the null mean beside it. Locked-split
AUCs are unaffected: the locked set is scored by a single model fit on all of dev, so every pair
is a within-fold pair, which probe D puts at 0.50.

## The two externally cited results, re-expressed

**engine-v0 retro-validation (locked 0.80, dev p=0.001): nothing moves.** The locked 0.80 is a
single fit-on-dev, score-once number with no cross-validation in it, so the artifact cannot touch
it. The dev p survives re-centring at p=0.0010 with dev AUC 0.9752 far above even the re-centred
p95 of 0.767. The conclusion in the README (mostly a transcriptional-magnitude effect with a small
increment beyond it) stands unchanged, and the increment claim rests on `all_features` locked 0.80
vs `magnitude_only` locked 0.70, both locked numbers, both unaffected.

**CiPA acceptance negative: nothing moves.** `multichannel_logreg` fails the gate as run
(p=0.236) and fails harder re-centred (p=0.382); `multichannel_plus_kernik` likewise (0.088 →
0.167). The two CiPA candidates that do certify, `herg_only_baseline` and `kernik_only`, are
fit-free column scorers whose nulls sit at 0.504 and 0.500, so they never had the artifact. The
finding that multichannel logistic regression does not beat hERG-alone on this panel is if
anything strengthened.

Both externally cited numbers are safe. What was wrong is the stated reason for the shift, not the
results that sit near it.

## Residual limits

- The probe uses Gaussian noise features. Real features are correlated and non-normal; that
  changes the size of the shift, not its sign or its mechanism (D is structural).
- Probes B and C use a synthetic balanced-block fold assignment, not curated mechanism groups.
  Curated groups can correlate with labels, which would add a separate effect on top of this one.
- The 1/n decay in B is eyeballed from eight points, not fitted. Do not quote a functional form.
