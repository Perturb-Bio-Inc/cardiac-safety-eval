# What delta could the variant × drug benchmark actually have detected?

**Date:** 2026-07-20 · **Verdict: (b) the benchmark cannot tell.**
Scope: the `variant_tdp` benchmark in an internal audit, the paired comparison
`variant_conditioned` vs `healthy_only` on the same drugs, both splits.

## Answer in one paragraph

At its actual size the benchmark had **~4% power to detect the largest AUC gain that is
arithmetically possible on it** (locked split), and ~2% on dev. The observed locked delta of
+0.0104 had essentially zero power. So "variant-conditioning does not beat the healthy-cell
model" is not a negative result about variant conditioning; it is a statement that the test
was silent. The wet-lab justification survives, but its logic changes: the wet data is
needed **because the simulation cannot answer the question**, not because the simulation
answered it in the negative.

## What the benchmark is

| | dev | locked |
|---|---|---|
| n drugs | 12 | 16 |
| class counts (Low/Int/High) | 4 / 4 / 4 | 5 / 7 / 4 |
| positives (High) vs rest | 4 vs 8 | 4 vs 12 |
| pos×neg pairs | 32 | 48 |
| **AUC granularity (one pair reversal)** | **0.0312** | **0.0208** |

`[Checked: ran VariantBenchmark().dev()/.locked() via ~/.venvs/myokit/bin/python 2026-07-20 →
dev n=12 pos=4 neg=8, locked n=16 pos=4 neg=12; class counts [4,4,4] and [5,7,4]]`

AUC here is a rank statistic over 48 (locked) or 32 (dev) pairs. It cannot take a value between
multiples of 1/48. That single fact bounds everything below.

## 1. Locked split (16 drugs)

Observed: `healthy_only` 0.9271, `variant_conditioned` 0.9375, **delta +0.0104**
`[Checked: recomputed both score vectors from models.VARIANT_REGISTRY, fit on dev, scored once
on locked 2026-07-20 → 0.9271 / 0.9375, matching loop_summary.json]`.

+0.0104 is **half a pair reversal**. It is a tie being broken, not a drug being re-ranked.

**Headroom.** The healthy baseline is already at 0.9271, so the maximum delta any model can
show on this panel is **+0.0729** (perfect separation). Deltas of +0.10 or +0.15 are not
merely undetected here, they are unconstructible `[Checked: computed 1 − 0.9271 on the locked
score vector 2026-07-20 → 0.0729 ceiling; only 3.5 discordant pairs of 48 exist to fix]`.

**Paired bootstrap on the real score vectors** (20,000 drug resamples, both models scored on
the identical resample, reusing `eval_core.bootstrap_ci`'s resampling scheme):

| | value |
|---|---|
| SE of the paired delta | 0.0173 |
| 95% percentile CI of delta | [0.0000, 0.0597] |
| resamples with delta exactly 0 | 58.7% |
| bootstrap correlation of the two AUCs | 0.972 |

`[Checked: paired bootstrap n=20000, seed 0, on locked score vectors 2026-07-20 → SE 0.0173,
CI [0.0000, 0.0597], 58.7% zero-delta, AUC corr 0.972]`

Taken at face value that SE implies MDE₈₀ = 2.80 × 0.0173 = **0.0485**, which would sit just
inside the 0.0729 ceiling. **That number is misleading and should not be quoted.** The SE is
small only because these two particular models rank the 16 drugs almost identically (corr 0.972;
three of five resamples give a delta of exactly zero). It is the SE *conditional on the two
models agreeing*, not the SE that would obtain if a variant model actually ranked the drugs differently.

**The right calculation** replaces the conditional SE with a simulation. Construct a
hypothetical variant model that truly beats healthy by k pair reversals on this population,
then draw fresh 16-drug panels from that population, run the same paired bootstrap inside each,
and count how often the 95% CI excludes zero:

| true delta | pairs fixed | power at n=16 |
|---|---|---|
| +0.0312 | 1.5 | 0.005 |
| +0.0521 | 2.5 | 0.018 |
| **+0.0729 (the ceiling)** | 3.5 | **0.039** |
| observed +0.0104 | 0.5 | ~0.000 |

`[Checked: simulation R=800 panels × B=400 paired bootstrap resamples, seed 1, on the real
locked score vectors 2026-07-20 → power 0.005 / 0.018 / 0.039 / 0.000]`

**A model that classified these drugs perfectly would have been called "no improvement" 96 times
out of 100.**

## 2. Dev split (12 drugs)

The README leans on the dev drop as the reason to discount the nominal locked win. The dev split
is weaker still.

Observed: `healthy_only` 0.9062 (OOF, mechanism-grouped), `variant_conditioned` 0.8125,
**delta −0.0938** `[Checked: grouped_oof_scores on dev, seed 0, 2026-07-20 → 0.9062 / 0.8125]`.

Paired bootstrap: SE 0.1072, 95% CI **[−0.3704, 0.0000]**, 0.0% of resamples positive, 37.4%
exactly zero `[Checked: paired bootstrap n=20000, seed 0, dev OOF vectors 2026-07-20]`.

Ceiling here is +0.0938 (3 discordant pairs of 32). Power under the same simulation:

| true delta | power at n=12 |
|---|---|
| +0.0312 | 0.001 |
| +0.0625 | 0.001 |
| **+0.0938 (the ceiling)** | **0.020** |
| observed −0.0938 | 0.024 |

`[Checked: simulation R=800 × B=400, seed 1, dev OOF vectors 2026-07-20]`

Two things follow. First, the dev delta of −0.0938 is itself detected only 2.4% of the time,
so the dev "drop" is as uninformative as the locked "win": its CI runs to −0.37 and touches
zero. Second, the whole dev-vs-locked discrepancy the README treats as evidence of instability
is what two coin flips look like. Neither split constrains the answer.

## 3. Verdict, and what would settle it

**(b) the benchmark cannot tell.** Not (a). The evidence is consistent with variant conditioning
being useless, being worth every AUC point available on this panel, or anything between. Both
splits are underpowered against their own arithmetic ceiling.

**Threshold worth caring about: ΔAUC = +0.05.** Reason: one pos/neg pair reversal on a panel of
this size is 0.021–0.031, so a gain under ~0.05 is a single tie-break or a single mislabeled
reference drug; +0.05 is the smallest gain that survives one label error and re-ranks more than
one drug.

**Drugs needed.** Simulating fresh panels of increasing size from the locked population at the
largest constructible delta (+0.0729):

| n drugs | power |
|---|---|
| 32 | 0.288 |
| 40 | 0.533 |
| 48 | 0.740 |
| 56 | 0.847 |
| 64 | 0.922 |

`[Checked: simulation R=600 × B=350, seed 5, locked score vectors, class balance preserved by
resampling 2026-07-20 → 80% power crossed between n=48 and n=56, ≈52 drugs]`

Dev-balance equivalent at +0.0938: 0.557 at n=48, 0.782 at n=64, 0.927 at n=80, so ≈66 drugs
`[Checked: same simulation, seed 5, dev OOF vectors 2026-07-20]`.

So: **~52 drugs to detect a +0.073 gain, and ~110 drugs to detect the +0.05 threshold**, the
latter by the (0.073/0.05)² = 2.13 variance scaling `[Not independently verified: extrapolation from the
measured n=52 point, not simulated directly; a delta of exactly +0.05 is not constructible on a
16-drug panel]`. Class balance matters as much as n: the binding constraint is 4 positives, not
16 drugs. A 52-drug panel with 13 High-risk compounds is the target, not 52 drugs with 4.

Both numbers are far outside CiPA-28. Answering this question in silico means a larger labelled
TdP reference set, not a better model on this one.

## 4. What this does to the wet-lab argument

It does not weaken it. It re-bases it. The old sentence ("no clear advantage in EP simulation") reads
as a measured negative and invites the obvious follow-up: if simulation says no, why would wet
data say yes? The correct sentence is that EP simulation on the available reference panel is
**silent** on the variant advantage, at any effect size that could exist on that panel. The wet
wet multimodal data is what makes the question answerable, both because it measures modalities the AP
model does not have and because it is the route to a labelled panel large enough to carry a test.

## Reproduce

`~/.venvs/myokit/bin/python` with `PYTHONPATH=science/cardiac_safety_loop`. Score vectors come
from `models.VARIANT_REGISTRY` and `eval_core.grouped_oof_scores`; the paired bootstrap resamples
drugs with replacement and scores both models on the identical resample. The power simulation
draws panels from the observed score population and runs the same paired bootstrap inside each.
Seeds as tagged above.
