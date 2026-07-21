# Is our DICTrank locked AUC 0.67 a good number? Comparator + in-house split-scheme test

2026-07-20. Answers the open question on result E1: 0.67 was carried with no comparator, and the working assumption was that our scaffold-disjoint split explained the gap to published DICTrank models.

**Headline: the scaffold split does not explain the gap. It is worth about +0.02 AUC, not the +0.17 that would close the distance to the best published number. The gap is the feature space.** We run a 1024-bit Morgan fingerprint alone; the published 0.84 comes from 1,038 physicochemical descriptors. E1 should be stated as evidence the evaluation harness is calibrated, never as evidence the model is competitive.

---

## 1. Published DICTrank results

| Study | Metric + value | n | Split scheme | Features | Leakage control described |
|---|---|---|---|---|---|
| JCIM 2024, "Insights into Drug Cardiotoxicity from Biological and Chemical Data: The First Public Classifiers for FDA DICTrank" (doi 10.1021/acs.jcim.3c01834) | External test AUC **0.84** (AUCPR 0.93) Mordred descriptors; **0.83** 6-model ensemble; **0.77** CELLSCAPE predicted targets; **0.77** MOA+Cmax | 1,020 (742 toxic / 278 nontoxic) | **Random hold-out**, 90 compounds (8.8%), constrained to compounds with complete annotation across all feature spaces, DICT-category balance matched | Morgan 2048-bit, Mordred (1,038 2-D physicochemical), MOA (264), CELLSCAPE targets (1,893), Cell Painting (1,783), L1000 (978), GO (4,438), Cmax | Partial. Similar compounds in the external test set excluded from v-NN profile imputation ("to avoid information leaks"). Applicability domain assessed by binning Tanimoto similarity 0.0–1.0 in five bins; Mordred models degrade on structurally dissimilar compounds | `[Checked: fetched PMC10900289 2026-07-20 → "we randomly selected 90 compounds (8.8% of the data set, 65 cardiotoxic and 21 nontoxic)"; "Mordred descriptors evaluated on the 90 compounds held-out test set (AUC: 0.84, AUCPR: 0.93)"]` |
| Chem Res Toxicol 2025, "DICTrank Is a Reliable Dataset for Cardiotoxicity Prediction Using Machine Learning Methods" (PMID 40146530) | AUC values **not obtainable**: absent from the abstract, full text paywalled. Qualitative result only: LR and XGBoost best of five methods; no association between model complexity and performance; performance varies by therapeutic category | not stated in abstract | **Temporal**: trained on drugs approved on or before 2005, tested on drugs approved after | Descriptor classes named as "structural and topological", "polarizability", "electronegativity" | Temporal split is itself the control; nothing further in the abstract | `[Checked: fetched pubmed.ncbi.nlm.nih.gov PMID 40146530 2026-07-20 → abstract verbatim: "models were trained on drugs approved before and within 2005 to predict the DICT risk of those approved thereafter"; no numeric AUC in abstract]` |

**Row dropped.** A "0.79 AUC on validation / 0.66 on unseen drugs across six cardiotoxicity types" figure recurs in search summaries and is often attached to DICTrank. It could not be traced to a specific paper, and the surrounding description (1,131 drugs, 9,933 transcriptional samples, six cardiotoxicity types) does not match the DICTrank release. **Do not cite it.** Recorded here so a future pass does not re-chase it. `[unconfirmed, attribution unresolved, deliberately excluded]`

**What is missing from the literature.** No published DICTrank paper reports a **Morgan-fingerprint-only external AUC** for the overall toxic-vs-nontoxic call. The JCIM paper includes Morgan 2048 among its feature spaces but only publishes per-feature-type numbers for Mordred, targets, MOA and the ensemble. So there is no direct head-to-head against our exact model class. That cuts both ways: no one has published a number we clearly underperform on matched features, and we cannot claim parity either. `[Checked: fetched PMC10614794 + PMC10900289 2026-07-20 → "Morgan fingerprints ... lack isolated performance reporting"]`

---

## 2. In-house measurement: what does our scaffold split actually cost?

Rather than assume the penalty, we measured it. `DICTrankBenchmark` now takes `split="random"`, which swaps the Bemis-Murcko scaffold grouping for molecule-level random assignment and plain random k-fold, holding everything else fixed: same 996 drugs, same 747 dev / 249 locked, same 5 folds, same seed, same candidate registry. It certifies through the normal `run_loop.py` path as benchmark id `dictrank_fp_random`; nothing was hand-written into `leaderboard.jsonl`.

**Seed-0 certified runs** (both in `leaderboard.jsonl`, `fp_logreg_l2_3.0`):

| | dev AUC (grouped OOF) | locked AUC | locked 95% CI | naive baseline | permutation p | null mean |
|---|---|---|---|---|---|---|
| scaffold-disjoint (`dictrank_fp`) | 0.7138 | **0.6669** | 0.5867–0.7437 | 0.5412 | 0.005 | 0.497 |
| random (`dictrank_fp_random`) | 0.6883 | **0.7448** | 0.6733–0.8158 | 0.5044 | 0.005 | 0.494 |
| delta (random − scaffold) | −0.0255 | **+0.0779** | | | | |

`[Checked: ran run_loop.py dictrank dictrank_random 2026-07-20 (run_ts 2026-07-20T00:09:16, after the fold-ordering fix below) → leaderboard.jsonl fp_logreg_l2_3.0: dictrank_fp dev 0.7138 locked 0.6669 null_mean 0.4970; dictrank_fp_random dev 0.6883 locked 0.7448 CI [0.6733, 0.8158] perm p 0.005 null_mean 0.4941]`

The dev column here was restated on 2026-07-20 after the fold-ordering fix in §4; the earlier values (0.7104 / 0.6797) were one draw of a fold assignment that varied per interpreter. Locked AUCs, the 8-seed table, and every conclusion below are unchanged, because they never used dev folds.

**The seed-0 delta is not the answer.** The two locked CIs overlap across most of their range, and a 249-drug locked set is a single noisy draw. Re-running the locked evaluation across 8 seeds under both schemes (`dictrank_split_scheme_run.py` → `dictrank_split_scheme_2026-07-20.json`):

| | mean locked AUC | SD across seeds | range of paired delta |
|---|---|---|---|
| scaffold-disjoint | **0.6996** | 0.0303 | |
| random | **0.7184** | 0.0304 | |
| paired delta (random − scaffold) | **+0.0189** | 0.0390 | −0.0454 to +0.0779, random higher in **6 of 8** seeds |

`[Checked: ran dictrank_split_scheme_run.py 2026-07-20 → summary {"scaffold_mean": 0.6996, "scaffold_sd": 0.0303, "random_mean": 0.7184, "random_sd": 0.0304, "delta_mean": 0.0189, "delta_sd": 0.039, "n_seeds_random_higher": 6}]`

Two things fall out of this, both uncomfortable and both worth knowing before a reviewer says them:

1. **The scaffold penalty is real but small: about +0.02 AUC, with a sign that flips in 2 of 8 seeds.** It is not a +0.08 effect and it is nowhere near the +0.17 needed to reach the published 0.84. The "our split is harder" defense is directionally true and quantitatively almost irrelevant.
2. **Our headline 0.67 is a low draw from our own distribution.** The 8-seed scaffold mean is 0.700 with SD 0.030; seed 0 sits about one SD below it, and seed 0's +0.078 delta is the single largest of the eight. Reporting 0.67 with no dispersion overstates the precision of a single 249-drug locked set.

---

## 3. Verdict

Once split scheme is held constant, **0.67 is not competitive with the best published DICTrank number, and the split does not excuse it.** Matched-condition comparison puts our random-split model at 0.718 (8-seed mean) against a published random-split 0.84, and the reason is the feature space, not the evaluation protocol: we use a 1024-bit Morgan fingerprint and nothing else, while 0.84 comes from 1,038 Mordred physicochemical descriptors. Even the published paper's biological feature spaces (predicted targets, MOA + Cmax) land at 0.77, above us. The one comparison that would settle it, Morgan-fingerprint-only external AUC on DICTrank, is not published by anyone. So the correct claim is narrow and defensible: our harness produces a leakage-free, permutation-calibrated number on ~900 drugs, and that number happens to be modest because the features are deliberately minimal. E1 has never been a model-quality claim, and this makes it unsafe to read as one.

**Most likely reviewer objection:** *"Published classifiers on this exact dataset reach AUC 0.84. You report 0.67. Why should I trust the rest of your evaluation?"*

**Can we answer it?** Yes, and better now than before. The answer is that 0.67 and 0.84 are not the same experiment: theirs is a 1,038-descriptor physicochemical model on a random 90-compound hold-out, ours is a fingerprint-only model on a scaffold-disjoint 249-drug locked set, and we have measured that the split accounts for roughly 0.02 of the difference and the features for the rest. What E1 certifies is the machinery: permutation null centered at 0.497 on 996 drugs, a random canary that fails every gate, a locked set scored exactly once. A reviewer who wants a higher number is asking for a richer feature space, which is a day of work rather than a research risk. The part this does not settle is precision: the seed-to-seed SD on the scaffold split is 0.03, so a single seed's locked AUC (0.6669) understates the spread. E1 is therefore reported as the 8-seed scaffold mean, 0.70 (SD 0.03), throughout.

---

## 4. Reproduce

```
~/.venvs/myokit/bin/python run_loop.py dictrank_random      # certifies dictrank_fp_random into leaderboard.jsonl
~/.venvs/myokit/bin/python dictrank_split_scheme_run.py     # 8-seed paired split-scheme delta
```

Note: as of 2026-07-20 `run_loop.py` merges a subset run into the existing `loop_summary.json` instead of replacing the file, and stamps each benchmark entry with its own `run_ts` (top-level `run_ts` = when the file was last written). The manual back-up-and-re-merge step this section used to require is gone; run any subset you like.

Second note, **fixed 2026-07-20 (T-517)**. DICTrank dev-side numbers (`dev_auc`, `dev_ci`, dev permutation stats) shift **between processes** because the CV fold assignment depends on `PYTHONHASHSEED`. The feature matrix is byte-identical across processes; only the group-to-fold mapping moves `[Checked: md5 of DICTrankBenchmark(split="random").dev() X under PYTHONHASHSEED 0/1/2 2026-07-20 → d447ff33133dee7e in all three; md5 of the group vector → 84358f61 / 3b6e9990 / 943425bf, i.e. differs]`. Measured swing across two unpinned full runs: `dictrank_fp_random` `fp_logreg_l2_3.0` dev AUC **0.6877 vs 0.7146 (0.027)**, `dictrank_fp` **0.6938 vs 0.7019 (0.008)** `[Checked: two full run_loop.py runs 2026-07-20, unpinned hash seed → values as stated]`. Locked-set numbers and every `passed` flag were unaffected (locked split is index-based, no folds).

That 0.027 mattered because the dev-side DICTrank AUC is quoted in grant-facing copy: a number reported to 2 decimal places was not reproducible at that precision.

**The fix:** `featurize.group_kfold_ids` now sorts the unique group keys before the seeded shuffle, so the group-to-fold mapping is derived from `seed` alone. One line, no change to the locked split, the gate logic, or any scorer. Three fresh interpreters with `PYTHONHASHSEED` unset now give identical fold sizes, and two fresh interpreters give bit-identical dev AUC, locked AUC and permutation stats `[Checked: 3x fresh `DICTrankBenchmark(...).dev()` 2026-07-20 → fold sizes [148 186 144 131 138] (scaffold) and [150 150 149 149 149] (random) in all three; 2x fresh `eval_core.certify` on fp_logreg_l2_3.0 → dictrank_fp dev 0.7138 locked 0.6669 null_mean 0.4970 p 0.005 and dictrank_fp_random dev 0.6883 locked 0.7448 null_mean 0.4941 p 0.005, identical both times]`.

Before → after, certified candidates, seed 0 (dev AUC; locked AUC in brackets, unchanged to 4dp in every case):

| candidate | `dictrank_fp` before → after | `dictrank_fp_random` before → after |
|---|---|---|
| `fp_logreg_l2_3.0` | 0.6895 → **0.7138** [0.6669] | 0.6996 → **0.6883** [0.7448] |
| `fp_logreg_l2_1.0` | 0.6877 → **0.7147** [0.6628] | 0.6995 → **0.6866** [0.7445] |
| `fp_logreg_l2_0.3` | 0.6870 → **0.7148** [0.6618] | 0.6991 → **0.6861** [0.7431] |
| `fp_logreg_l2_0.1` | 0.6868 → **0.7147** [0.6613] | 0.6988 → **0.6859** [0.7427] |
| `fp_bitcount_baseline` | 0.4848 → 0.4848 [0.5412] | 0.4925 → 0.4925 [0.5044] |

("before" = run_ts 2026-07-19T23:41:23, the last unpinned run. The baseline is unchanged because its scorer ignores the training fold entirely.)

**No gate outcome moved.** The same four logreg candidates certify on each benchmark, the bitcount baseline still fails, and the random canary still fails both (dev 0.466 / 0.522, perm p 0.905 / 0.204). The four small benchmarks are untouched: they group by `np.arange`, and sorted order equals the previous set-iteration order for every n from 2 to 399 `[Checked: compared old vs new group_kfold_ids on np.arange(n) for n=2..399 2026-07-20 → arrays identical for every n]`.
