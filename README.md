# cardiac-safety-eval

An evaluation harness for cardiac drug-safety models. It scores each model against a simple baseline, with controls that catch leakage, and it keeps every result, including the ones that failed.

We built it after one of our own results turned out to be an artifact. A transcriptomic "cardiotoxicity signal" was tracking how strongly cells responded to a drug, not anything specific to the heart. That claim was the first thing we ran through the harness, and it did not hold up.

Built by [Perturb Bio](https://perturb.bio).

## How it guards against leakage

**Selection on dev, one look at the test set.** Each benchmark has a `dev` split and a `locked` split. Every candidate is fitted and scored on `dev` only. The loop then picks one champion: the candidate with the highest dev AUC among those that beat the shuffled-label null. Only three models are scored on `locked`: the pre-declared baseline, the champion, and a random canary. The model interface, `fit_score(X_tr, y_tr, X_te)`, never receives test labels. On the fingerprint benchmarks the split is scaffold-disjoint, so a near-copy of a training molecule cannot appear in the test set.

**Shuffled-label null.** We shuffle the labels many times and rerun the full grouped cross-validation on each shuffle. That gives a null distribution for the metric. A real signal lands far out in its tail. A pipeline that leaks scores well even on shuffled labels, and the null exposes it. We report an empirical p-value and the null mean.

**A baseline to beat.** The champion passes only if its permutation p-value is below 0.05, its dev AUC clears the null's 95th percentile, and its locked AUC beats the baseline's locked AUC. Each baseline is the simple thing a real model has to beat: hERG block alone for CiPA, total transcriptional response for the transcriptomic benchmark, and a bit count for DICTrank.

Cross-validation is grouped, by mechanism class on CiPA and by scaffold on DICTrank, so a drug is never tested while a close relative is in training.

### Protocol change, September 2026

Until September 2026, `run_loop.py` scored every candidate on the locked set, and the pass/fail gate used those scores. That made certification a selection step. An internal audit found that the DICTrank model we reported had the lowest dev AUC and the highest locked AUC of the four candidates. The loop now selects on dev before it touches the locked set, and `build_dashboard_data.py`, `visualize.py` and `dictrank_split_scheme_run.py` follow the same rule. The numbers below come from the corrected protocol. Entries in `leaderboard.jsonl` from before the change have a locked score for every candidate; entries after it carry `role` and `locked_scored` fields.

### Nulls on small benchmarks sit below 0.5

On the 996-drug DICTrank benchmark the null mean is 0.50. On the small benchmarks (12 to 28 drugs), fitted-model nulls center around 0.33 to 0.45. This is an artifact of leave-one-group-out cross-validation on small samples. It is not leakage: leakage pushes a null above chance, and these sit below it. Details are in `small_n_null_calibration_2026-07.md`.

## Benchmarks

| Benchmark | n | ground truth | features | split | baseline |
|---|---|---|---|---|---|
| `cipa28_tdp` | 28 | FDA/CiPA torsade class | hERG/ICaL/INa block + Kernik AP risk | 12 dev / 16 locked | hERG-only margin |
| `enginev0_cardiotox` | 33 | iPSC-CM cardiotoxicity label | panel score, DE counts | 22 dev / 11 locked | transcriptional magnitude |
| `variant_tdp` | 28 | FDA/CiPA torsade class | repolarization risk in healthy + LQT2/LQT1 backgrounds | 12 dev / 16 locked | healthy background alone |
| `dictrank_fp` | 996 | FDA DICTrank cardiotoxicity | Morgan fingerprint (r=2, 1024 bit) | 747 dev / 249 locked, scaffold-disjoint | naive bit-count |
| `dictrank_fp_random` | 996 | as above | as above | random, matched n/folds/seed | naive bit-count |

The inputs are in `data/`, with sources and licenses in `data/README.md`.

## Results

Run of 2026-09-27, seed 0. Locked AUCs with 95% bootstrap intervals.

**CiPA-28: nothing beats hERG alone.** The champion is the Kernik action-potential model alone. It scores 0.906 on the locked set, below the hERG-only baseline at 0.938, so it fails the gate. On this benchmark, an iPSC-cardiomyocyte model does not classify torsade risk in healthy cells better than hERG block does.

**Transcriptomic features: a small gain that the test set cannot confirm.** The champion uses all four features and passes the gate, 0.800 (0.43 to 1.00) against 0.700 (0.30 to 1.00) for response magnitude alone. The locked set holds 11 drugs, 5 of them cardiotoxic, and the two intervals overlap almost entirely. On the dev split, panel score alone reaches 0.694, below magnitude alone at 0.876. We previously reported that the panel predicted cardiotoxicity. That signal came from response magnitude, and we have withdrawn the claim.

**Variant conditioning: too few drugs to tell.** The variant-conditioned champion passes the gate, 0.938 against 0.927 for the healthy-only baseline. The gap is one drug out of sixteen and sits inside the confidence interval. A power analysis gives this benchmark 3.9% power to detect even the largest gain the arithmetic allows (+0.073). Settling it takes about 52 labelled drugs at that effect size. See `variant_benchmark_power_2026-07.md`.

**DICTrank: a structure-only model is modest, and the split is not the reason.** On the scaffold-disjoint split, the champion (L2 logistic regression, C = 0.3) scores 0.657 (0.58 to 0.73) against 0.541 (0.46 to 0.63) for the bit-count baseline. Its dev permutation p-value is at most 0.005: none of 200 shuffles matched it. Across 8 seeds it averages 0.695 (SD 0.032) on the scaffold split and 0.714 (SD 0.030) on a random split, so scaffold splitting costs about 0.02 AUC, with the random split higher in 6 of 8 seeds (`dictrank_split_scheme.json`). The best published result on this dataset is 0.84, from 1,038 Mordred descriptors on a random 90-compound hold-out restricted to compounds with complete annotation. The rest of the gap comes from the features and the test set: we use one 1024-bit Morgan fingerprint and score it on 249 scaffold-disjoint drugs. See `dictrank_comparator_2026-07.md`.

## Reproduce

```
pip install -r requirements.txt
python run_loop.py                      # every benchmark; appends to leaderboard.jsonl
python run_loop.py dictrank             # one benchmark
python dictrank_split_scheme_run.py     # 8-seed paired split test (run after dictrank)
python build_dashboard_data.py          # regenerate dashboard_data.json
python visualize.py                     # regenerate loop_scorecard.svg and permutation_null.svg
```

DICTrank dev-side numbers can shift between processes, because cross-validation fold assignment depends on `PYTHONHASHSEED`. The feature matrix is byte-identical across processes; only the group-to-fold mapping moves. Locked-set numbers are unaffected, since the locked split is index-based.

## Layout

- `eval_core.py`: permutation nulls, grouped CV, dev evaluation, the certification gate
- `benchmarks.py`: benchmark definitions and their locked splits
- `models.py`, `featurize.py`: candidate models and featurization
- `run_loop.py`: scores every candidate on dev, selects the champion, scores it on locked
- `data/`: benchmark inputs, with `data/README.md` for sources and licenses
- `leaderboard.jsonl`, `loop_summary.json`: every candidate ever scored, with its gate outcome
- `dashboard.html`, `visualize.py`: static views of the results
- `*_2026-07*.md`: dated methodology notes behind each result

## License

Code: MIT, see `LICENSE`. Data files: see `data/README.md`.
