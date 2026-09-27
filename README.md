# cardiac-safety-eval

An evaluation harness for cardiac drug-safety models. It scores each model against a simple baseline, with controls that catch leakage, and it keeps every result, including the ones that failed.

We built it after one of our own results turned out to be an artifact. A transcriptomic "cardiotoxicity signal" was tracking how strongly cells responded to a drug, not anything specific to the heart. That claim was the first thing we ran through the harness, and it did not hold up.

Built by [Perturb Bio](https://perturb.bio). Python, with numpy, scipy, rdkit and matplotlib.

## How it guards against leakage

**Held-out test labels.** Each benchmark has a `dev` split and a `locked` split. Models are fitted and compared on `dev`. The model interface, `fit_score(X_tr, y_tr, X_te)`, never receives the test labels. On the fingerprint benchmark the split is scaffold-disjoint, so a near-copy of a training molecule cannot appear in the test set.

**Shuffled-label null.** We shuffle the labels many times and rerun the full grouped cross-validation on each shuffle. That gives a null distribution for the metric. A real signal lands far out in its tail. A pipeline that leaks scores well even on shuffled labels, and the null exposes it. We report an empirical p-value and the null mean.

**A baseline to beat.** A candidate passes only if its permutation p-value is below 0.05, its out-of-fold dev score clears the null's 95th percentile, and its locked score beats the benchmark's baseline. Each baseline is the simple thing a real model has to beat: hERG block alone for CiPA, total transcriptional response for the transcriptomic benchmark, and a bit count for DICTrank.

Cross-validation is grouped, by mechanism class on CiPA and by scaffold on DICTrank, so a drug is never tested while a close relative is in training.

### Known issue: the locked set is not fully sealed

`run_loop.py` scores every candidate on the locked set at certification, and the pass/fail gate uses that score. That turns certification into a selection step. A July 2026 audit found that the reported DICTrank model had the lowest dev AUC and the highest locked AUC of the four candidates. The effect is small, 0.006 AUC, but it is the failure the locked split exists to prevent. The fix is to score a single pre-chosen candidate per benchmark on `locked`. That change is not yet made, so read the locked numbers below as slightly optimistic.

### Nulls on small benchmarks sit below 0.5

On the ~900-drug DICTrank benchmark the null mean is 0.497. On the small benchmarks (12 to 28 drugs), fitted-model nulls center around 0.33 to 0.45. This is an artifact of leave-one-group-out cross-validation on small samples. It is not leakage: leakage pushes a null above chance, and these sit below it. Details are in `small_n_null_calibration_2026-07.md`.

## Benchmarks

| Benchmark | n | ground truth | features | split | baseline |
|---|---|---|---|---|---|
| `cipa28_tdp` | 28 | FDA/CiPA torsade class | hERG/ICaL/INa block + Kernik AP risk | 12 dev / 16 locked | hERG-only margin |
| `enginev0_cardiotox` | 33 | iPSC-CM cardiotoxicity label | panel score, DE counts | stratified 5-fold | transcriptional magnitude |
| `variant_tdp` | 28 | FDA/CiPA torsade class | repolarization risk in healthy + LQT2/LQT1 backgrounds | 12 dev / 16 locked | healthy background alone |
| `dictrank_fp` | ~900 | FDA DICTrank cardiotoxicity | Morgan fingerprint (r=2, 1024 bit) | scaffold-disjoint | naive bit-count |
| `dictrank_fp_random` | ~900 | as above | as above | random, matched n/folds/seed | naive bit-count |

## Results

None of the four results is a win for a new model.

**CiPA-28: nothing beats hERG alone.** The hERG-only baseline scores a locked AUC of 0.9375. A multichannel logistic model ties it at 0.9375, so it fails the gate. The Kernik action-potential model alone reaches 0.9062. On this benchmark, an iPSC-cardiomyocyte model does not classify torsade risk in healthy cells better than hERG block does.

**The transcriptomic panel does not predict cardiotoxicity.** `panel_only` scores a locked AUC of 0.5333, which is chance. The magnitude-only baseline scores 0.70. We previously reported that the panel predicted cardiotoxicity. That signal came from response magnitude, and we have withdrawn the claim.

**Variant conditioning: too few drugs to tell.** `variant_conditioned` reaches a locked AUC of 0.9375, against 0.9271 for the healthy-only baseline. The gap is one drug out of sixteen and sits inside the confidence interval. A power analysis gives this benchmark 3.9% power to detect even the largest gain the arithmetic allows (+0.073). Settling it takes about 52 labelled drugs at that effect size. See `variant_benchmark_power_2026-07.md`.

**DICTrank: a structure-only model is modest, and the split is not the reason.** On the scaffold-disjoint split, the best fingerprint model reaches 0.70 (SD 0.03 across 8 seeds; the single certified seed scores 0.6669). The bit-count baseline scores 0.5412. The best published result on this dataset is 0.84, from 1,038 Mordred descriptors on a random 90-compound hold-out restricted to compounds with complete annotation. An 8-seed paired test puts the cost of scaffold splitting at about 0.019 AUC. The rest of the gap comes from the features and the test set: we use one 1024-bit Morgan fingerprint and score it on a 249-drug scaffold-disjoint locked set. See `dictrank_comparator_2026-07.md`.

## Reproduce

```
python run_loop.py                      # run every benchmark, write leaderboard.jsonl
python run_loop.py dictrank_random      # run one subset
python dictrank_split_scheme_run.py     # the 8-seed paired split-scheme test
python build_dashboard_data.py          # regenerate dashboard_data.json
```

DICTrank dev-side numbers shift between processes, because cross-validation fold assignment depends on `PYTHONHASHSEED`. The feature matrix is byte-identical across processes; only the group-to-fold mapping moves. Locked-set numbers and every `passed` flag are unaffected, since the locked split is index-based.

## Layout

- `eval_core.py`: firewalls, permutation nulls, grouped CV, the certification gate
- `benchmarks.py`: benchmark definitions and their locked splits
- `models.py`, `featurize.py`: candidate models and featurization
- `run_loop.py`: the propose, score, keep loop
- `leaderboard.jsonl`, `loop_summary.json`: every candidate ever scored, with its gate outcome
- `dashboard.html`, `visualize.py`: a static view of the leaderboard
- `*_2026-07*.md`: the methodology notes behind each result above

## License

MIT. See `LICENSE`.
