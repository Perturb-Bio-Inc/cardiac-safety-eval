# cardiac-safety-eval

A leakage-controlled evaluation harness for cardiac drug-safety models, plus the benchmarks it grades and the results it has produced, including the ones that did not work.

The point is not any single model. It is that every score here is trustworthy enough to put in a grant, so a person or an automated loop cannot fool itself. We were burned once by a confident result that turned out to be a magnitude artifact. This harness is built so that failure mode gets caught by construction, and the first thing we did with it was re-run our own prior claim through it and adjudicate it. That claim did not survive.

Built by [Perturb Bio](https://perturb.bio). Runs on Python with numpy, scipy, rdkit and matplotlib.

## The three anti-spurious devices

**1. Data firewall.** Each benchmark splits into `dev` (the loop may fit and select on it) and `locked` (scored once, at certification, never for selection). A model physically never receives test labels: `fit_score(X_tr, y_tr, X_te)` gets no `y_te`. On the fingerprint benchmark the split is scaffold-disjoint, so a near-clone of a training molecule cannot sit in the test set.

**2. Label-permutation null.** Shuffle the labels many times, run the whole grouped-CV pipeline on each shuffle, and build a null distribution of the metric. A real signal sits far in the tail. A leaking pipeline scores high even on shuffled labels and gets caught here. Reported as an empirical p-value and a null mean.

**3. Baseline gates.** A candidate certifies only if its permutation p is below 0.05, its out-of-fold dev score clears the null's 95th percentile, and its locked score beats the benchmark's baseline. Each baseline is chosen to be the thing actually worth beating: hERG-only for CiPA, pure transcriptional magnitude for the transcriptomic benchmark, a naive bit-count for DICTrank.

Cross-validation is grouped (by mechanism class on CiPA, by scaffold on DICTrank) so a drug is never tested while a sibling trains.

### A calibration caveat we measured rather than discovered later

Permutation nulls do not sit at 0.50 everywhere, and pretending otherwise would defeat the purpose. On the ~900-drug DICTrank benchmark the null mean is 0.497. On the small benchmarks (12 to 28 drugs) fitted-model nulls center around 0.33 to 0.45. That deficit is a leave-one-group-out artifact, not leakage. Leakage pushes a null *above* chance; this sits below it, for a different reason. See `small_n_null_calibration_2026-07.md`.

## Benchmarks

| Benchmark | n | ground truth | features | split | baseline |
|---|---|---|---|---|---|
| `cipa28_tdp` | 28 | FDA/CiPA torsade class | hERG/ICaL/INa block + Kernik AP risk | 12 dev / 16 locked | hERG-only margin |
| `enginev0_cardiotox` | 33 | iPSC-CM cardiotoxicity label | panel score, DE counts | stratified 5-fold | transcriptional magnitude |
| `variant_tdp` | 28 | FDA/CiPA torsade class | repolarization risk in healthy + LQT2/LQT1 backgrounds | 12 dev / 16 locked | healthy background alone |
| `dictrank_fp` | ~900 | FDA DICTrank cardiotoxicity | Morgan fingerprint (r=2, 1024 bit) | scaffold-disjoint | naive bit-count |
| `dictrank_fp_random` | ~900 | as above | as above | random, matched n/folds/seed | naive bit-count |

## What it found

Four of these are negative or null results. That is the honest state of the work, and it is why the harness exists.

**Nothing beats hERG-only on CiPA-28.** The hERG-only baseline certifies at locked AUC 0.9375. A multichannel logistic model ties it at 0.9375 and fails the gate; the Kernik action-potential model alone reaches 0.9062. A plain iPSC-cardiomyocyte model does not beat the incumbent at classifying torsade risk in unaffected cells on this benchmark, and we do not claim it does.

**The transcriptomic panel does not classify cardiotoxicity.** `panel_only` scores locked AUC 0.5333, chance, against a magnitude-only baseline at 0.70. An earlier claim that the panel classified cardiotoxicity was a magnitude artifact and is retracted.

**Variant conditioning is not adjudicated.** `variant_conditioned` reaches locked 0.9375 against a healthy-only baseline at 0.9271. The difference is one drug out of sixteen and sits inside the confidence interval. A power analysis puts this benchmark at 3.9% power to detect even the arithmetic-ceiling gain of +0.073, and settling the question needs roughly 52 labelled drugs at that effect size. See `variant_benchmark_power_2026-07.md`. The direction is interesting; the benchmark cannot currently adjudicate it.

**The structure-only cardiotoxicity model is modest, and the split is not the reason.** On the scaffold-disjoint DICTrank split, the best fingerprint model reaches 0.70 (SD 0.03 across 8 seeds; the single certified seed is 0.6669) against a bit-count baseline at 0.5412. The best published number on this dataset is 0.84. An 8-seed paired test shows the scaffold-versus-random split accounts for about 0.019 of that gap, so the rest is the feature space: we use a 1024-bit Morgan fingerprint and nothing else, while 0.84 comes from 1,038 physicochemical descriptors. See `dictrank_comparator_2026-07.md`.

## Reproduce

```
python run_loop.py                      # run every benchmark, write leaderboard.jsonl
python run_loop.py dictrank_random      # run one subset
python dictrank_split_scheme_run.py     # the 8-seed paired split-scheme test
python build_dashboard_data.py          # regenerate dashboard_data.json
```

Note: DICTrank dev-side numbers shift between processes because CV fold assignment depends on `PYTHONHASHSEED`. The feature matrix is byte-identical across processes; only the group-to-fold mapping moves. Locked-set numbers and every `passed` flag are unaffected, since the locked split is index-based.

## Layout

- `eval_core.py` — firewalls, permutation nulls, grouped CV, the certification gate
- `benchmarks.py` — benchmark definitions and their locked splits
- `models.py`, `featurize.py` — candidate models and featurization
- `run_loop.py` — the propose, score, keep loop
- `leaderboard.jsonl`, `loop_summary.json` — every candidate ever scored, with its gate outcome
- `dashboard.html`, `visualize.py` — a static view of the leaderboard
- `*_2026-07*.md` — the methodology notes behind each result above

## License

MIT. See `LICENSE`.
