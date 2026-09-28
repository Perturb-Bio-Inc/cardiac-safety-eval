# cardiac-safety-eval

An evaluation harness for cardiac drug-safety models. It scores each model against a simple baseline, with controls that catch leakage, and it keeps every result, including the ones that failed.

We built it after one of our own results turned out to be an artifact. A transcriptomic "cardiotoxicity signal" was tracking how strongly cells responded to a drug, not anything specific to the heart. That claim was the first thing we ran through the harness, and it did not hold up.

Built by [Perturb Bio](https://perturb.bio).

## How it guards against leakage

**Selection on dev before the test set.** Each benchmark has a `dev` split and a `locked` split. Every candidate is fitted and scored on `dev` only. The loop then picks one champion: the candidate with the highest dev AUC among those that beat the shuffled-label null. Only three models are scored on `locked`: the pre-declared baseline, the champion, and a random canary. The model interface, `fit_score(X_tr, y_tr, X_te)`, never receives test labels. On `dictrank_fp` the split is scaffold-disjoint, so no Bemis–Murcko scaffold appears in both splits.

**Shuffled-label null.** We shuffle the labels many times and rerun the full grouped cross-validation on each shuffle. That gives a null distribution for the metric. A real signal lands far out in its tail. A pipeline that leaks scores well even on shuffled labels, and the null exposes it. We report an empirical p-value and the null mean.

**A baseline to beat.** The champion passes only if its permutation p-value is below 0.05, its dev AUC clears the null's 95th percentile, and its locked AUC beats the baseline's locked AUC. Each baseline is the simple thing a real model has to beat: hERG block alone for CiPA, total transcriptional response for the transcriptomic benchmark, and a bit count for DICTrank.

Cross-validation on dev is grouped by mechanism class on CiPA and the variant benchmark, and by scaffold on `dictrank_fp`. A shared scaffold does not capture every close relative, and the transcriptomic and random-split DICTrank benchmarks use ungrouped folds.

### Protocol change, September 2026

Until September 2026, `run_loop.py` scored every candidate on the locked set, and the pass/fail gate used those scores. That made certification a selection step. An internal audit found that the DICTrank model we reported had the lowest dev AUC and the highest locked AUC of the four candidates. The loop now selects on dev before it touches the locked set, and `build_dashboard_data.py`, `visualize.py` and `dictrank_split_scheme_run.py` follow the same rule. The numbers below come from the corrected protocol. Entries in `leaderboard.jsonl` from before the change have a locked score for every candidate; entries after it carry `role` and `locked_scored` fields.

Two caveats follow. First, the locked sets are not unseen: ten runs before the change (145 records) scored every candidate on them. Second, locked AUCs moved between July and September 2026 with no change to code or data, by about 0.005 on DICTrank. The likeliest cause is package versions; we have not isolated it. So the drop in the reported DICTrank number, from 0.667 to 0.657, splits about evenly between the selection fix and that drift. `requirements.txt` pins the versions behind the current numbers.

### Known label leak in the transcriptomic benchmark

The panel features in `enginev0_cardiotox` are not label-free. The panel's gene directions came from the average response of the DToxS drugs labelled cardiotoxic, and those include this benchmark's 5 locked and 11 dev cardiotoxic drugs. So every score on this benchmark that uses `panel_score`, `n_up_panel` or `n_down_panel` is contaminated, on dev and on locked. Only `magnitude_only` (`log_n_de`, the number of differentially expressed genes) uses no labels. The fix is to rebuild the panel without this benchmark's drugs; it is not yet done.

### Nulls on small benchmarks sit below 0.5

On `dictrank_fp` (747 dev drugs) the null means are 0.50. On the small benchmarks (dev splits of 12 and 22 drugs), several fitted models have null means between 0.30 and 0.40. This is an artifact of leave-one-group-out cross-validation on small samples. It is not leakage: leakage pushes a null above chance, and these sit below it. Details are in `small_n_null_calibration_2026-07.md`.

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

**Transcriptomic features: no valid result.** The champion uses all four features and scores 0.800 on the locked set, against 0.700 for response magnitude alone. The label leak described above invalidates that comparison. Even with the leak, panel score alone scores below magnitude alone on dev (0.694 against 0.876). We previously reported that the panel predicted cardiotoxicity. That signal came from response magnitude, and we have withdrawn the claim.

**Variant conditioning: too few drugs to tell.** The variant-conditioned champion passes the gate, 0.938 against 0.927 for the healthy-only baseline. The whole gain is half of one drug pair: of 48 cardiotoxic-versus-safe pairs on the locked set, the baseline orders 44 correctly and ties one, and the champion orders 45 correctly. On dev the champion scores below the baseline (0.812 against 0.906). A power analysis gives this benchmark 3.9% power to detect even the largest gain the arithmetic allows (+0.073). Settling it takes about 52 labelled drugs at that effect size. See `variant_benchmark_power_2026-07.md`.

**DICTrank: a structure-only model is modest, and the split is not the reason.** On the scaffold-disjoint split, the champion (L2 logistic regression, C = 0.3) scores 0.657 (0.58 to 0.73) against 0.541 (0.46 to 0.63) for the bit-count baseline. Its dev permutation p-value is at most 0.005: none of 200 shuffles matched it. The four regularization strengths tie on dev to within 0.001 AUC, so which one becomes champion is arbitrary. Across 8 seeds the champion averages 0.695 (SD 0.032) on the scaffold split and 0.714 (SD 0.030) on a random split (`dictrank_split_scheme.json`). That difference of about 0.02 is not significant (paired t-test p = 0.20; random higher in 6 of 8 seeds, sign test p = 0.29). The highest published result we found on this dataset is 0.84, from 1,038 Mordred descriptors on a random 90-compound hold-out restricted to compounds with complete annotation. That result differs from ours in both features and test set, and we have not tested which difference matters. See `dictrank_comparator_2026-07.md`.

## Reproduce

```
pip install -r requirements.txt
python run_loop.py                      # every benchmark; appends to leaderboard.jsonl
python run_loop.py dictrank             # one benchmark
python dictrank_split_scheme_run.py     # 8-seed paired split test (run after dictrank)
python build_dashboard_data.py          # regenerate dashboard_data.json
python visualize.py                     # regenerate loop_scorecard.svg and permutation_null.svg
```

Results can shift in the third decimal with package versions. The current numbers come from the versions in `requirements.txt`.

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
