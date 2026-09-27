# Benchmark inputs

The benchmarks in `benchmarks.py` read these files. Set `CSE_DATA_DIR` to use a different
directory.

| File | Used by | Source | License |
|---|---|---|---|
| `cipa_28drug_reference.csv` | `cipa28_tdp`, `variant_tdp` | FDA/CiPA 28-drug reference set, as compiled in [cipa-validation](https://github.com/Perturb-Bio-Inc/cipa-validation) (`cipa_validation/`) | CC BY 4.0 |
| `cipa_validation_results.csv` | `cipa28_tdp` | Kernik-2019 action-potential risk per drug, output of `cipa_validation_harness.py` in cipa-validation | CC BY 4.0 |
| `variant_susceptibility_results.csv` | `variant_tdp` | Repolarization risk in healthy and long-QT backgrounds, from cipa-validation | CC BY 4.0 |
| `enginev0_classifier_results.csv` | `enginev0_cardiotox` | Per-drug transcriptomic features (panel score, DE counts) computed by Perturb Bio from public DToxS, Sharma et al. 2017 (GSE114686), Burridge et al. 2016 and Wu–Liu iPSC-cardiomyocyte data | CC BY 4.0 |
| `DICTrank_binarised.csv.gz` | `dictrank_fp`, `dictrank_fp_random` | Seal et al., "Insights into Drug Cardiotoxicity from Biological and Chemical Data: The First Public Classifiers for FDA DICTrank," J Chem Inf Model 2024. Figshare, [doi:10.6084/m9.figshare.24312274](https://doi.org/10.6084/m9.figshare.24312274), file `DICTrank_binarised.csv.gz`, unmodified | CC BY 4.0 |

The code in this repository is MIT-licensed (see `../LICENSE`). The data files keep the
licenses above.

The supplementary scripts `panel_replication_run.py`, `panel_replication_clean_run.py` and
`lodo_pooled_stat.py` need raw expression tables from GEO and the LINCS DToxS site, which are
not included. Point `ENGINE_V0_DIR` at a directory holding them. Their results are recorded
in the dated notes `panel_replication_firewall_2026-07-09.md` and
`lodo_clean_strengthening_2026-07.md`.
