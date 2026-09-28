# Panel cross-dataset replication: permutation-null firewall

> **SCOPE NOTE added 2026-07-17. No number in this doc changes; two clarifications that protect them.**
>
> **1. "Membership" here means panel-selector membership.** Firewall D removes the leak where a gene was in the tested panel because the held-out dataset helped select it. Sharma and DToxS were also planned as training data for a separate model. That does not affect any number here, because every result in this note is a sign-permutation statistic over public DEG matrices with no fitted model.
>
> **2. The Sharma fold is GSE114686 RNA-seq, not microarray.** This doc calls it "multi-compound, multi-platform microarray" (Firewall B section) and the standing caveats call it multi-compound. (Source: read panel_replication_clean_run.py 2026-07-17 → line 57 loads `data/sharma2017/GSE114686_ProcessedData.csv`; the directory name `sharma2017` is a legacy misnomer, and the file is TMM-normalized STAR feature counts per a separate internal note) **The "microarray" descriptor is unsupported; do not repeat it.** The 0.945 (Firewall B) and 0.791 (Firewall D) Sharma numbers were computed on GSE114686 and are unaffected. The "most independent hold-out" reasoning stands on multi-compound TKI content and a separate lab, not on platform diversity.
>

**2026-07-09.** Scripts: `panel_replication_run.py` (Firewalls A/B/C) and `panel_replication_clean_run.py` (Firewall D, membership-clean LODO). Both numpy+scipy+csv, seed=0, 10,000 permutations per test. Data: `engine-v0/data/richer_panel_genes.csv` (panel + per-dataset z-scores), `engine-v0/data/variant_validation/GSE198258_DEG.xls` (disease-variant DEG), the three on-disk raw matrices (Sharma/Burridge/Wu-Liu), and the re-downloaded DToxS full DEG TSVs (LINCS DToxS SVD `Degs_initial_iPSCdCMs_P0`, `iyengarlab.org/dtoxs/files/LINCS_DToxS_SVD.zip`, held in scratch outside the repo).

## Purpose

Harden the claim that the **81-gene cardiac-identity panel replicates across 4 independent iPSC-CM datasets** (DToxS/Hansen 2024, Sharma 2017, Burridge 2016, Liu-Wu 2024) against a reviewer objection. The panel was *selected* by a Stouffer-Z meta-analysis with a Bonferroni cut (`p<5e-7`) plus a directional-consistency rule (`>=3/4 datasets same sign`), so the genes are concordant partly by construction. The firewall asks whether the cross-dataset agreement exceeds what that selection alone forces, using permutation nulls in the eval_core.py discipline (shuffle identity/labels, put the observed statistic in the null tail).

## What the selection guarantees (the floor a reviewer will name)

The `>=3/4 same-sign` rule guarantees every panel gene agrees with its assigned direction in at least 3 of 4 datasets. So "coordinated direction across all four" is not free of leakage. Firewall B is built to escape this floor; A is reported with the floor stated.

## Firewall A: pairwise sign-concordance vs gene-identity shuffle

Among the 55 panel genes measured in all 4 datasets, statistic = mean over the 6 dataset-pairs of the fraction of genes whose z-signs agree. Null shuffles each dataset's sign vector across genes independently, preserving each dataset's up/down marginal.

(Source: ran panel_replication_run.py 2026-07-09 -> observed mean pairwise concordance = 0.9273; null mean 0.7379, null p95 0.7636, null max (10,000 shuffles) 0.8061; empirical p = 1.0e-4)

The observed value sits above the entire null distribution (max 0.806), so p is floored at 1/(N+1)=1.0e-4 and is in fact smaller. Caveat: this test is partly floored by the `>=3/4` selection rule, which is why B is the primary result.

## Firewall B: leave-one-dataset-out (the leakage-controlled test)

For each held-out dataset, the consensus direction is built from the **other three** (sign of the summed z), then we test whether the held-out dataset agrees. The direction being tested is independent of the held-out data, so this mostly escapes the selection floor. Null shuffles the held-out signs.

(Source: ran panel_replication_run.py 2026-07-09 -> held-out agreement, n=55 each)

| Held out | Observed agreement | Null mean | Null p95 | Empirical p |
|---|---|---|---|---|
| DToxS | 0.909 | 0.760 | 0.800 | 2.0e-4 |
| Sharma | 0.945 | 0.714 | 0.764 | 1.0e-4 |
| Burridge | 1.000 | 0.702 | 0.782 | 1.0e-4 |
| Liu-Wu | 1.000 | 0.703 | 0.782 | 1.0e-4 |

Every held-out dataset points the same way as the consensus of the other three far more than chance. Sharma is the most independent hold-out (multi-compound, multi-platform microarray) and still lands at 0.945, p=1.0e-4.

## Firewall C: disease-variant enrichment, recomputed + permutation-checked

The load-bearing number for the disease-variant claim is the GSE198258 enrichment (isogenic MYH7 R723C / MYH6 R725C iPSC-CM vs base-edited control). Recomputed independently from the primary DEG table, no reuse of the prior script's output.

(Source: ran panel_replication_run.py 2026-07-09 -> 28,043 genes; 79 of 81 panel genes measured, 28 significant (padj<0.05) = 35.4% vs 5.0% background; odds ratio 10.38, 7.05x background, one-sided Fisher p = 8.03e-17)

This confirms the previously reported "odds 10.4, 7.0x, p=8e-17" exactly. Distribution-free check: draw random 79-gene sets from the measured background and count significant genes.

(Source: ran panel_replication_run.py 2026-07-09 -> random 79-gene sets: mean 4.0 significant, max 13 over 10,000 draws; observed 28; empirical p = 1.0e-4)

No random set of the same size reaches even half the observed count, so the Fisher p is not an artifact of the parametric approximation.

## Firewall D: fully-clean LODO (membership leakage removed), all four folds

Firewall B removed the *direction* leak but not the *membership* leak: a panel gene was in the tested set because the 4-dataset Stouffer + `>=3/4` rule put it there, using the held-out dataset. Firewall D closes that. For a held-out dataset, the panel is **re-selected from scratch over the full transcriptome background using only the other three datasets** (same Stouffer-Z + Bonferroni `p<5e-7` + unanimous-3/3 rule), then the held-out dataset's directional agreement is tested on that held-out-blind panel with a sign-permutation null. Script: `panel_replication_clean_run.py`.

**Data: all four full backgrounds now available (2026-07-09).** The DToxS full DEG TSVs were re-downloaded (LINCS DToxS SVD, `iyengarlab.org/dtoxs/files/LINCS_DToxS_SVD.zip`, the `Degs_initial_iPSCdCMs_P0` directory, 266 per-drug/cell-line TSVs, ~16.4k genes each, the full transcriptome rather than the top-600 on disk). The DToxS full-background z now reconstructs from the same the original panel-selection script pipeline and matches `richer_panel_genes.csv` exactly, alongside the three that already did:

(Source: ran panel_replication_clean_run.py 2026-07-09 -> reconstructed full-background z reproduces richer_panel_genes.csv z-scores to machine precision: DToxS max|diff| 1.78e-15 over 717 panel genes, Sharma 2.66e-15 / 584, Burridge 6.66e-16 / 705, Wu-Liu 0.0 / 725)

Because every dataset now has a full background, every leave-one-out fold re-selects its panel over the full transcriptome from the other three and tests the held-out direction using the **same full-background z sign** the panel selection and Firewalls A/B use. Held-out direction = sign of the held-out dataset's full-background z (mean logFC across its cardiotox+ conditions, z-scored across all its genes).

(Source: ran panel_replication_clean_run.py 2026-07-09 -> fully-clean LODO, 10,000 permutations, seed 0)

| Held out | Selectors | Panel size | Covered | Observed | Null mean | Null max | Empirical p |
|---|---|---|---|---|---|---|---|
| DToxS | Sharma+Burridge+WuLiu | 98 | 98 | 0.704 (69/98) | 0.637 | 0.704 | 4.70e-3 |
| Sharma | DToxS+Burridge+WuLiu | 65 | 43 | 0.791 (34/43) | 0.681 | 0.884 | 4.18e-2 |
| Burridge | DToxS+Sharma+WuLiu | 26 | 23 | 1.000 (23/23) | 0.917 | 1.000 | 4.24e-2 |
| Wu-Liu | DToxS+Sharma+Burridge | 10 | 9 | 0.889 (8/9) | 0.716 | 0.889 | 2.20e-1 |

Every held-out dataset agrees with the other-three consensus above its own shuffle-null mean. Three of four folds reach p<0.05 (DToxS p=4.7e-3, Sharma p=4.2e-2, Burridge p=4.2e-2); the Wu-Liu fold does not (p=0.22), on only 9 covered genes (the re-selection from DToxS+Sharma+Burridge yields a small 10-gene panel, so this fold is underpowered). The nulls are high (0.64 to 0.92) because the re-selected panels and the held-out directions are both sign-skewed toward cardiac-identity loss, so a shuffle already agrees often by chance; the sign-permutation null is the correct leakage-controlled test and it costs power when both vectors are skewed.

**This is materially weaker than Firewall B and than the earlier partial Firewall D.** Firewall B (membership-leaky, direction-clean) reported p<=2e-4 for every fold; the fully-clean version is p=4.7e-3 to 0.22. Removing the *membership* leak is what moves the numbers: in Firewall B the tested gene set was chosen with the held-out dataset in it, so it was pre-loaded to agree.

**Correction to the prior DToxS-fold number.** The earlier run (before the DToxS full DEG re-download) tested the DToxS held-out direction using the on-disk **top-600** DEG file and reported 0.990 (95/96), p=1.0e-4. That number was a top-600 artifact, not the consistent measure. On the identical 98-gene panel, the top-600 direction reproduces 0.990 but the full-background direction gives 0.704: 27 of 96 covered genes flip sign between the two measures. The flippers are the anthracycline DNA-damage / p53 program (GADD45A, CDKN1A, FDXR, ICAM1, PARP14, HERC5, MAP1LC3B and similar), which dominate the top-600 lists of the dox-heavy selectors but average out toward zero (or flip) when weighted equally across all 16 DToxS cardiotox drugs, most of which are TKIs and antibodies that do not trigger that program. The full-background z sign is the measure the panel was selected on and that every other firewall uses, so 0.704 is the number that belongs in the LODO; 0.990 was measuring agreement among only the strongly-DE genes and is not leakage-clean-comparable.

(Source: ran cross-check 2026-07-09 -> same 98-gene DToxS-blind panel; top-600 direction 95/96=0.990; full-background direction 69/98=0.704; 27/96 genes covered by both differ in sign)

## What is verified

**Verified (2026-07-09):**
- All three Firewall A/B/C permutation nulls, from primary data on disk.
- The GSE198258 Fisher p=8.03e-17 reproduced independently.
- Firewall D fully-clean LODO for **all four folds**: the DToxS full DEG tables were re-downloaded, all four full-background z-matrices reconstruct to machine precision against `richer_panel_genes.csv`, and each fold re-selects a held-out-blind panel and tests the held-out full-background direction. DToxS 0.704 (p=4.7e-3), Sharma 0.791 (p=4.2e-2), Burridge 1.000 (p=4.2e-2), Wu-Liu 0.889 (p=0.22).
- The prior DToxS-fold 0.990 was corrected: it used the top-600 direction; the consistent full-background direction on the identical panel is 0.704.

**Fully closed. No further download needed.**
- All four LODO folds are now membership-clean and direction-consistent. The DToxS full DEG TSVs are not committed: they unzip to about 1 GB from the 2 GB LINCS DToxS SVD zip file. Set `DTOXS_DEG_DIR` to their location; `panel_replication_clean_run.py` also checks `/tmp`. All reconstruction z-matrices are computed in memory and not committed, since they derive from the raw files.

**Standing caveats to carry (carry these forward):**
- Asymmetric replication (pre-registered): Burridge and Liu-Wu are both doxorubicin datasets, so their folds partly echo each other. Effective independent replication is ~2.5 datasets (DToxS + Sharma multi-compound + one dox). In Firewall D the DToxS hold-out (p=4.7e-3, 98 genes) is the load-bearing independent point; Sharma (p=4.2e-2) is second.
- GSE198258 is n=4, single study, and a different specific MYH7 variant (R723C) than other MYH7 residues of clinical interest, same gene and sarcomeric-HCM class.
- Firewall D power note: the sign-permutation null loses power when the re-selected panel and the held-out direction are both skewed toward cardiac-identity loss. This is why the fully-clean p-values (1e-2 to 1e-3 on the strong folds, 0.22 on the 9-gene Wu-Liu fold) are far higher than Firewall B's floored 1e-4, even though every observed agreement exceeds its null mean. The Firewall A/B/C floored p<=1e-4 values still hold for those tests (observed above the null max), but do not carry them over to Firewall D.

## Verdict

**The cross-dataset replication claim survives, but the fully-clean test is modest, not the near-perfect result Firewall B suggested.** The honest picture has two layers:

- **Firewall B (membership-leaky, direction-clean):** every held-out dataset agrees with the other-three consensus at 0.909 to 1.000, p<=2e-4. This is the figure earlier write-ups relied on, and it is inflated by membership leakage (the tested gene set was chosen with the held-out dataset in it).
- **Firewall D (fully clean, membership + direction):** re-selecting the panel without the held-out dataset and testing on the consistent full-background direction, three of four folds are significant at p<0.05 (DToxS 0.704 / p=4.7e-3, Sharma 0.791 / p=4.2e-2, Burridge 1.000 / p=4.2e-2) and the Wu-Liu fold is not (0.889 / p=0.22, 9 covered genes, underpowered). Every fold's observed agreement still exceeds its shuffle-null mean.

The disease-variant enrichment (GSE198258, p=8e-17) is independent of the LODO and unaffected: confirmed both parametrically and against a distribution-free null. The load-bearing correction for future write-ups: drop the "0.990 / strengthens once the leak is removed" framing. The membership-clean signal is directionally consistent in every fold, but it is p~1e-2 to 1e-3 on the strong folds, not p<1e-4, and one dox-only fold is non-significant. Use the DToxS fold (p=4.7e-3, 98 genes, multi-compound held-out) as the load-bearing clean result, with the underpowered Wu-Liu fold named.

## Summary paragraph

The 81-gene cardiac-identity panel was selected by a Stouffer-Z cross-dataset meta-analysis (Bonferroni p<5e-7) over four independent iPSC-CM transcriptomic datasets. To rule out that the cross-dataset agreement is an artifact of the selection cut, we ran a fully leakage-clean leave-one-dataset-out test: for each held-out dataset, the panel was re-selected from scratch over the full transcriptome using only the other three datasets, and the held-out dataset's direction was tested against the resulting held-out-blind panel with a sign-permutation null. The held-out direction agrees with the re-selected consensus above chance in all four folds, reaching significance in three (held-out DToxS 70% of 98 genes, p=4.7x10^-3; Sharma 79%, p=4.2x10^-2; Burridge 100% of 23 genes, p=4.2x10^-2) and not in the smallest fold (Wu-Liu, 9 genes, p=0.22). In an isogenic disease-variant line (GSE198258, MYH7 R723C vs base-edited control), the panel is 7.0x enriched for differential expression over the transcriptome background (35.4% vs 5.0%; one-sided Fisher p = 8.0x10^-17), a value no random equal-size gene set approaches (max 13 vs observed 28 significant over 10,000 draws). The replication is a property of the cardiac-identity program rather than the gene-selection procedure, though the fully-clean cross-dataset effect is modest.

**Scope on the enrichment figure.** The 7.0x enrichment is measured on GSE198258, a proxy residue (MYH7 R723C carried with a paralogous MYH6 R725C edit), not on the residue class the figure is most often quoted for. A second public isogenic dataset for a different MYH7 residue returns 2.76x (Fisher p = 0.097, non-significant). Variant signatures in this gene are heterogeneous, so the 7.0x figure should not be carried alone.
