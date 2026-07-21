# Strengthening or bounding the membership-clean LODO generalization

> **NOTE added 2026-07-17 (T-360): the pooled p ≈ 1e-3 is NOT contaminated by the engine pretraining corpus. No number here changes.** The audit checked whether the planned corpus overlaps these folds. **Two of the four folds' datasets are slated corpus members** (Sharma GSE114686 = member H; DToxS = member F). **This does not touch the pooled statistic or any fold**, because everything in this doc is a sign-permutation computation over public DEG matrices with no fitted model. **The LODO folds are not engine benchmarks and never will be**, so "held out from the model" does not apply to them.
>
> **What to carry instead: a placement rule.** Do not put the pooled p ≈ 1e-3 next to a description of an engine pretrained on DToxS and Sharma without separating the claims. Both halves are true; a reviewer who joins them asks why the engine trained on the datasets the generalization claim holds out. The answer (different claims, different objects, no model in the LODO) is correct but takes a paragraph, and it is much better made before the question than after.
>
> **Two corrections to §3b's prose, neither of which moves a number.** (a) It calls Sharma "microarray, EGFR/VEGFR kinase inhibitors"; **GSE114686 is RNA-seq** `[Checked: read panel_replication_clean_run.py 2026-07-17 → line 57 loads `data/sharma2017/GSE114686_ProcessedData.csv`, a legacy dir name; TMM-normalized STAR feature counts]`. (b) §3b's "GSE114686 in the same search is Sharma, already a fold; do not double-count it" is **correct and now confirmed against the loader**, and it is load-bearing: GSE114686 is simultaneously this doc's Sharma fold, the Task-1 external benchmark, and a slated corpus member. It is eval-only from here (spec §7d).

**2026-07-11.** Companion to the canonical firewall doc `panel_replication_firewall_2026-07-09.md` (this note does not overwrite or change it). Scripts: `panel_replication_clean_run.py` (canonical, unchanged) and `lodo_pooled_stat.py` (new, this note). Both numpy+scipy only, seed 0, 10,000 permutations. Purpose: take the single most-cited generalization number in the downstream write-ups (membership-clean leave-one-dataset-out, held-out DToxS 0.704) and make it as strong as the on-host data allows, or document where its ceiling is.

## Why this matters

After the 2026-07-09 correction, the fully-clean LODO is the load-bearing "our direction generalizes" evidence. It is modest: three of four folds significant, the strongest independent fold at p=4.7e-3, and one fold (Wu-Liu) underpowered at 9 genes and non-significant. A reviewer who leans on the near-perfect Burridge fold (1.000) is leaning on the weakest kind of evidence. The goal here is a single generalization statistic that pools all four folds without resting on any one of them.

## 1. Current clean fold table, reproduced

`[Checked: ran ~/.venvs/myokit/bin/python panel_replication_clean_run.py 2026-07-11 → fully-clean LODO, all four folds, seed 0, 10,000 permutations]`

| Held out | Selectors | Panel | Covered | Observed | Null mean | Null max | Empirical p |
|---|---|---|---|---|---|---|---|
| DToxS | Sharma+Burridge+WuLiu | 98 | 98 | 0.704 (69/98) | 0.637 | 0.704 | 4.70e-3 |
| Sharma | DToxS+Burridge+WuLiu | 65 | 43 | 0.791 (34/43) | 0.681 | 0.884 | 4.18e-2 |
| Burridge | DToxS+Sharma+WuLiu | 26 | 23 | 1.000 (23/23) | 0.917 | 1.000 | 4.24e-2 |
| Wu-Liu | DToxS+Sharma+Burridge | 10 | 9 | 0.889 (8/9) | 0.716 | 0.889 | 2.20e-1 |

Every value matches the canonical firewall doc to the digit. The canonical DToxS 0.704 is confirmed, not changed. No published number moved.

## 2. The strengthening: a pooled cross-fold statistic

`[Checked: ran ~/.venvs/myokit/bin/python lodo_pooled_stat.py 2026-07-11 → pooled statistics below, seed 0, 10,000 permutations]`

**Joint sign-permutation pooled test (the single defensible number).** Pool every covered panel gene across all four folds into one statistic: total directional agreements. Under the joint null, each fold's held-out sign vector is permuted independently, so the pooled null carries each fold's own baseline skew. A fold whose null already agrees 92% of the time (Burridge) contributes ~92% agreements even under the null, so it cannot dominate the pooled result.

- Observed total agreements: **134 / 173 = 0.775**
- Joint-null mean: 119.3 / 173 = 0.689; joint-null max over 10,000 permutations: 132
- Observed (134) sits above the entire joint null (max 132), so **pooled empirical p ≤ 1.0e-4** (floored at 1/(N+1)).

**Parametric cross-checks on the four per-fold p-values** (equal weights):

- Fisher's method: X² = 26.42, df = 8 → **p = 8.9e-4**
- Stouffer's Z: Z = 3.412 → **p = 3.2e-4**

All three methods land at p on the order of 1e-3 to 1e-4. This is the properly-powered summary to quote downstream: the panel direction generalizes across the four held-out folds at pooled p ≈ 1e-3, and it does not depend on the one near-perfect fold (removing Burridge entirely, the other three still pool well below 0.05).

**Caveat that travels with the pooled number.** Both parametric combinations and the joint-permutation null assume the four folds are independent. They are not fully independent: Burridge and Wu-Liu are both doxorubicin datasets, and every fold's consensus is built from three datasets that overlap across folds. Positive dependence widens the true null, so the pooled p is optimistic (a lower bound on the true p). The pre-registered "effective independent replication ≈ 2.5 datasets" caveat from the firewall doc applies here unchanged. Read the pooled result as p on the order of 1e-3, not as a hard 1e-4.

## 3. Why the other two strengthening levers do not pay off on-host

**(a) Deepening the underpowered Wu-Liu fold: not possible without changing the selection rule.**

`[Checked: ran lodo_pooled_stat.py Wu-Liu diagnostic 2026-07-11 → panel re-selected from DToxS+Sharma+Burridge = 10 genes; 9 of 10 covered by Wu-Liu, 1 uncovered]`

The Wu-Liu fold is stuck at 9 genes because the panel re-selected from the *other three* datasets is only 10 genes to begin with. Wu-Liu already covers 9 of those 10. Recovering more Wu-Liu overlap genes can add at most the 1 uncovered gene; it cannot lift the fold out of "underpowered," because the cap is the panel size, not Wu-Liu's coverage. The panel from DToxS+Sharma+Burridge is small because Sharma (microarray, EGFR/VEGFR kinase inhibitors) shares few unanimous-direction, Bonferroni-surviving hits with the doxorubicin-heavy Burridge. Growing that panel would require relaxing the Bonferroni cut (p<5e-7), which changes the selection rule and breaks comparability with the canonical clean LODO. So this lever is closed without a rule change we do not want to make. Verified negative.

**(b) Adding a qualifying 5th independent iPSC-CM dataset: not loadable in this iteration; concrete acquisition step recorded.**

No qualifying 5th dataset is present on this host, so it cannot be added tonight. The most promising candidate for a future run is **GEO GSE217421** `[Checked: fetched GSE217421_family.soft.gz + counts header 2026-07-17 (T-356) → !Series_overall_design "Six different iPSC-derived cardiomyocyte cell lines were treated with one out of 54 FDA-approaved drugs or DMSP for 48 hours" (sic); 1,171 samples across cell lines MSN01/02/05/06/08/09; counts file 38,478 gene rows × 1,171 sample columns, HGNC-symbol-keyed]`. **Confirmed to be what the 2026-07-11 WebSearch described.** Two things a future run should know before pipeline-matching it (per an internal audit): it is the **Conv** (conventional RNA-seq) arm of superseries GSE217424, the sibling **GSE217423** being the 3'-DGE arm at only 41 drugs × 2 lines; and it is the **GEO-hosted equivalent of the LINCS-portal `Degs_initial_iPSCdCMs_P0` tables this doc's DToxS fold already uses** (same 6 MSN lines, same 54 drugs). That last point cuts both ways: it is the clean-licence route to the same data, but **it is therefore NOT independent of the existing DToxS fold** and cannot serve as the 5th independent fold on its own. Step 2 below (confirm iPSC-CM, not primary) is closed: it is iPSC-CM. Note `GSE146096` is the trap to avoid; it is PromoCell **primary** cardiac myocytes. It would add independence only if it contributes non-doxorubicin, non-Sharma-TKI cardiotoxicants (the current four are already TKI-heavy via Sharma and dox-heavy via Burridge/Wu-Liu). Note GSE114686 in the same search is Sharma, already a fold; do not double-count it.

Exact acquisition step a future iteration would take:
1. Pull the GSE217421 series matrix / supplementary count or DEG files from GEO (`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE217421`).
2. Confirm it is iPSC-CM (not primary or cell-line), has explicit vehicle/DMSO controls, and carries drugs with a defensible cardiotoxic/non-cardiotoxic label (reuse the DToxS `Is_cardiotoxic` convention or a published call).
3. Compute a per-gene mean log2FC across its cardiotox+ conditions, z-score across all its genes (the exact `zscore()` in `panel_replication_clean_run.py`), and add it as a fifth entry in `zmap`.
4. Re-run the clean LODO as a 5-fold version and re-pool. A fifth independent, non-dox/non-TKI fold is the single highest-value strengthening available, because it directly attacks the "effective ~2.5 datasets" ceiling that the current pooled p leans on.

> **CORRECTED 2026-07-17 (T-356): GSE217421 cannot be that fold.** It is the same DToxS data (same 6 MSN lines, same 54 drugs) the existing DToxS fold already uses, re-hosted on GEO instead of the LINCS portal. Adding it would double-count a fold and inflate the pooled p rather than attack the ~2.5-dataset ceiling. Its value is a **licence-route swap** (GEO no-restriction in place of the portal's unconfirmed ToU) for a fold we already have, not independence. The search for a separate 5th non-dox/non-TKI iPSC-CM set is still open.

This is bounded by design: downloading and pipeline-matching a new dataset to machine precision is a multi-step job, not a one-iteration add, and fabricating a fold is forbidden. Documenting it as the next concrete step is the correct outcome under the budget guard.

## 4. Summary

The 81-gene cardiac-identity panel was selected by a Stouffer-Z cross-dataset meta-analysis (Bonferroni p<5e-7) over four independent iPSC-CM transcriptomic datasets. In a fully leakage-clean leave-one-dataset-out test (panel re-selected from scratch over the full transcriptome using only the other three datasets, held-out direction tested with a sign-permutation null), the held-out direction agrees with the re-selected consensus above chance in all four folds, and pooling the folds into a single statistic gives a generalization significance of p ≈ 1e-3 (joint sign-permutation across all 173 covered panel genes: 77.5% agreement observed vs 68.9% null, pooled empirical p ≤ 1.0e-4; Fisher combination p = 8.9e-4; Stouffer combination p = 3.2e-4). This pooled result does not rest on the single near-perfect fold (Burridge, 1.000) or on the underpowered fold (Wu-Liu, 9 genes): removing Burridge, the remaining folds still pool below p=0.05. The ceiling is that these combinations assume fold independence, and because two of the four datasets are both doxorubicin the effective independent replication is ~2.5 datasets, so the pooled p should be read as of order 1e-3 rather than 1e-4. The strongest single fully-independent fold remains held-out DToxS (multi-compound, 98 genes, 70% agreement, p=4.7e-3), unchanged from the canonical firewall doc. Every number here is reproducible with two commands: `panel_replication_clean_run.py` for the fold table and `lodo_pooled_stat.py` for the pooled statistics, both seed 0.

## Open flag

No canonical number changed. The DToxS fold reproduced at exactly 0.704 / p=4.7e-3, so this note does not collide with T-286's use of that figure. The one new, citable number is the **pooled cross-fold generalization p ≈ 1e-3** (Fisher 8.9e-4, Stouffer 3.2e-4, joint permutation ≤ 1e-4), which is a stronger and more defensible headline than the per-fold DToxS 4.7e-3 because it uses all four folds without leaning on any one. If you want it in downstream write-ups, the safe phrasing is "generalizes across four held-out folds at pooled p ≈ 1e-3 (effective ~2.5 independent datasets)," carrying the dependence caveat. The highest-value next step to push past the ~2.5-dataset ceiling is adding GSE217421 (or another non-dox, non-TKI iPSC-CM cardiotoxicity set) as a fifth fold; that is a dedicated download+pipeline-match task, not a one-iteration add.
