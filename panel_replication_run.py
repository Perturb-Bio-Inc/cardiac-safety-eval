#!/usr/bin/env python3
"""
Permutation-null firewall for the "81-gene cardiac-identity panel replicates across
4 independent datasets" claim.

Mirrors the leakage-proof discipline of eval_core.py (permutation_null): define the
observed replication statistic, build an empirical null by shuffling gene identity /
labels, and report where the observed value sits in the null tail.

The panel-selection recipe (RICHER_PANEL_ANALYSIS.md, T-048) that this test interrogates:
  per-gene per-dataset logFC -> z-score within each dataset -> Stouffer Z = sum(z)/sqrt(n)
  -> panel = genes directionally consistent in >=3/4 datasets AND Bonferroni p<5e-7.
Selection therefore GUARANTEES >=3/4 sign agreement with the panel direction for every
member. Any honest firewall of "replicates across all four" has to escape that floor.

Three tests, in increasing leakage-robustness:
  A. Pairwise sign-concordance among the all-4 genes, vs a per-dataset sign-shuffle null.
     (Preserves each dataset's up/down marginal; asks whether the SAME genes go the SAME
     way across datasets beyond the marginal mix. Partly floored by the >=3/4 rule.)
  B. Leave-one-dataset-out (LODO): hold out one dataset, build the consensus direction
     from the other three, test whether the held-out dataset agrees, vs a shuffle null of
     the held-out signs. The direction tested against is held-out-independent, so this
     mostly escapes the selection floor. Residual leak: panel membership still used the
     held-out dataset, stated in the writeup.
  C. Disease-variant enrichment (GSE198258) recomputed from primary DEG, Fisher exact,
     plus a label-shuffle null (random equal-size gene sets from the measured background).

numpy + scipy + csv only. Deterministic given SEED.

Run:  ~/.venvs/myokit/bin/python panel_replication_run.py
"""
import os
import csv
import numpy as np
from scipy.stats import fisher_exact

SEED = 0
N_PERM = 10000
EV0 = os.environ.get("ENGINE_V0_DIR", "../engine-v0")  # set ENGINE_V0_DIR to your data checkout
PANEL_CSV = f"{EV0}/data/richer_panel_genes.csv"
DEG_XLS = f"{EV0}/data/variant_validation/GSE198258_DEG.xls"
DATASETS = ["DToxS", "Sharma", "Burridge", "WuLiu"]


def load_panel():
    rows = [r for r in csv.DictReader(open(PANEL_CSV)) if r["pass_bonferroni"] == "True"]
    genes = [r["gene"] for r in rows]
    dir_sign = np.array([-1 if r["direction"] == "DOWN" else 1 for r in rows])

    def zf(x):
        return float(x) if x not in ("", "nan", "NaN") else np.nan

    Z = {d: np.array([zf(r["z_" + d]) for r in rows]) for d in DATASETS}
    return genes, dir_sign, Z


# ----------------------------------------------------------------------------------
# Firewall A: pairwise sign-concordance among all-4 genes, sign-shuffle null
# ----------------------------------------------------------------------------------
def firewall_A(Z, rng):
    measured = np.all([~np.isnan(Z[d]) for d in DATASETS], axis=0)
    S = np.stack([np.sign(Z[d][measured]) for d in DATASETS], axis=1)  # (n, 4)
    n = S.shape[0]

    def mean_pairwise_concordance(sign_mat):
        agree = 0
        pairs = 0
        for i in range(4):
            for j in range(i + 1, 4):
                agree += np.mean(sign_mat[:, i] == sign_mat[:, j])
                pairs += 1
        return agree / pairs

    obs = mean_pairwise_concordance(S)
    null = np.empty(N_PERM)
    for k in range(N_PERM):
        Sp = np.empty_like(S)
        for c in range(4):
            Sp[:, c] = rng.permutation(S[:, c])  # shuffle gene identity within dataset
        null[k] = mean_pairwise_concordance(Sp)
    pval = (np.sum(null >= obs) + 1) / (N_PERM + 1)
    # also the unanimity fraction (floored by selection; reported for context only)
    frac_unanimous = np.mean(np.all(S == S[:, [0]], axis=1))
    return dict(n=int(n), observed=float(obs), null_mean=float(null.mean()),
                null_p95=float(np.percentile(null, 95)), null_max=float(null.max()),
                pval=float(pval), frac_unanimous=float(frac_unanimous))


# ----------------------------------------------------------------------------------
# Firewall B: leave-one-dataset-out held-out agreement, shuffle null on held-out signs
# ----------------------------------------------------------------------------------
def firewall_B(Z, rng):
    out = {}
    for held in DATASETS:
        others = [d for d in DATASETS if d != held]
        # genes measured in held-out AND in all three others
        meas = ~np.isnan(Z[held])
        for d in others:
            meas = meas & ~np.isnan(Z[d])
        idx = np.where(meas)[0]
        consensus = np.sign(np.sum([Z[d][idx] for d in others], axis=0))
        held_sign = np.sign(Z[held][idx])
        keep = consensus != 0
        consensus, held_sign = consensus[keep], held_sign[keep]
        n = len(consensus)
        obs = float(np.mean(held_sign == consensus))
        null = np.empty(N_PERM)
        for k in range(N_PERM):
            null[k] = np.mean(rng.permutation(held_sign) == consensus)
        pval = (np.sum(null >= obs) + 1) / (N_PERM + 1)
        out[held] = dict(n=int(n), observed=obs, null_mean=float(null.mean()),
                         null_p95=float(np.percentile(null, 95)), pval=float(pval))
    return out


# ----------------------------------------------------------------------------------
# Firewall C: GSE198258 disease-variant enrichment recompute + label-shuffle null
# ----------------------------------------------------------------------------------
def firewall_C(panel_genes, rng):
    with open(DEG_XLS) as f:
        rdr = csv.DictReader(f, delimiter="\t")
        rows = list(rdr)
    seen = set()
    deg = []  # (gene, padj)
    for r in rows:
        g, p = r.get("gene_name", ""), r.get("padj", "")
        if not g or g in seen or p in ("", "NA", "NaN", "nan"):
            continue
        try:
            pv = float(p)
        except ValueError:
            continue
        seen.add(g)
        deg.append((g, pv))
    SIG = 0.05
    in_panel = np.array([g in panel_genes for g, _ in deg])
    sig = np.array([pv < SIG for _, pv in deg])
    npm, npmsig = int(in_panel.sum()), int((sig & in_panel).sum())
    nbg = int((~in_panel).sum())
    nbgsig = int((sig & ~in_panel).sum())
    odds, p = fisher_exact([[npmsig, npm - npmsig], [nbgsig, nbg - nbgsig]],
                           alternative="greater")
    fold = (npmsig / npm) / (nbgsig / nbg)
    # label-shuffle null: draw random gene sets of size npm from all measured genes
    sig_all = sig.astype(int)
    N = len(deg)
    null_sig = np.empty(N_PERM)
    for k in range(N_PERM):
        pick = rng.choice(N, size=npm, replace=False)
        null_sig[k] = sig_all[pick].sum()
    perm_p = (np.sum(null_sig >= npmsig) + 1) / (N_PERM + 1)
    return dict(n_deg=len(deg), panel_measured=npm, panel_sig=npmsig,
                panel_rate=npmsig / npm, bg_measured=nbg, bg_sig=nbgsig,
                bg_rate=nbgsig / nbg, odds=float(odds), fold=float(fold),
                fisher_p=float(p), perm_null_mean=float(null_sig.mean()),
                perm_null_max=float(null_sig.max()), perm_p=float(perm_p))


def main():
    rng = np.random.default_rng(SEED)
    genes, dir_sign, Z = load_panel()
    panel_genes = set(genes)

    print(f"Panel: {len(genes)} Bonferroni genes; "
          f"{int(np.all([~np.isnan(Z[d]) for d in DATASETS], axis=0).sum())} measured in all 4\n")

    A = firewall_A(Z, rng)
    print("FIREWALL A - pairwise sign-concordance vs gene-identity shuffle null")
    print(f"  n genes (all 4) = {A['n']}")
    print(f"  observed mean pairwise concordance = {A['observed']:.4f}")
    print(f"  null mean / p95 / max = {A['null_mean']:.4f} / {A['null_p95']:.4f} / {A['null_max']:.4f}")
    print(f"  empirical p = {A['pval']:.2e}   (unanimity frac, floored context = {A['frac_unanimous']:.3f})\n")

    B = firewall_B(Z, rng)
    print("FIREWALL B - leave-one-dataset-out held-out agreement (consensus from other 3)")
    for d in DATASETS:
        b = B[d]
        print(f"  hold out {d:9s} n={b['n']:3d}  observed={b['observed']:.3f}  "
              f"null_mean={b['null_mean']:.3f}  null_p95={b['null_p95']:.3f}  p={b['pval']:.2e}")
    print()

    C = firewall_C(panel_genes, rng)
    print("FIREWALL C - GSE198258 disease-variant enrichment (recomputed from primary DEG)")
    print(f"  {C['n_deg']} genes; panel {C['panel_measured']} measured, {C['panel_sig']} sig "
          f"({100*C['panel_rate']:.1f}%) vs bg {100*C['bg_rate']:.1f}%")
    print(f"  odds={C['odds']:.2f}  fold={C['fold']:.2f}x  Fisher one-sided p={C['fisher_p']:.3e}")
    print(f"  label-shuffle null: mean sig in random 79-gene sets = {C['perm_null_mean']:.1f}, "
          f"max over {N_PERM} = {C['perm_null_max']:.0f}; empirical p = {C['perm_p']:.2e}")


if __name__ == "__main__":
    main()
