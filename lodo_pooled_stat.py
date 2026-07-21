#!/usr/bin/env python3
"""
Pooled cross-fold statistic for the fully-clean LODO (Firewall D).

Companion to panel_replication_clean_run.py. Does NOT re-derive or change the
canonical per-fold numbers; it imports that script's loaders/selection and adds:

  1. A joint sign-permutation pooled test across all four folds that respects each
     fold's own null skew (the honest single generalization number, does not lean
     on the one near-perfect Burridge fold).
  2. Parametric cross-checks (Fisher's method, Stouffer's Z) on the four per-fold p.
  3. A diagnostic of the underpowered Wu-Liu fold (is the limiter panel size or
     held-out coverage?).

numpy + scipy only. Deterministic given SEED (same seed as the canonical run).
Run:  ~/.venvs/myokit/bin/python lodo_pooled_stat.py
"""
import numpy as np
from scipy.stats import norm, chi2

import panel_replication_clean_run as C

SEED = 0
N_PERM = 10000
FOLDS = ["DToxS", "Sharma", "Burridge", "WuLiu"]


def build_zmap():
    dtoxs_logfc, dtoxs_drugs = C.load_dtoxs_full()
    zmap = {"DToxS": C.zscore(dtoxs_logfc),
            "Sharma": C.zscore(C.load_sharma()),
            "Burridge": C.zscore(C.load_burridge()),
            "WuLiu": C.zscore(C.load_wuliu())}
    return zmap, dtoxs_drugs


def fold_vectors(held_out, zmap):
    """Replicate panel_replication_clean_run.clean_lodo selection, but return the
    per-gene consensus-sign and held-out-sign vectors so folds can be pooled."""
    selectors = [d for d in FOLDS if d != held_out]
    union = sorted(set().union(*[set(zmap[s]) for s in selectors]))
    stouffer, direction = {}, {}
    for g in union:
        zs = [zmap[s][g] for s in selectors if g in zmap[s]]
        if len(zs) < 3:
            continue
        zsum = float(np.sum(zs))
        stouffer[g] = zsum / np.sqrt(len(zs))
        direction[g] = np.sign(zsum)
    panel = []
    for g, st in stouffer.items():
        p_two = 2 * (1 - norm.cdf(abs(st)))
        signs = np.sign([zmap[s][g] for s in selectors])
        unanimous = len(set(signs)) == 1 and 0 not in signs
        if p_two < C.BONF and unanimous:
            panel.append(g)
    held_z = zmap[held_out]
    tested = [g for g in panel if g in held_z]
    cons = np.array([direction[g] for g in tested])
    held = np.array([np.sign(held_z[g]) for g in tested])
    return dict(held_out=held_out, panel_size=len(panel), tested=tested,
                cons=cons, held=held, covered=len(tested))


def main():
    rng = np.random.default_rng(SEED)
    print("Loading full-background datasets...")
    zmap, dtoxs_drugs = build_zmap()

    folds = {h: fold_vectors(h, zmap) for h in FOLDS}

    # Per-fold observed + independent per-fold null p (matches canonical run).
    print("\n=== Per-fold recap (should match panel_replication_clean_run.py) ===")
    per_p = {}
    for h in FOLDS:
        f = folds[h]
        obs = float(np.mean(f["held"] == f["cons"]))
        null = np.array([np.mean(rng.permutation(f["held"]) == f["cons"]) for _ in range(N_PERM)])
        p = (np.sum(null >= obs) + 1) / (N_PERM + 1)
        per_p[h] = p
        agree = int(round(obs * f["covered"]))
        print(f"  {h:9s} panel={f['panel_size']:3d} covered={f['covered']:3d} "
              f"obs={obs:.3f} ({agree}/{f['covered']}) null_mean={null.mean():.3f} p={p:.2e}")

    # ---- 1. Joint pooled sign-permutation test across all four folds ----
    # Statistic = total agreements summed across the four folds' covered panel genes.
    # Under the joint null, each fold's held vector is permuted independently, so the
    # pooled null already carries each fold's own baseline skew (a fold whose null
    # agrees 92% of the time contributes ~92% agreements even under the null). This is
    # why a fold near 1.000 cannot dominate: its null is near 1.000 too.
    obs_total = int(sum(np.sum(folds[h]["held"] == folds[h]["cons"]) for h in FOLDS))
    total_genes = int(sum(folds[h]["covered"] for h in FOLDS))
    null_totals = np.zeros(N_PERM, dtype=int)
    for k in range(N_PERM):
        s = 0
        for h in FOLDS:
            f = folds[h]
            s += int(np.sum(rng.permutation(f["held"]) == f["cons"]))
        null_totals[k] = s
    pooled_p = (np.sum(null_totals >= obs_total) + 1) / (N_PERM + 1)
    null_mean = null_totals.mean()
    print("\n=== 1. Joint pooled sign-permutation test (all four folds, one number) ===")
    print(f"  observed total agreements: {obs_total}/{total_genes} = {obs_total/total_genes:.3f}")
    print(f"  joint-null mean total: {null_mean:.1f}/{total_genes} = {null_mean/total_genes:.3f}")
    print(f"  joint-null max: {null_totals.max()}   pooled empirical p = {pooled_p:.2e}"
          f"  ({N_PERM} permutations, seed {SEED})")

    # ---- 2. Parametric combinations of the four per-fold p-values ----
    ps = np.array([per_p[h] for h in FOLDS])
    # Fisher
    fisher_x = -2.0 * np.sum(np.log(ps))
    fisher_df = 2 * len(ps)
    fisher_p = chi2.sf(fisher_x, fisher_df)
    # Stouffer (equal weights)
    zs = norm.isf(ps)
    stouffer_z = np.sum(zs) / np.sqrt(len(ps))
    stouffer_p = norm.sf(stouffer_z)
    print("\n=== 2. Parametric p-value combinations (assume fold independence) ===")
    print(f"  per-fold p: " + ", ".join(f"{h}={per_p[h]:.2e}" for h in FOLDS))
    print(f"  Fisher X2={fisher_x:.2f} (df={fisher_df}) -> p={fisher_p:.2e}")
    print(f"  Stouffer Z={stouffer_z:.3f} -> p={stouffer_p:.2e}")
    print("  NOTE: both assume the four folds are independent. Burridge and Wu-Liu are")
    print("  both doxorubicin, so effective independent replication is ~2.5 datasets;")
    print("  treat these parametric combined-p as optimistic (lower bounds on the true p).")

    # ---- 3. Wu-Liu underpowered-fold diagnostic ----
    wl = folds["WuLiu"]
    # how many of the DToxS+Sharma+Burridge-selected panel genes are NOT in Wu-Liu?
    sel = [d for d in FOLDS if d != "WuLiu"]
    union = sorted(set().union(*[set(zmap[s]) for s in sel]))
    # recompute full panel gene list to inspect coverage
    fpanel = wl["panel_size"]
    print("\n=== 3. Wu-Liu fold diagnostic (is the limiter coverage or panel size?) ===")
    print(f"  panel re-selected from DToxS+Sharma+Burridge: {fpanel} genes")
    print(f"  of those, covered by Wu-Liu background: {wl['covered']}  (uncovered: {fpanel - wl['covered']})")
    print("  => the fold is capped by PANEL SIZE, not by Wu-Liu coverage: even at 100%")
    print("     Wu-Liu coverage the fold would test <=", fpanel, "genes. Recovering more")
    print("     Wu-Liu genes cannot deepen it; only a larger DToxS+Sharma+Burridge")
    print("     unanimous panel could, which would require relaxing the Bonferroni cut")
    print("     (a change to the selection rule, not leakage-clean-comparable).")


if __name__ == "__main__":
    main()
