#!/usr/bin/env python3
"""
Candidate risk-scoring models, one registry per benchmark.

Every candidate is a `fit_score(X_tr, y_tr, X_te) -> scores_te` function: fit on the training
fold, return a continuous risk (higher = more risk) per test drug. That single signature lets
the eval core run them all through the identical firewall. Adding a model = adding one
function to a registry; the core decides whether it certifies.

numpy only; a tiny L2-regularized logistic regression stands in for sklearn.
"""
import numpy as np


def _standardize(Xtr, Xte):
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd == 0] = 1.0
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _fit_logreg(X, y_bin, l2=1.0, iters=800, lr=0.3):
    """L2-regularized logistic regression by gradient descent. Regularization is load-bearing
    on small correlated panels and on wide fingerprint spaces where an unpenalized fit
    separates and the decision score becomes meaningless."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    y = y_bin.astype(float)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
        w -= lr * (X.T @ (p - y) / n + l2 * w / n)
        b -= lr * np.mean(p - y)
    return w, b


def _logreg_scorer(cols, pos=2, l2=1.0):
    """Standardized logistic regression on selected feature columns (for tabular benchmarks)."""
    def fit_score(Xtr, ytr, Xte):
        Xtr2, Xte2 = _standardize(Xtr[:, cols], Xte[:, cols])
        w, b = _fit_logreg(Xtr2, (ytr == pos).astype(int), l2=l2)
        return Xte2 @ w + b
    return fit_score


def _column_scorer(col):
    """Parameter-free: risk = a single precomputed feature (no fit, no leakage surface)."""
    def fit_score(Xtr, ytr, Xte):
        return Xte[:, col]
    return fit_score


def _sum_scorer(cols):
    def fit_score(Xtr, ytr, Xte):
        return Xte[:, cols].sum(axis=1)
    return fit_score


def _fp_logreg_scorer(l2=1.0, iters=300, pos=1):
    """Logistic regression directly on binary Morgan fingerprints (no standardization)."""
    def fit_score(Xtr, ytr, Xte):
        w, b = _fit_logreg(Xtr, (ytr == pos).astype(int), l2=l2, iters=iters, lr=0.5)
        return Xte @ w + b
    return fit_score


def _delta_logreg_scorer(base_col, cols, pos=2, l2=1.0):
    """Logistic regression on variant-minus-healthy deltas: isolates the amplification a
    disease background adds over the healthy-cell score, which is the variant-conditioning signal."""
    def fit_score(Xtr, ytr, Xte):
        Dtr = np.column_stack([Xtr[:, c] - Xtr[:, base_col] for c in cols])
        Dte = np.column_stack([Xte[:, c] - Xte[:, base_col] for c in cols])
        Dtr2, Dte2 = _standardize(Dtr, Dte)
        w, b = _fit_logreg(Dtr2, (ytr == pos).astype(int), l2=l2)
        return Dte2 @ w + b
    return fit_score


def _bitcount_scorer():
    """Naive structural baseline: risk = number of on-bits (a crude molecular-size proxy)."""
    def fit_score(Xtr, ytr, Xte):
        return Xte.sum(axis=1).astype(float)
    return fit_score


# --- CiPA-28 (feature cols: 0 herg, 1 ical, 2 ina, 3 kernik) ---
CIPA_REGISTRY = {
    "herg_only_baseline":       _column_scorer(0),
    "multichannel_sum":         _sum_scorer([0, 1, 2]),
    "multichannel_logreg":      _logreg_scorer([0, 1, 2], pos=2),
    "kernik_only":              _column_scorer(3),
    "multichannel_plus_kernik": _logreg_scorer([0, 1, 2, 3], pos=2),
}

# --- engine-v0 (cols: 0 panel_score, 1 log_n_de, 2 n_up, 3 n_down). Baseline = pure magnitude:
# a candidate must beat log_n_de alone to claim real cardiac signal rather than a magnitude
# artifact. This is the question the original engine-v0 analysis left open. ---
ENGINEV0_REGISTRY = {
    "magnitude_only":       _logreg_scorer([1], pos=1),           # the confound / baseline
    "panel_only":           _logreg_scorer([0], pos=1),           # the claimed cardiac signal
    "panel_no_magnitude":   _logreg_scorer([0, 2, 3], pos=1),     # panel features, magnitude removed
    "all_features":         _logreg_scorer([0, 1, 2, 3], pos=1),
}

# --- DICTrank fingerprints. Baseline = naive bit-count; the loop's job is to beat it with
# learned models (several L2 settings = successive candidates). ---
DICTRANK_REGISTRY = {
    "fp_bitcount_baseline": _bitcount_scorer(),
    "fp_logreg_l2_3.0":     _fp_logreg_scorer(l2=3.0),
    "fp_logreg_l2_1.0":     _fp_logreg_scorer(l2=1.0),
    "fp_logreg_l2_0.3":     _fp_logreg_scorer(l2=0.3),
    "fp_logreg_l2_0.1":     _fp_logreg_scorer(l2=0.1),
}

# --- variant (cols: 0 healthy, 1 LQT2_mild, 2 LQT2_mod, 3 LQT1). Baseline = healthy background
# alone; a candidate must beat it to show variant-conditioning adds classification power. ---
VARIANT_REGISTRY = {
    "healthy_only":         _column_scorer(0),
    "variant_conditioned":  _logreg_scorer([0, 1, 2, 3], pos=2),
    "variant_deltas":       _delta_logreg_scorer(0, [1, 2, 3], pos=2),
    "lqt2_amplification":   _delta_logreg_scorer(0, [2], pos=2),
}

REGISTRIES = {
    "cipa28_tdp":        (CIPA_REGISTRY, "herg_only_baseline"),
    "enginev0_cardiotox": (ENGINEV0_REGISTRY, "magnitude_only"),
    "variant_tdp":       (VARIANT_REGISTRY, "healthy_only"),
    "dictrank_fp":       (DICTRANK_REGISTRY, "fp_bitcount_baseline"),
    # same candidates, random (molecule-level) split: the published-comparator condition
    "dictrank_fp_random": (DICTRANK_REGISTRY, "fp_bitcount_baseline"),
}
