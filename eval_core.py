#!/usr/bin/env python3
"""
Evaluation core for cardiac drug-safety models.

This module decides whether a model's score can be trusted. It does not care which model
wins. Three checks do the work:

  1. HELD-OUT LOCKED SPLIT. Each benchmark splits into `dev` and `locked`. Candidates are
     fitted, compared and selected on `dev` only. The locked split is scored for three
     models: the pre-declared baseline, a random canary, and the one candidate selected on
     dev. Each benchmark builds dev and locked as disjoint sets of rows. Near-duplicate
     molecules can still fall on both sides of a random split.

  2. SHUFFLED-LABEL NULL. Shuffle the labels many times and run the whole pipeline (fit,
     out-of-fold score, metric) on each shuffle. That gives a null distribution. A real
     signal sits far in its tail. A pipeline that leaks scores well even on shuffled labels,
     and the null shows it. Reported as an empirical p-value and the null's 95th percentile.

  3. BASELINE GATE. A candidate passes only if its permutation p-value is below 0.05, its
     dev AUC exceeds the null's 95th percentile, and its locked AUC exceeds the baseline's
     locked AUC.

AUC is rank-based and needs no threshold. With data this small, fitting a threshold is one
more way to overfit.

Requires numpy only (scipy is used elsewhere in the repo). Deterministic given a seed.
"""
import os
import json
import hashlib
import datetime as _dt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "leaderboard.jsonl")


# --------------------------------------------------------------------------------------
# Metrics (rank-based, threshold-free)
# --------------------------------------------------------------------------------------
def rank_auc(scores, positive):
    """AUC that `scores` rank the binary `positive` mask above the rest (Mann-Whitney U,
    tie-corrected via average ranks). Returns NaN if a class is empty."""
    s = np.asarray(scores, float)
    y = np.asarray(positive).astype(bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return (ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def high_vs_rest_auc(scores, y_ordinal):
    return rank_auc(scores, np.asarray(y_ordinal) == 2)


def low_vs_rest_auc(scores, y_ordinal):
    # "safe drug" call: low risk should get the LOWEST scores, so rank the non-low above low
    return rank_auc(scores, np.asarray(y_ordinal) != 0)


def auc_for(pos_mask):
    """Build a metric(scores, y) = rank AUC of scores separating pos_mask(y) from the rest.
    Lets each benchmark declare its own positive class (High for CiPA, cardiotox for binary)."""
    return lambda scores, y: rank_auc(scores, pos_mask(y))


def bootstrap_ci(scores, y_ordinal, metric=high_vs_rest_auc, n=2000, seed=0, alpha=0.05):
    """Percentile bootstrap CI for a metric. Resamples drugs with replacement."""
    rng = np.random.default_rng(seed)
    s, y = np.asarray(scores, float), np.asarray(y_ordinal)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(s), len(s))
        v = metric(s[idx], y[idx])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


# --------------------------------------------------------------------------------------
# Grouped cross-validation (leakage-proof out-of-fold scoring on dev)
# --------------------------------------------------------------------------------------
def grouped_oof_scores(fit_score, X, y, groups, seed=0):
    """Leave-one-group-out out-of-fold risk scores. `fit_score(X_tr, y_tr, X_te) -> scores_te`
    is the candidate: it fits on the training fold and returns a continuous risk per test
    drug. Grouping by mechanism/scaffold means a drug is never tested while a mechanistic
    sibling trains, which is where random CV leaks optimism on small correlated panels."""
    X, y, groups = np.asarray(X, float), np.asarray(y), np.asarray(groups)
    oof = np.full(len(y), np.nan)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        oof[te] = fit_score(X[tr], y[tr], X[te])
    return oof


# --------------------------------------------------------------------------------------
# The anti-spurious centerpiece: label-permutation null
# --------------------------------------------------------------------------------------
def permutation_null(fit_score, X, y, groups, metric=high_vs_rest_auc, n=1000, seed=0,
                     keep_dist=False):
    """Run the full grouped-OOF pipeline on `n` label shuffles and return the null
    distribution of `metric` plus the observed value and an empirical p-value.

    A trustworthy pipeline returns a null centered near 0.5 (chance). If the null mass
    sits well above 0.5, the pipeline is leaking information about the labels through the
    features/CV and NO reported score from it can be believed. This is the single check
    that would have caught the engine-v0 magnitude artifact."""
    rng = np.random.default_rng(seed)
    X, y, groups = np.asarray(X, float), np.asarray(y), np.asarray(groups)
    observed = metric(grouped_oof_scores(fit_score, X, y, groups, seed=seed), y)
    null = []
    for i in range(n):
        yp = rng.permutation(y)
        v = metric(grouped_oof_scores(fit_score, X, yp, groups, seed=seed), yp)
        if not np.isnan(v):
            null.append(v)
    null = np.asarray(null)
    # one-sided p: fraction of null at least as extreme as observed (+1 smoothing)
    pval = (np.sum(null >= observed) + 1) / (len(null) + 1)
    out = dict(observed=float(observed),
               null_mean=float(np.mean(null)), null_p95=float(np.percentile(null, 95)),
               null_max=float(np.max(null)), pval=float(pval), n_null=int(len(null)))
    if keep_dist:
        out["null_dist"] = null.tolist()
    return out


# --------------------------------------------------------------------------------------
# Certification: score the LOCKED split exactly once, gate, and log
# --------------------------------------------------------------------------------------
def _config_hash(name, benchmark_id, code_path):
    h = hashlib.sha256()
    h.update(name.encode())
    h.update(benchmark_id.encode())
    if code_path and os.path.exists(code_path):
        h.update(open(code_path, "rb").read())
    return h.hexdigest()[:12]


def evaluate_dev(fit_score, bench, seed=0, n_null=1000, n_boot=2000, metric=high_vs_rest_auc):
    """Dev-only evaluation: grouped out-of-fold AUC, its bootstrap CI, and the
    label-permutation null. Touches nothing in the locked split, so it is safe to run on
    every candidate and to select on."""
    Xd, yd, gd, _ = bench.dev()
    perm = permutation_null(fit_score, Xd, yd, gd, metric=metric, n=n_null, seed=seed)
    oof = grouped_oof_scores(fit_score, Xd, yd, gd, seed=seed)
    dev_auc = float(metric(oof, yd))
    dev_ci = bootstrap_ci(oof, yd, metric=metric, n=n_boot, seed=seed)
    beats_null = bool((perm["pval"] < 0.05) and (dev_auc > perm["null_p95"]))
    return dict(perm=perm, dev_auc=dev_auc, dev_ci=dev_ci, beats_null=beats_null)


def certify(name, bench, fit_score, code_path=None, seed=0, n_null=1000, n_boot=2000,
            baseline_locked_auc=None, metric=high_vs_rest_auc, log=True, iteration=None,
            run_ts=None, dev=None, score_locked=True, role="candidate"):
    """Evaluate one model and append the record to the leaderboard.

    bench must expose: dev() and locked() -> (X, y, groups, drug_names), and .id.
    fit_score(X_tr, y_tr, X_te) -> scores_te is the model.

    `dev` takes a precomputed evaluate_dev() result, so a candidate is not re-run.
    `score_locked` controls whether the locked split is touched. run_loop.py scores it only
    for the pre-declared baseline, the random canary, and the single candidate selected on
    dev. Every other candidate gets locked_auc = None and cannot pass.

    A candidate PASSES only if, on dev: p < 0.05 in the permutation null AND observed OOF
    AUC exceeds the null p95; and on the locked split: AUC beats the supplied baseline
    (if any)."""
    if dev is None:
        dev = evaluate_dev(fit_score, bench, seed=seed, n_null=n_null, n_boot=n_boot,
                           metric=metric)
    perm, dev_auc, dev_ci, beats_null = dev["perm"], dev["dev_auc"], dev["dev_ci"], dev["beats_null"]

    locked_auc = locked_ci = None
    locked_per_drug = None
    beats_baseline = None
    if score_locked:
        # fit on all of dev, score locked once
        Xd, yd, _, _ = bench.dev()
        Xl, yl, _, locked_names = bench.locked()
        locked_scores = fit_score(Xd, yd, Xl)
        locked_auc = float(metric(locked_scores, yl))
        locked_ci = bootstrap_ci(locked_scores, yl, metric=metric, n=n_boot, seed=seed)
        beats_baseline = True if baseline_locked_auc is None else (locked_auc > baseline_locked_auc)
        locked_per_drug = [[n, int(t), round(float(sc), 4)]
                           for n, t, sc in zip(locked_names, yl, locked_scores)]
    passed = bool(beats_null and beats_baseline)

    rec = dict(
        name=name, benchmark=bench.id, config_hash=_config_hash(name, bench.id, code_path),
        iteration=iteration, run_ts=run_ts, role=role,
        timestamp=_dt.datetime.now().isoformat(timespec="seconds"), seed=seed,
        dev_auc=round(dev_auc, 4), dev_ci=[round(dev_ci[0], 4), round(dev_ci[1], 4)],
        locked_scored=bool(score_locked),
        locked_auc=None if locked_auc is None else round(locked_auc, 4),
        locked_ci=None if locked_ci is None else [round(locked_ci[0], 4), round(locked_ci[1], 4)],
        permutation=dict(pval=round(perm["pval"], 4), null_mean=round(perm["null_mean"], 4),
                         null_p95=round(perm["null_p95"], 4), n=perm["n_null"]),
        gates=dict(beats_null=bool(beats_null),
                   beats_baseline=None if beats_baseline is None else bool(beats_baseline)),
        passed=passed,
        locked_per_drug=locked_per_drug,
    )
    if log:
        with open(LEDGER, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    return rec
