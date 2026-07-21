#!/usr/bin/env python3
"""
Synthetic probe for the sub-0.5 permutation null (T-511).

The README caveat asserts the small-benchmark permutation null "centers slightly below 0.50
(an overfitting-under-shuffle effect, not leakage)". This script tests that assertion by
running the harness's OWN permutation machinery (eval_core.permutation_null / grouped_oof_scores,
unmodified) on features drawn independent of the labels, so there is no signal to find and any
departure from 0.50 is a property of the estimator, not the data.

Four probes:
  A. real fold structure (CiPA dev, engine-v0 dev), noise features, models from a fit-free
     scorer up to a 16-parameter logistic  -> is the shift flexibility-driven?
  B. n sweep at fixed group count and fixed model                -> how does the shift scale with n?
  C. group-count sweep at fixed n                                -> how does it scale with folds?
  D. within-fold vs between-fold AUC decomposition               -> where does the shift live?

Run: ~/.venvs/myokit/bin/python small_n_null_probe.py
Writes small_n_null_probe.json next to this file. Does not touch eval_core.py.
"""
import json
import os
import numpy as np

import eval_core as E
import models as M
import benchmarks as B

HERE = os.path.dirname(os.path.abspath(__file__))
RNG_FEATURES = 12345


def noise_X(n, d, seed):
    return np.random.default_rng(seed).normal(size=(n, d))


def scorers(d, pos=1):
    """Candidates ordered by number of fitted parameters. `pos` must match the metric's
    positive class or the fitted models are scored against a class they were not fit for."""
    out = [("fit_free_col0", 0, M._column_scorer(0)),
           ("sum_cols_fitfree", 0, M._sum_scorer(list(range(min(3, d)))))]
    for k in [1, 2, 3, 4, 8, 16]:
        if k <= d:
            out.append((f"logreg_{k}col", k + 1, M._logreg_scorer(list(range(k)), pos=pos)))
    return out


def null_mean_for(fit_score, X, y, groups, metric, n_null, seed):
    r = E.permutation_null(fit_score, X, y, groups, metric=metric, n=n_null, seed=seed,
                           keep_dist=True)
    dist = np.asarray(r["null_dist"])
    return dict(null_mean=float(dist.mean()),
                se=float(dist.std(ddof=1) / np.sqrt(len(dist))),
                null_p95=float(np.percentile(dist, 95)),
                frac_below_half=float(np.mean(dist < 0.5)),
                n=len(dist))


# ------------------------------------------------------------------ probe A
def probe_A(n_null=2000):
    res = {}
    cipa = B.CiPABenchmark()
    ev = B.EngineV0Benchmark()
    cases = [
        ("cipa28_tdp_dev", cipa.dev(), lambda s, y: E.rank_auc(s, np.asarray(y) == 2), 2),
        ("enginev0_dev", ev.dev(), lambda s, y: E.rank_auc(s, np.asarray(y) == 1), 1),
    ]
    for label, (Xreal, y, g, _), metric, poscls in cases:
        n = len(y)
        d = 16
        X = noise_X(n, d, RNG_FEATURES)
        rows = []
        for name, npar, fs in scorers(d, pos=poscls):
            r = null_mean_for(fs, X, y, g, metric, n_null, seed=0)
            r.update(model=name, n_params=npar)
            rows.append(r)
            print(f"[A] {label:16s} n={n:3d} k={len(np.unique(g))} {name:18s} "
                  f"params={npar:2d} null_mean={r['null_mean']:.4f} +/- {r['se']:.4f}")
        res[label] = dict(n=int(n), n_groups=int(len(np.unique(g))),
                          class_counts=np.bincount(y).tolist(), rows=rows)
    return res


# ------------------------------------------------------------------ probe B / C
def synth_labels(n, prevalence, seed):
    npos = max(1, int(round(prevalence * n)))
    y = np.zeros(n, int)
    y[:npos] = 1
    return np.random.default_rng(seed).permutation(y)


def blocks(n, k):
    return np.arange(n) % k


def probe_B(n_null=1500, k=4, prevalence=1 / 3.0, n_list=(12, 16, 22, 28, 40, 60, 100, 200)):
    rows = []
    metric = lambda s, y: E.rank_auc(s, np.asarray(y) == 1)
    fs = M._logreg_scorer([0, 1, 2], pos=1)
    for n in n_list:
        y = synth_labels(n, prevalence, seed=n)
        g = blocks(n, k)
        X = noise_X(n, 3, RNG_FEATURES + n)
        r = null_mean_for(fs, X, y, g, metric, n_null, seed=0)
        r.update(n=int(n), k=k)
        rows.append(r)
        print(f"[B] n={n:4d} k={k} null_mean={r['null_mean']:.4f} +/- {r['se']:.4f}")
    return rows


def probe_C(n_null=1500, n=28, prevalence=1 / 3.0, k_list=(2, 3, 4, 7, 14, 28)):
    rows = []
    metric = lambda s, y: E.rank_auc(s, np.asarray(y) == 1)
    fs = M._logreg_scorer([0, 1, 2], pos=1)
    y = synth_labels(n, prevalence, seed=7)
    X = noise_X(n, 3, RNG_FEATURES + 7)
    for k in k_list:
        g = blocks(n, k)
        r = null_mean_for(fs, X, y, g, metric, n_null, seed=0)
        r.update(n=int(n), k=int(k))
        rows.append(r)
        print(f"[C] n={n} k={k:3d} (fold size {n/k:.1f}) null_mean={r['null_mean']:.4f} "
              f"+/- {r['se']:.4f}")
    return rows


# ------------------------------------------------------------------ probe D
def split_auc(scores, pos, groups):
    """AUC decomposed into pos/neg pairs that share a test fold vs pairs that do not."""
    s, p, g = np.asarray(scores, float), np.asarray(pos, bool), np.asarray(groups)
    ip, ineg = np.where(p)[0], np.where(~p)[0]
    within = [[], []]
    between = [[], []]
    for a in ip:
        for b in ineg:
            v = 1.0 if s[a] > s[b] else (0.5 if s[a] == s[b] else 0.0)
            (within if g[a] == g[b] else between)[0].append(v)
    return (float(np.mean(within[0])) if within[0] else float("nan"),
            float(np.mean(between[0])) if between[0] else float("nan"),
            len(within[0]), len(between[0]))


def probe_D(n_perm=2000):
    out = {}
    cipa = B.CiPABenchmark()
    ev = B.EngineV0Benchmark()
    cases = [("cipa28_tdp_dev", cipa.dev(), 2, M._logreg_scorer([0, 1, 2], pos=2)),
             ("enginev0_dev", ev.dev(), 1, M._logreg_scorer([0, 1, 2], pos=1))]
    for label, (Xreal, y, g, _), poscls, fs in cases:
        n = len(y)
        X = noise_X(n, 3, RNG_FEATURES)
        rng = np.random.default_rng(0)
        w, bt = [], []
        for _ in range(n_perm):
            yp = rng.permutation(y)
            oof = E.grouped_oof_scores(fs, X, yp, g, seed=0)
            a, b, nw, nb = split_auc(oof, yp == poscls, g)
            if not np.isnan(a):
                w.append(a)
            if not np.isnan(b):
                bt.append(b)
        out[label] = dict(within_fold_auc=float(np.mean(w)),
                          within_se=float(np.std(w, ddof=1) / np.sqrt(len(w))),
                          between_fold_auc=float(np.mean(bt)),
                          between_se=float(np.std(bt, ddof=1) / np.sqrt(len(bt))),
                          n_perm_within=len(w), n_perm_between=len(bt))
        print(f"[D] {label:16s} within-fold AUC={out[label]['within_fold_auc']:.4f} "
              f"+/- {out[label]['within_se']:.4f}   between-fold AUC="
              f"{out[label]['between_fold_auc']:.4f} +/- {out[label]['between_se']:.4f}")
    return out


if __name__ == "__main__":
    res = dict(A=probe_A(), B=probe_B(), C=probe_C(), D=probe_D())
    with open(os.path.join(HERE, "small_n_null_probe.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("wrote small_n_null_probe.json")
