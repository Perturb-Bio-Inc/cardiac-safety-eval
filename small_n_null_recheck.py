#!/usr/bin/env python3
"""
Sensitivity re-check for the two externally cited small-benchmark results (T-511).

The permutation p-value is already computed against the pipeline's own null, so it is exact
under label exchangeability whatever the null mean is. This script asks the separate question
a reader will ask: if the null were forced to centre at 0.50 (i.e. if the downward shift were
removed), would the same candidates still clear the gate? It re-computes p and the p95 gate
against the null distribution shifted up by (0.50 - null_mean).

Run: ~/.venvs/myokit/bin/python small_n_null_recheck.py
"""
import json
import os
import numpy as np

import eval_core as E
import models as M
import benchmarks as B

HERE = os.path.dirname(os.path.abspath(__file__))

CASES = [
    ("enginev0_cardiotox", B.EngineV0Benchmark, ["magnitude_only", "all_features",
                                                 "panel_only", "panel_no_magnitude"],
     M.ENGINEV0_REGISTRY, lambda s, y: E.rank_auc(s, np.asarray(y) == 1)),
    ("cipa28_tdp", B.CiPABenchmark, ["herg_only_baseline", "multichannel_logreg",
                                     "kernik_only", "multichannel_plus_kernik"],
     M.CIPA_REGISTRY, lambda s, y: E.rank_auc(s, np.asarray(y) == 2)),
]

out = []
for bid, cls, names, reg, metric in CASES:
    bench = cls()
    X, y, g, _ = bench.dev()
    for nm in names:
        fs = reg[nm]
        r = E.permutation_null(fs, X, y, g, metric=metric, n=2000, seed=0, keep_dist=True)
        d = np.asarray(r["null_dist"])
        obs = r["observed"]
        shift = 0.50 - d.mean()
        dc = d + shift
        p_raw = (np.sum(d >= obs) + 1) / (len(d) + 1)
        p_corr = (np.sum(dc >= obs) + 1) / (len(dc) + 1)
        rec = dict(benchmark=bid, model=nm, dev_auc=round(obs, 4),
                   null_mean=round(float(d.mean()), 4), shift=round(float(shift), 4),
                   p_asrun=round(float(p_raw), 4), p95_asrun=round(float(np.percentile(d, 95)), 4),
                   p_recentred=round(float(p_corr), 4),
                   p95_recentred=round(float(np.percentile(dc, 95)), 4),
                   gate_asrun=bool(p_raw < 0.05 and obs > np.percentile(d, 95)),
                   gate_recentred=bool(p_corr < 0.05 and obs > np.percentile(dc, 95)))
        out.append(rec)
        print(json.dumps(rec))

with open(os.path.join(HERE, "small_n_null_recheck.json"), "w") as fh:
    json.dump(out, fh, indent=2)
print("wrote small_n_null_recheck.json")
