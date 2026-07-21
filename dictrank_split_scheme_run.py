#!/usr/bin/env python3
"""
How much does the scaffold split cost us on DICTrank?

The leaderboard carries one seed-0 run per split scheme (dictrank_fp, dictrank_fp_random),
certified through the normal path. A single locked set of 249 drugs has a wide CI, so a
one-draw difference between the two cannot settle the question on its own. This re-runs the
locked evaluation of the winning candidate across several seeds under both schemes and
reports the paired per-seed difference, which is the number the comparator doc needs.

No permutation null here (the seed-0 leaderboard entries carry that); no leaderboard writes.

Usage: ~/.venvs/myokit/bin/python dictrank_split_scheme_run.py
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_core as E     # noqa: E402
import benchmarks as B    # noqa: E402
import models as M        # noqa: E402

CANDIDATE = "fp_logreg_l2_3.0"
BASELINE = "fp_bitcount_baseline"
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]


def locked_auc(bench, fit_score, metric, seed):
    Xd, yd, _, _ = bench.dev()
    Xl, yl, _, _ = bench.locked()
    s = fit_score(Xd, yd, Xl)
    return float(metric(s, yl)), E.bootstrap_ci(s, yl, metric=metric, n=1000, seed=seed)


def main():
    reg, _ = M.REGISTRIES["dictrank_fp"]
    rows = []
    for seed in SEEDS:
        rec = {"seed": seed}
        for scheme in ("scaffold", "random"):
            bench = B.DICTrankBenchmark(seed=seed, split=scheme)
            metric = E.auc_for(bench.pos_mask)
            auc, ci = locked_auc(bench, reg[CANDIDATE], metric, seed)
            b_auc, _ = locked_auc(bench, reg[BASELINE], metric, seed)
            rec[scheme] = {"locked_auc": round(auc, 4),
                           "locked_ci": [round(ci[0], 4), round(ci[1], 4)],
                           "baseline_locked_auc": round(b_auc, 4)}
        rec["delta_random_minus_scaffold"] = round(
            rec["random"]["locked_auc"] - rec["scaffold"]["locked_auc"], 4)
        rows.append(rec)
        print(f"seed {seed}: scaffold {rec['scaffold']['locked_auc']:.4f}  "
              f"random {rec['random']['locked_auc']:.4f}  "
              f"delta {rec['delta_random_minus_scaffold']:+.4f}")

    d = np.array([r["delta_random_minus_scaffold"] for r in rows])
    sc = np.array([r["scaffold"]["locked_auc"] for r in rows])
    rd = np.array([r["random"]["locked_auc"] for r in rows])
    # paired sign test over seeds: how often does random beat scaffold?
    n_pos = int((d > 0).sum())
    summary = {
        "candidate": CANDIDATE, "seeds": SEEDS,
        "scaffold_mean": round(float(sc.mean()), 4), "scaffold_sd": round(float(sc.std(ddof=1)), 4),
        "random_mean": round(float(rd.mean()), 4), "random_sd": round(float(rd.std(ddof=1)), 4),
        "delta_mean": round(float(d.mean()), 4), "delta_sd": round(float(d.std(ddof=1)), 4),
        "delta_min": round(float(d.min()), 4), "delta_max": round(float(d.max()), 4),
        "n_seeds_random_higher": n_pos, "n_seeds": len(SEEDS),
    }
    print("\n" + json.dumps(summary, indent=2))
    out = os.path.join(HERE, "dictrank_split_scheme_2026-07-20.json")
    json.dump({"summary": summary, "per_seed": rows}, open(out, "w"), indent=2)
    print(f"wrote {os.path.basename(out)}")


if __name__ == "__main__":
    main()
