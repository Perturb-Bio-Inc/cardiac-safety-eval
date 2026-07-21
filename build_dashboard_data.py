#!/usr/bin/env python3
"""
Assemble the self-contained data blob the dashboard embeds: the loop summary plus, for each
benchmark, the permutation-null distribution of its headline model (so the dashboard can draw
the observed-vs-null histogram without any external calls).

Run after run_loop.py. Writes dashboard_data.json.
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_core as E     # noqa: E402
import benchmarks as B    # noqa: E402
import models as M        # noqa: E402

BENCH_CLASS = {"cipa28_tdp": B.CiPABenchmark, "enginev0_cardiotox": B.EngineV0Benchmark,
               "variant_tdp": B.VariantBenchmark, "dictrank_fp": B.DICTrankBenchmark}
N_NULL = {"cipa28_tdp": 1000, "enginev0_cardiotox": 1000, "variant_tdp": 1000, "dictrank_fp": 200}


def main():
    summ = json.load(open(os.path.join(HERE, "loop_summary.json")))
    nulls = {}
    for bid, b in summ["benchmarks"].items():
        passed = [c for c in b["candidates"] if c["passed"]]
        head = max(passed or b["candidates"], key=lambda c: c["locked_auc"])
        bench = BENCH_CLASS[bid]()
        registry, _ = M.REGISTRIES[bid]
        Xd, yd, gd, _ = bench.dev()
        perm = E.permutation_null(registry[head["name"]], Xd, yd, gd,
                                  metric=E.auc_for(bench.pos_mask), n=N_NULL[bid], keep_dist=True)
        nulls[bid] = dict(model=head["name"], observed=round(perm["observed"], 4),
                          pval=round(perm["pval"], 4),
                          null=[round(x, 4) for x in perm["null_dist"]])
        print(f"{bid}: headline={head['name']} observed={perm['observed']:.3f} p={perm['pval']:.3f}")
    out = dict(run_ts=summ["run_ts"], benchmarks=summ["benchmarks"], nulls=nulls)
    json.dump(out, open(os.path.join(HERE, "dashboard_data.json"), "w"))
    print("wrote dashboard_data.json")


if __name__ == "__main__":
    main()
