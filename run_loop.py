#!/usr/bin/env python3
"""
Drive every candidate through the leakage-proof eval core, per benchmark, gate it, and log a
sequenced leaderboard the dashboard reads. One invocation = one loop pass over the current
candidate set. A future autonomous loop adds candidates to models.py and re-runs; the core
guarantees a new idea cannot certify unless it beats the benchmark's baseline and the
permutation null on drugs it was never fit on.

Usage: ~/.venvs/myokit/bin/python run_loop.py [cipa|enginev0|dictrank ...]   (default: all)
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_core as E          # noqa: E402
import benchmarks as B         # noqa: E402
import models as M             # noqa: E402

CONF = {
    "cipa":     (B.CiPABenchmark,   dict(n_null=1000, n_boot=2000)),
    "enginev0": (B.EngineV0Benchmark, dict(n_null=1000, n_boot=2000)),
    "variant":  (B.VariantBenchmark, dict(n_null=1000, n_boot=2000)),
    "dictrank": (B.DICTrankBenchmark, dict(n_null=200,  n_boot=1000)),
    "dictrank_random": (lambda: B.DICTrankBenchmark(split="random"),
                        dict(n_null=200, n_boot=1000)),
}


def random_canary(seed):
    """Pure noise. Must FAIL every gate: it proves the guardrail rejects chance rather than
    rubber-stamping whatever runs."""
    def fit_score(Xtr, ytr, Xte):
        return np.random.default_rng(seed + len(Xte)).standard_normal(len(Xte))
    return fit_score


def run_benchmark(bench, n_null, n_boot, seed=0, run_ts=None):
    registry, baseline_name = M.REGISTRIES[bench.id]
    metric = E.auc_for(bench.pos_mask)
    code = os.path.join(HERE, "models.py")

    base = E.certify(baseline_name, bench, registry[baseline_name], code_path=code, seed=seed,
                     n_null=n_null, n_boot=n_boot, metric=metric, iteration=0, run_ts=run_ts)
    baseline_locked = base["locked_auc"]
    records = [base]
    it = 1
    for name, fn in registry.items():
        if name == baseline_name:
            continue
        records.append(E.certify(name, bench, fn, code_path=code, seed=seed, n_null=n_null,
                                 n_boot=n_boot, baseline_locked_auc=baseline_locked,
                                 metric=metric, iteration=it, run_ts=run_ts))
        it += 1
    canary = E.certify("random_canary", bench, random_canary(seed), code_path=code, seed=seed,
                       n_null=n_null, n_boot=n_boot, baseline_locked_auc=baseline_locked,
                       metric=metric, log=False, iteration=it, run_ts=run_ts)
    return base, records, canary


def print_report(bench, base, records, canary):
    ranked = sorted(records, key=lambda r: (r["passed"], r["dev_auc"]), reverse=True)
    print(f"\n{'='*84}\nBenchmark: {bench.id}   baseline={base['name']} (locked AUC {base['locked_auc']:.3f})")
    hdr = f"{'model':26s} {'devAUC':>7s} {'dev95CI':>13s} {'lockAUC':>8s} {'perm_p':>7s} {'nullμ':>6s} {'pass':>6s}"
    print(hdr + "\n" + "-" * len(hdr))
    for r in ranked + [canary]:
        ci = f"[{r['dev_ci'][0]:.2f},{r['dev_ci'][1]:.2f}]"
        tag = "CANARY" if r["name"] == "random_canary" else ("PASS" if r["passed"] else "no")
        print(f"{r['name']:26s} {r['dev_auc']:7.3f} {ci:>13s} {r['locked_auc']:8.3f} "
              f"{r['permutation']['pval']:7.3f} {r['permutation']['null_mean']:6.2f} {tag:>6s}")


def main():
    which = sys.argv[1:] or list(CONF)
    run_ts = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    benches = {}
    for key in which:
        cls, kw = CONF[key]
        bench = cls()
        base, records, canary = run_benchmark(bench, run_ts=run_ts, **kw)
        print_report(bench, base, records, canary)
        benches[bench.id] = dict(
            run_ts=run_ts,
            baseline=base["name"], baseline_locked_auc=base["locked_auc"],
            candidates=[{k: r[k] for k in ("name", "iteration", "dev_auc", "dev_ci",
                        "locked_auc", "locked_ci", "permutation", "gates", "passed")}
                        for r in records],
            canary={k: canary[k] for k in ("dev_auc", "locked_auc", "permutation", "passed")})
    # Merge, don't clobber: a subset run (e.g. `run_loop.py dictrank_random`) must leave the
    # other benchmarks' entries untouched, since loop_summary.json is the cited provenance
    # source for grant-facing numbers. Each benchmark entry carries its own run_ts; the
    # top-level run_ts is the timestamp of the most recent invocation (i.e. this one), so it
    # answers "when was this file last written" and never implies every entry is that fresh.
    path = os.path.join(HERE, "loop_summary.json")
    merged = {}
    if os.path.exists(path):
        merged = json.load(open(path)).get("benchmarks", {})
    merged.update(benches)
    json.dump(dict(run_ts=run_ts, benchmarks=merged), open(path, "w"), indent=2)
    print("\nGate: perm_p<0.05 AND devAUC>null_p95 AND lockAUC>baseline.  nullμ~0.50 = no leak.")
    print("wrote loop_summary.json + appended to leaderboard.jsonl")


if __name__ == "__main__":
    main()
