#!/usr/bin/env python3
"""
Run every candidate in models.py through the evaluation core, one benchmark at a time, and
append the results to leaderboard.jsonl.

Selection happens on dev only. Each candidate gets a dev AUC and a permutation null. The
candidate with the highest dev AUC among those that beat the null becomes the champion
(ties go to registry order). Only the champion, the pre-declared baseline and a random
canary are scored on the locked split. The champion passes if its locked AUC beats the
baseline's.

Usage: python run_loop.py [cipa|enginev0|variant|dictrank|dictrank_random ...]   (default: all)
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


def select_champion(devs, order):
    """Highest dev AUC among candidates that beat the permutation null. Ties go to the
    earlier entry in the registry. Returns None if no candidate beats the null."""
    eligible = [n for n in order if devs[n]["beats_null"]]
    if not eligible:
        return None
    return max(eligible, key=lambda n: (devs[n]["dev_auc"], -order.index(n)))


def run_benchmark(bench, n_null, n_boot, seed=0, run_ts=None):
    registry, baseline_name = M.REGISTRIES[bench.id]
    metric = E.auc_for(bench.pos_mask)
    code = os.path.join(HERE, "models.py")
    kw = dict(code_path=code, seed=seed, n_null=n_null, n_boot=n_boot, metric=metric,
              run_ts=run_ts)

    base = E.certify(baseline_name, bench, registry[baseline_name], iteration=0,
                     role="baseline", **kw)
    baseline_locked = base["locked_auc"]

    order = [n for n in registry if n != baseline_name]
    devs = {n: E.evaluate_dev(registry[n], bench, seed=seed, n_null=n_null, n_boot=n_boot,
                              metric=metric) for n in order}
    champion = select_champion(devs, order)

    records = [base]
    for it, name in enumerate(order, start=1):
        is_champ = name == champion
        records.append(E.certify(name, bench, registry[name], dev=devs[name],
                                 score_locked=is_champ, baseline_locked_auc=baseline_locked,
                                 iteration=it, role="champion" if is_champ else "candidate",
                                 **kw))
    canary = E.certify("random_canary", bench, random_canary(seed),
                       baseline_locked_auc=baseline_locked, log=False,
                       iteration=len(order) + 1, role="canary", **kw)
    return base, records, canary, champion


def print_report(bench, base, records, canary, champion):
    ranked = sorted(records, key=lambda r: r["dev_auc"], reverse=True)
    print(f"\n{'='*84}\nBenchmark: {bench.id}   baseline={base['name']} "
          f"(locked AUC {base['locked_auc']:.3f})   champion={champion or 'none'}")
    hdr = (f"{'model':26s} {'devAUC':>7s} {'dev95CI':>13s} {'lockAUC':>8s} {'perm_p':>7s} "
           f"{'nullμ':>6s} {'role':>9s} {'pass':>5s}")
    print(hdr + "\n" + "-" * len(hdr))
    for r in ranked + [canary]:
        ci = f"[{r['dev_ci'][0]:.2f},{r['dev_ci'][1]:.2f}]"
        lock = "      —" if r["locked_auc"] is None else f"{r['locked_auc']:8.3f}"
        tag = "PASS" if r["passed"] else "no"
        print(f"{r['name']:26s} {r['dev_auc']:7.3f} {ci:>13s} {lock:>8s} "
              f"{r['permutation']['pval']:7.3f} {r['permutation']['null_mean']:6.2f} "
              f"{r['role']:>9s} {tag:>5s}")


def main():
    which = sys.argv[1:] or list(CONF)
    run_ts = __import__("datetime").datetime.now().isoformat(timespec="seconds")
    benches = {}
    for key in which:
        cls, kw = CONF[key]
        bench = cls()
        base, records, canary, champion = run_benchmark(bench, run_ts=run_ts, **kw)
        print_report(bench, base, records, canary, champion)
        benches[bench.id] = dict(
            run_ts=run_ts,
            baseline=base["name"], baseline_locked_auc=base["locked_auc"],
            champion=champion,
            candidates=[{k: r[k] for k in ("name", "iteration", "role", "dev_auc", "dev_ci",
                        "locked_auc", "locked_ci", "permutation", "gates", "passed")}
                        for r in records],
            canary={k: canary[k] for k in ("dev_auc", "locked_auc", "permutation", "passed")})
    # Merge rather than overwrite, so a subset run (e.g. `run_loop.py dictrank_random`) leaves
    # the other benchmarks' entries alone. Each benchmark entry carries its own run_ts. The
    # top-level run_ts records when this file was last written, not when every entry ran.
    path = os.path.join(HERE, "loop_summary.json")
    merged = {}
    if os.path.exists(path):
        merged = json.load(open(path)).get("benchmarks", {})
    merged.update(benches)
    json.dump(dict(run_ts=run_ts, benchmarks=merged), open(path, "w"), indent=2)
    print("\nGate: perm_p<0.05 AND devAUC>null_p95 AND lockAUC>baseline. Locked is scored for "
          "the baseline, the canary and the dev-selected champion only.")
    print("wrote loop_summary.json + appended to leaderboard.jsonl")


if __name__ == "__main__":
    main()
