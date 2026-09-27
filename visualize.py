#!/usr/bin/env python3
"""
Visualize the loop: a scorecard (where each candidate lands vs its baseline, and the
running-best over the loop's iterations) and the permutation-null figure (the anti-spurious
money shot: the certified model's real AUC sits far right of the shuffled-label null).

Reads leaderboard.jsonl (append-only, grows every run) + loop_summary.json.
Run: python visualize.py
Writes loop_scorecard.{png,svg} and permutation_null.{png,svg}.
"""
import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_core as E     # noqa: E402
import benchmarks as B    # noqa: E402
import models as M        # noqa: E402

# colorblind-safe: identity by hue + shape, never color alone
C_PASS = "#0f766e"     # teal: certified
C_FAIL = "#9ca3af"     # gray: not certified
C_BASE = "#b45309"     # amber: baseline reference
C_NULL = "#cbd5e1"     # light gray: shuffled-label null mass
C_OBS = "#0f766e"      # teal: observed
INK = "#1f2937"
MUTED = "#6b7280"
BENCH_TITLE = {"cipa28_tdp": "CiPA-28 (TdP class)",
               "enginev0_cardiotox": "engine-v0 (iPSC-CM cardiotox)",
               "variant_tdp": "variant × drug (variant-conditioning test)",
               "dictrank_fp": "DICTrank (structure → cardiotox)"}
BENCH_ORDER = ["cipa28_tdp", "enginev0_cardiotox", "variant_tdp", "dictrank_fp"]
BENCH_CLASS = {"cipa28_tdp": B.CiPABenchmark, "enginev0_cardiotox": B.EngineV0Benchmark,
               "variant_tdp": B.VariantBenchmark, "dictrank_fp": B.DICTrankBenchmark}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#d1d5db")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="x", color="#f0f0f0", lw=0.8)
    ax.set_axisbelow(True)


def load():
    rows = [json.loads(l) for l in open(os.path.join(HERE, "leaderboard.jsonl"))]
    summ = json.load(open(os.path.join(HERE, "loop_summary.json")))
    return rows, summ


def scorecard(summ):
    benches = summ["benchmarks"]
    order = [b for b in BENCH_ORDER if b in benches]
    n = len(order)
    fig = plt.figure(figsize=(3.6 * n, 7.2))
    gs = gridspec.GridSpec(2, n, height_ratios=[1.35, 1], hspace=0.5, wspace=0.75,
                           left=0.06, right=0.98, top=0.83, bottom=0.09)

    # top row: every candidate's dev AUC (hollow); locked AUC (filled) only where the locked
    # split was scored, i.e. the baseline and the dev-selected champion
    for j, bid in enumerate(order):
        b = benches[bid]
        ax = fig.add_subplot(gs[0, j])
        cands = sorted(b["candidates"], key=lambda c: c["dev_auc"])
        ys = np.arange(len(cands))
        for i, c in enumerate(cands):
            passed = c["passed"] and c["name"] != b["baseline"]
            col = C_PASS if passed else (C_BASE if c["name"] == b["baseline"] else C_FAIL)
            ax.scatter([c["dev_auc"]], [i], s=40, facecolor="none", edgecolor=col,
                       linewidth=1.2, zorder=2)
            if c.get("locked_auc") is not None:
                lo, hi = c["locked_ci"]
                ax.plot([lo, hi], [i, i], color=col, lw=2, alpha=0.35, solid_capstyle="round")
                ax.scatter([c["locked_auc"]], [i], s=48, color=col, zorder=3,
                           marker=("D" if c["name"] == b["baseline"] else "o"),
                           edgecolor="white", linewidth=0.8)
        ax.axvline(b["baseline_locked_auc"], color=C_BASE, ls="--", lw=1.2, alpha=0.8)
        ax.axvline(0.5, color="#e5e7eb", lw=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([c["name"].replace("_", " ") for c in cands], fontsize=7.2)
        ax.set_xlim(0.35, 1.02)
        ax.set_title(BENCH_TITLE[bid], fontsize=9.5, color=INK, pad=8)
        _style(ax)
        if j == 0:
            ax.set_xlabel("AUC (hollow = dev, filled = locked)", fontsize=8, color=MUTED)

    # bottom: running-best certified locked AUC over the loop's iterations, per benchmark
    axp = fig.add_subplot(gs[1, :])
    for bid in order:
        b = benches[bid]
        cands = sorted(b["candidates"], key=lambda c: c["iteration"])
        xs, best = [], []
        cur = b["baseline_locked_auc"]
        for c in cands:
            if c["passed"] and c["name"] != b["baseline"] and c.get("locked_auc") is not None:
                cur = max(cur, c["locked_auc"])
            xs.append(c["iteration"])
            best.append(cur)
        axp.plot(xs, best, marker="o", ms=5, lw=2, label=BENCH_TITLE[bid])
        axp.annotate(f"{best[-1]:.2f}", (xs[-1], best[-1]), textcoords="offset points",
                     xytext=(6, 0), fontsize=8, color=INK, va="center")
    axp.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    axp.set_xlabel("loop iteration (candidate tried, in order)", fontsize=8, color=MUTED)
    axp.set_ylabel("best certified\nlocked AUC", fontsize=8, color=MUTED)
    axp.set_title("Loop progress: best certified model so far", fontsize=9.5, color=INK, pad=6)
    axp.legend(frameon=False, fontsize=8, loc="lower right")
    _style(axp)

    fig.suptitle("Cardiac-safety loop scorecard  ·  " + summ.get("run_ts", ""),
                 fontsize=12, color=INK, x=0.07, ha="left", y=0.965, weight="bold")
    fig.text(0.07, 0.9, "diamond = baseline (amber dashed) · teal = certified (beats baseline "
             "+ permutation null) · gray = not certified · hollow = dev AUC · "
             "filled = locked AUC, scored for baseline and champion only · bars = 95% CI",
             fontsize=8, color=MUTED, ha="left")
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(HERE, f"loop_scorecard.{ext}"), dpi=150,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


def null_figure(summ):
    """Recompute the null distribution for each benchmark's headline certified model and show
    the observed AUC sitting to its right. This is the check that the score is signal."""
    benches = summ["benchmarks"]
    order = [b for b in BENCH_ORDER if b in benches]
    n = len(order)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.6))
    n_null = {"cipa28_tdp": 1000, "enginev0_cardiotox": 1000, "variant_tdp": 1000,
              "dictrank_fp": 200}
    for ax, bid in zip(axes, order):
        b = benches[bid]
        # headline = the champion selected on dev, else the baseline. Never pick by locked AUC.
        head_name = b.get("champion") or b["baseline"]
        head = next(c for c in b["candidates"] if c["name"] == head_name)
        bench = BENCH_CLASS[bid]()
        registry, _ = M.REGISTRIES[bid]
        Xd, yd, gd, _ = bench.dev()
        perm = E.permutation_null(registry[head["name"]], Xd, yd, gd,
                                  metric=E.auc_for(bench.pos_mask), n=n_null[bid], keep_dist=True)
        ax.hist(perm["null_dist"], bins=24, color=C_NULL, edgecolor="white", linewidth=0.4)
        ax.axvline(perm["observed"], color=C_OBS, lw=2.4)
        ax.axvline(0.5, color="#e5e7eb", lw=1)
        ymax = ax.get_ylim()[1]
        ax.annotate(f"observed\n{perm['observed']:.2f}", (perm["observed"], ymax * 0.92),
                    color=C_OBS, fontsize=8, ha="center", va="top", weight="bold")
        ax.text(0.02, 0.96, f"p = {perm['pval']:.3f}", transform=ax.transAxes, fontsize=8.5,
                color=INK, va="top")
        ax.set_title(f"{BENCH_TITLE[bid]}\n{head['name'].replace('_',' ')}",
                     fontsize=8.8, color=INK)
        ax.set_xlabel("AUC under shuffled labels (null)", fontsize=8, color=MUTED)
        ax.set_xlim(0.2, 1.0)
        _style(ax)
        ax.grid(axis="y", color="#f0f0f0", lw=0.8)
    fig.suptitle("Permutation null: the certified score vs shuffled-label chance",
                 fontsize=11.5, color=INK, x=0.07, ha="left", weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    for ext in ("png", "svg"):
        fig.savefig(os.path.join(HERE, f"permutation_null.{ext}"), dpi=150,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main():
    _, summ = load()
    scorecard(summ)
    null_figure(summ)
    print("wrote loop_scorecard.{png,svg} and permutation_null.{png,svg}")


if __name__ == "__main__":
    main()
