#!/usr/bin/env python3
"""
Fully-clean leave-one-dataset-out (membership-leakage removed) for the
"81-gene cardiac-identity panel replicates across 4 datasets" claim.

Background
----------
Firewall B (panel_replication_run.py) built the held-out consensus direction from
the other three datasets, but panel MEMBERSHIP still used the held-out dataset (the
gene was in the panel because the 4-dataset Stouffer + >=3/4 rule put it there). So a
residual leak remained: the SET of genes being tested was chosen with knowledge of the
held-out data.

The fully-clean test removes that leak: for a held-out dataset, RE-SELECT the panel from
scratch over the FULL transcriptome background using only the OTHER THREE datasets (same
Stouffer-Z + Bonferroni p<5e-7 + directional-consistency rule), then test whether the
held-out dataset agrees directionally on that freshly-selected, held-out-blind panel.

Data (all four full backgrounds now available, 2026-07-09)
----------------------------------------------------------
All four raw inputs are on disk as FULL expression / DEG matrices, and their
full-background z-scores reconstruct the original richer_panel_genes.csv exactly:
  - Sharma 2017  : data/sharma2017/GSE114686_ProcessedData.csv   (13,226 genes)
  - Burridge 2016: data/burridge2016_log_fpkm.csv                (21,616 genes)
  - Wu/Liu 2024  : data/GSE213311_doxo_counts.csv                (35,236 genes)
  - DToxS        : LINCS DToxS SVD Degs_initial_iPSCdCMs_P0 full per-drug/cell-line
                   DEG TSVs (all genes), re-downloaded 2026-07-09 from
                   iyengarlab.org/dtoxs/files/LINCS_DToxS_SVD.zip. The earlier on-disk
                   file (data/dtoxs/dtoxs_degs_per_celline.txt) was truncated to the
                   top-600 DEGs per drug x cell-line, which could not give DToxS's
                   full-background mean/SD; the full TSVs restore it.

With all four full backgrounds, every leave-one-out fold re-selects its panel over the
full transcriptome from the OTHER THREE datasets and tests the held-out dataset's
direction on that held-out-blind panel.

numpy + scipy + csv only. Deterministic given SEED.
Run:  ENGINE_V0_DIR=<data checkout> python panel_replication_clean_run.py
"""
import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import norm

SEED = 0
N_PERM = 10000
BONF = 5e-7          # same fixed Bonferroni cutoff the original panel selection used
EV0 = os.environ.get("ENGINE_V0_DIR", "../engine-v0")  # set ENGINE_V0_DIR to your data checkout
PANEL_CSV = f"{EV0}/data/richer_panel_genes.csv"

BURRIDGE_CSV = f"{EV0}/data/burridge2016_log_fpkm.csv"
WULIU_CSV = f"{EV0}/data/GSE213311_doxo_counts.csv"
SHARMA_CSV = f"{EV0}/data/sharma2017/GSE114686_ProcessedData.csv"
DTOXS_META = f"{EV0}/data/dtoxs/dtoxs_drug_metadata.txt"

# Full DToxS DEG TSV directory (all genes per drug x cell-line). Auto-detected.
DTOXS_DEG_DIR_CANDIDATES = [p for p in [
    os.environ.get("DTOXS_DEG_DIR", ""),   # set to .../LINCS_DToxS_SVD/Experimental_data/Degs_initial_iPSCdCMs_P0
    "/tmp/dtoxs_full_unzipped/LINCS_DToxS_SVD/Experimental_data/Degs_initial_iPSCdCMs_P0",
] if p]

BURRIDGE_DONORS = ["CON1", "CON3", "CON4", "CH1", "CH3", "CH4"]
WULIU_TREAT = ["wtD_1", "wtD_2"]
WULIU_CTRL = ["wt_1", "wt_2"]

SHARMA_DRUG = {"D": "DMSO", "E": "Erlotinib", "L": "Lapatinib", "S": "Sorafenib", "U": "Sunitinib"}
SHARMA_TIME = {"1": 6, "2": 24, "3": 72, "4": 168}
SHARMA_DOSE = {"A": 0.001, "B": 1.0, "C": 3.0, "D": 10.0}
SHARMA_CARDIO = {"Sunitinib", "Sorafenib", "Lapatinib"}

DTOXS_FILE_RE = re.compile(
    r"Human-(MSN\d{2})-\d+R-CM-Hour-48-Plate-0-Calc\.CTRL-([A-Z]+)\.tsv$"
)


# ----------------------------------------------------------------------------------
# Full-background loaders (pure numpy/csv; replicate the original Stouffer panel selection)
# ----------------------------------------------------------------------------------
def load_burridge():
    with open(BURRIDGE_CSV) as f:
        rdr = csv.reader(f)
        cols = [c.strip().strip('"') for c in next(rdr)]
        idx = {c: i for i, c in enumerate(cols)}
        out, seen = {}, set()
        for row in rdr:
            g = row[0].strip().strip('"').upper()
            if g in seen:
                continue
            seen.add(g)
            deltas = [float(row[idx[f"{d}_1uM"]]) - float(row[idx[f"{d}_0uM"]]) for d in BURRIDGE_DONORS]
            out[g] = float(np.mean(deltas))
    return out


def load_wuliu():
    with open(WULIU_CSV) as f:
        rdr = csv.reader(f)
        cols = [c.strip().strip('"').lstrip("﻿") for c in next(rdr)]
        samples = cols[1:]
        sidx = {s: i for i, s in enumerate(samples)}
        genes, mat, seen = [], [], set()
        for row in rdr:
            g = row[0].strip().strip('"').upper()
            if g in seen:
                continue
            seen.add(g)
            genes.append(g)
            mat.append([float(row[i + 1]) for i in range(len(samples))])
    M = np.array(mat)
    cpm = M / M.sum(axis=0) * 1e6
    logcpm = np.log2(cpm + 1.0)
    t = logcpm[:, [sidx[c] for c in WULIU_TREAT]].mean(axis=1)
    c = logcpm[:, [sidx[c] for c in WULIU_CTRL]].mean(axis=1)
    return dict(zip(genes, t - c))


def _parse_sharma(name):
    if len(name) != 4:
        return None
    d, t, ds = SHARMA_DRUG.get(name[0]), SHARMA_TIME.get(name[1]), SHARMA_DOSE.get(name[2])
    if d is None or t is None or ds is None or not name[3].isdigit():
        return None
    return (name, d, t, ds)


def load_sharma():
    with open(SHARMA_CSV) as f:
        rdr = csv.reader(f)
        hdr = next(rdr)
        samples = hdr[2:]
        accum = defaultdict(list)
        for row in rdr:
            gname = row[1]
            if gname in ("", "NA"):
                continue
            accum[gname].append(np.log2(np.array([float(x) for x in row[2:]]) + 1.0))
    genes = list(accum.keys())
    expr = {g: np.mean(accum[g], axis=0) for g in genes}
    sidx = {s: i for i, s in enumerate(samples)}
    meta = [p for p in (_parse_sharma(s) for s in samples) if p is not None]
    dmso_by_time = defaultdict(list)
    treat_by_cond = defaultdict(list)
    conds = set()
    for nm, d, t, ds in meta:
        if d == "DMSO":
            dmso_by_time[t].append(nm)
        elif d in SHARMA_CARDIO:
            treat_by_cond[(d, t, ds)].append(nm)
            conds.add((d, t, ds))
    lfc = {g: [] for g in genes}
    for d, t, ds in sorted(conds):
        tsamps, csamps = treat_by_cond[(d, t, ds)], dmso_by_time.get(t, [])
        if not tsamps or not csamps:
            continue
        ti = [sidx[s] for s in tsamps]
        ci = [sidx[s] for s in csamps]
        for g in genes:
            e = expr[g]
            lfc[g].append(e[ti].mean() - e[ci].mean())
    return {g: float(np.mean(v)) for g, v in lfc.items() if v}


def _find_dtoxs_dir():
    for p in DTOXS_DEG_DIR_CANDIDATES:
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "DToxS full DEG dir not found. Tried: " + "; ".join(DTOXS_DEG_DIR_CANDIDATES)
    )


def _load_dtoxs_meta():
    """abbrev -> lower drug name ; lower drug name -> Is_cardiotoxic."""
    abbrev_to_drug, drug_to_label = {}, {}
    with open(DTOXS_META, encoding="latin-1") as f:
        rdr = csv.DictReader(f, delimiter="\t")
        for r in rdr:
            ab = r["Drug abbreviation"].strip()
            nm = r["Drug name"].strip().lower()
            abbrev_to_drug[ab] = nm
            drug_to_label[nm] = r["Is_cardiotoxic"].strip()
    return abbrev_to_drug, drug_to_label


def _read_deg_logfc(path):
    """Read one DToxS edgeR DEG TSV. These are R write.table(row.names=TRUE)
    files: the header has one FEWER field than each data row, so the gene symbol
    is data column 0 (the unnamed row-name / index) and the named 'logFC' header
    at position h maps to data column h+1. This is exactly what pandas
    read_csv(index_col=0, usecols=['logFC']) does in the original panel-selection script. Returns
    {GENE_UPPER: logFC}, first occurrence of a duplicate symbol kept
    (matches the original script's drop_duplicates keep='first')."""
    with open(path, encoding="latin-1") as f:
        rdr = csv.reader(f, delimiter="\t")
        hdr = next(rdr)
        fc_h = hdr.index("logFC")   # position among the NAMED columns
        fc_i = fc_h + 1             # +1 for the unnamed gene/index column in data rows
        out, seen = {}, set()
        for row in rdr:
            if not row:
                continue
            g = row[0].strip().strip('"').upper()
            if g in seen:
                continue
            seen.add(g)
            out[g] = float(row[fc_i])
    return out


def load_dtoxs_full():
    """Per-gene mean log2FC for cardiotox+ drugs from the FULL DToxS DEG TSVs.
    Replicates the original panel-selection script's DToxS loader exactly:
      per (drug, MSN): first replicate file wins (sorted order);
      per drug: mean across MSN cell lines (skip genes absent in an MSN);
      per gene: mean across cardiotox+ drugs (skip drugs absent for a gene)."""
    deg_dir = _find_dtoxs_dir()
    abbrev_to_drug, drug_to_label = _load_dtoxs_meta()

    files = sorted(glob.glob(os.path.join(deg_dir, "Human-MSN*-CM-Hour-48-Plate-0-Calc.CTRL-*.tsv")))
    per_drug_msn = defaultdict(dict)   # drug -> {MSN -> {gene: logFC}}
    for f in files:
        m = DTOXS_FILE_RE.search(os.path.basename(f))
        if not m:
            continue
        msn, abbrev = m.group(1), m.group(2)
        drug = abbrev_to_drug.get(abbrev)
        if drug is None or drug_to_label.get(drug) != "Yes":
            continue
        if msn in per_drug_msn[drug]:
            continue  # keep first replicate
        per_drug_msn[drug][msn] = _read_deg_logfc(f)

    # per drug: mean across MSN cell lines (skipna)
    drug_means = {}
    for drug, msn_dict in per_drug_msn.items():
        gene_vals = defaultdict(list)
        for msn, gdict in msn_dict.items():
            for g, v in gdict.items():
                gene_vals[g].append(v)
        drug_means[drug] = {g: float(np.mean(vs)) for g, vs in gene_vals.items()}

    # per gene: mean across cardiotox+ drugs (skipna)
    gene_vals = defaultdict(list)
    for drug, gdict in drug_means.items():
        for g, v in gdict.items():
            gene_vals[g].append(v)
    dtoxs_logfc = {g: float(np.mean(vs)) for g, vs in gene_vals.items()}
    return dtoxs_logfc, sorted(per_drug_msn.keys())


def zscore(d):
    genes = list(d.keys())
    v = np.array([d[g] for g in genes])
    z = (v - v.mean()) / v.std(ddof=1)   # ddof=1 matches pandas .std() in the original script
    return dict(zip(genes, z))


# ----------------------------------------------------------------------------------
# Validation: reconstructed z must match richer_panel_genes.csv exactly
# ----------------------------------------------------------------------------------
def validate(zmap, rows, tol=1e-6):
    checks = {"DToxS": "z_DToxS", "Sharma": "z_Sharma",
              "Burridge": "z_Burridge", "WuLiu": "z_WuLiu"}
    report = {}
    for ds, col in checks.items():
        if ds not in zmap:
            continue
        diffs = []
        for r in rows:
            v = r[col]
            if v in ("", "nan", "NaN"):
                continue
            g = r["gene"]
            if g in zmap[ds]:
                diffs.append(abs(zmap[ds][g] - float(v)))
        mx = max(diffs) if diffs else 0.0
        report[ds] = (len(diffs), mx)
        assert mx < tol, f"{ds} reconstruction mismatch: max|diff|={mx:g} over {len(diffs)} genes"
    return report


# ----------------------------------------------------------------------------------
# Clean LODO: hold out one dataset, re-select from the other three full backgrounds
# ----------------------------------------------------------------------------------
def clean_lodo(held_out, zmap, rng):
    selectors = [d for d in ["DToxS", "Sharma", "Burridge", "WuLiu"] if d != held_out]
    union = sorted(set().union(*[set(zmap[s]) for s in selectors]))  # sorted => order-deterministic across processes (no PYTHONHASHSEED dependence)

    stouffer, direction, n_tested = {}, {}, 0
    for g in union:
        zs = [zmap[s][g] for s in selectors if g in zmap[s]]
        if len(zs) < 3:                       # require present in all 3 selectors
            continue
        n_tested += 1
        zsum = float(np.sum(zs))
        stouffer[g] = zsum / np.sqrt(len(zs))
        direction[g] = np.sign(zsum)

    # panel = Bonferroni p<5e-7 AND unanimous across the 3 selectors
    panel = []
    for g, st in stouffer.items():
        p_two = 2 * (1 - norm.cdf(abs(st)))
        signs = np.sign([zmap[s][g] for s in selectors])
        unanimous = len(set(signs)) == 1 and 0 not in signs
        if p_two < BONF and unanimous:
            panel.append(g)

    # held-out direction test on covered panel genes
    held_z = zmap[held_out]
    tested = [g for g in panel if g in held_z]
    covered = len(tested)
    cons = np.array([direction[g] for g in tested])
    held = np.array([np.sign(held_z[g]) for g in tested])
    obs = float(np.mean(held == cons))

    null = np.empty(N_PERM)
    for k in range(N_PERM):
        null[k] = np.mean(rng.permutation(held) == cons)
    pval = (np.sum(null >= obs) + 1) / (N_PERM + 1)

    return dict(held_out=held_out, selectors=selectors, n_tested_genes=n_tested,
                panel_size=len(panel), covered=covered, uncovered=len(panel) - covered,
                observed=obs, null_mean=float(null.mean()),
                null_p95=float(np.percentile(null, 95)), null_max=float(null.max()),
                pval=float(pval))


def main():
    rng = np.random.default_rng(SEED)
    rows = list(csv.DictReader(open(PANEL_CSV)))

    print("Loading full-background datasets (DToxS, Sharma, Burridge, Wu/Liu)...")
    dtoxs_logfc, dtoxs_drugs = load_dtoxs_full()
    zmap = {"DToxS": zscore(dtoxs_logfc),
            "Sharma": zscore(load_sharma()),
            "Burridge": zscore(load_burridge()),
            "WuLiu": zscore(load_wuliu())}
    print(f"  DToxS: {len(zmap['DToxS'])} genes (full background), "
          f"{len(dtoxs_drugs)} cardiotox+ drugs")
    for ds in ["Sharma", "Burridge", "WuLiu"]:
        print(f"  {ds}: {len(zmap[ds])} genes (full background)")

    rep = validate(zmap, rows)
    print("\nReconstruction check vs richer_panel_genes.csv (must match to <1e-6):")
    for ds, (n, mx) in rep.items():
        print(f"  {ds}: {n} panel z-scores checked, max|diff|={mx:.2e}")

    print("\n=== FULLY-CLEAN LODO: all 4 folds (panel re-selected from the other 3 full backgrounds) ===")
    results = {}
    for held in ["DToxS", "Sharma", "Burridge", "WuLiu"]:
        res = clean_lodo(held, zmap, rng)
        results[held] = res
        print(f"\n  Hold out {held}  (selectors: {', '.join(res['selectors'])})")
        print(f"    genes tested in 3-dataset meta (present in all 3): {res['n_tested_genes']}")
        print(f"    fresh Bonferroni panel size (p<{BONF:g}, unanimous 3/3): {res['panel_size']}")
        print(f"    panel genes covered by held-out (testable): {res['covered']} "
              f"(uncovered: {res['uncovered']})")
        print(f"    observed {held} directional agreement: {res['observed']:.3f} "
              f"({int(round(res['observed']*res['covered']))}/{res['covered']})")
        print(f"    null mean / p95 / max: {res['null_mean']:.3f} / {res['null_p95']:.3f} / {res['null_max']:.3f}")
        print(f"    empirical p = {res['pval']:.2e}  ({N_PERM} permutations, seed {SEED})")

    print("\n=== SUMMARY (fully-clean LODO, all 4 folds) ===")
    print(f"  {'held out':10s} {'panel':>6s} {'cov':>5s} {'obs':>7s} {'null_mean':>10s} {'null_max':>9s} {'p':>9s}")
    for held in ["DToxS", "Sharma", "Burridge", "WuLiu"]:
        r = results[held]
        print(f"  {held:10s} {r['panel_size']:6d} {r['covered']:5d} {r['observed']:7.3f} "
              f"{r['null_mean']:10.3f} {r['null_max']:9.3f} {r['pval']:9.2e}")


if __name__ == "__main__":
    main()
