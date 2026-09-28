#!/usr/bin/env python3
"""
Benchmarks for the cardiac-safety loop, with the data firewall built in.

CiPABenchmark
  - The pristine gold standard: 28 canonical CiPA drugs, mechanistic IC50/Hill features,
    ordinal TdP class (Low/Interm/High -> 0/1/2).
  - dev  = the 12 CiPA training drugs (loop may fit + select).
  - locked = the 16 CiPA validation drugs (scored once, at certification).
  - Features per drug = log10(Cmax / IC50) block potency for hERG, ICaL, INa-peak (missing
    channel -> a low sentinel = negligible block). Higher feature = more channel block.
  - Grouped CV uses a curated mechanism-class column so a drug is never tested while a
    mechanistic sibling trains. The classes are textbook (pure-hERG, multichannel,
    Ca-blocker, Na/INaL-blocker); the assignment file is tagged as curated.

DICTrank cross-check (external, independent label source)
  - About 1,000 FDA drugs ranked for drug-induced cardiotoxicity. Used two ways: as a
    trained target (DICTrankBenchmark, Morgan fingerprints from featurize.py), and as an
    independent label check on the drugs that overlap CiPA (dictrank_crosscheck).
"""
import os
import csv
import gzip
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("CSE_DATA_DIR", os.path.join(HERE, "data"))
REF = os.path.join(DATA, "cipa_28drug_reference.csv")
KERNIK = os.path.join(DATA, "cipa_validation_results.csv")
GROUPS_CSV = os.path.join(HERE, "cipa_mechanism_groups.csv")
DICTRANK = os.path.join(DATA, "DICTrank_binarised.csv.gz")

CLS_IDX = {"Low": 0, "Interm": 1, "Intermediate": 1, "High": 2}
SENTINEL = -4.0   # log10(Cmax/IC50) when a channel has no measured block (IC50 ~1e4 x Cmax)


def _num(x):
    try:
        v = float(x)
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def _block_feature(cmax, ic50):
    """log10(Cmax / IC50): higher = more potent channel block at clinical exposure."""
    if ic50 is None or cmax is None:
        return SENTINEL
    return max(SENTINEL, float(np.log10(cmax / ic50)))


def _load_mechanism_groups():
    if not os.path.exists(GROUPS_CSV):
        return {}
    return {r["drug"]: r["mechanism_group"] for r in csv.DictReader(open(GROUPS_CSV))}


def _load_kernik_risk():
    """Precomputed Kernik-2019 iPSC-CM AP risk per drug from cipa_validation_harness.py (cipa-validation repository).
    Reused as a feature so the mechanistic model plugs through the same firewall without a
    Myokit re-run. Missing file -> {} and the kernik feature falls back to sentinel."""
    if not os.path.exists(KERNIK):
        return {}
    return {r["drug"]: float(r["ipsc_risk"]) for r in csv.DictReader(open(KERNIK))}


class CiPABenchmark:
    id = "cipa28_tdp"
    feature_names = ("herg_block", "ical_block", "ina_block", "kernik_risk")

    def __init__(self):
        groups = _load_mechanism_groups()
        kernik = _load_kernik_risk()
        self.rows = []
        for r in csv.DictReader(open(REF)):
            if r["set"] == "extra":
                continue
            cmax = _num(r["cmax_free_nM"])
            if cmax is None:
                continue
            herg = _num(r["hERG_static_IC50_nM"]) or _num(r["hERG_IC50_nM"])
            feats = [_block_feature(cmax, herg),
                     _block_feature(cmax, _num(r["ICaL_IC50_nM"])),
                     _block_feature(cmax, _num(r["INa_peak_IC50_nM"])),
                     kernik.get(r["drug"], SENTINEL)]
            self.rows.append(dict(
                drug=r["drug"], set=r["set"], y=CLS_IDX[r["tdp_class"]],
                X=feats, cmax=cmax, herg_ic50=herg,
                group=groups.get(r["drug"], r["drug"]),   # fall back to per-drug group
            ))
        assert len(self.rows) == 28, f"expected 28 CiPA drugs, got {len(self.rows)}"

    def _view(self, which):
        rs = [r for r in self.rows if r["set"] == which]
        X = np.array([r["X"] for r in rs], float)
        y = np.array([r["y"] for r in rs], int)
        g = np.array([r["group"] for r in rs])
        names = [r["drug"] for r in rs]
        return X, y, g, names

    def dev(self):
        return self._view("training")

    def locked(self):
        return self._view("validation")

    def herg_margin_scores(self, which):
        """The hERG-only baseline risk = -log10(hERG_IC50 / Cmax). Higher = tighter margin."""
        rs = [r for r in self.rows if r["set"] == which]
        return np.array([-np.log10(r["herg_ic50"] / r["cmax"]) if r["herg_ic50"] else -9.0
                         for r in rs], float)

    @staticmethod
    def pos_mask(y):
        return np.asarray(y) == 2   # High-risk vs rest


# --------------------------------------------------------------------------------------
# EngineV0Benchmark: retro-validate the prior cardiotox classifier through the firewall
# --------------------------------------------------------------------------------------
ENGINE = os.path.join(DATA, "enginev0_classifier_results.csv")


class EngineV0Benchmark:
    """The engine-v0 iPSC-CM cardiotox classifier features + label, run through the same
    firewall + permutation null that the original analysis lacked. That analysis found that the
    panel signal is largely a transcriptional-magnitude (log_n_de) artifact that does not
    generalize across mechanism classes. This benchmark lets the null adjudicate: does
    panel_score clear chance independent of magnitude?

    LABEL LEAK: the panel's gene directions were set from the mean response of the DToxS drugs
    labelled cardiotoxic, which include this benchmark's dev and locked positives. Every
    feature except log_n_de therefore carries label information. See README.

    33 drugs (Yes/No only; the 21 ND compounds are dropped). No SMILES coverage here, so CV is
    5-fold over drugs (each drug its own group) and the permutation null is the primary guard.
    Features: panel_score, log_n_de, n_up_panel, n_down_panel."""
    id = "enginev0_cardiotox"
    feature_names = ("panel_score", "log_n_de", "n_up_panel", "n_down_panel")

    def __init__(self, frac_locked=0.33, seed=0):
        self.rows = []
        for r in csv.DictReader(open(ENGINE)):
            if r["cardiotox_class"] not in ("Yes", "No"):
                continue
            if not r["panel_score"]:
                continue
            self.rows.append(dict(
                drug=r["drug"], y=1 if r["cardiotox_class"] == "Yes" else 0,
                X=[float(r["panel_score"]), float(r["log_n_de"]),
                   float(r["n_up_panel"]), float(r["n_down_panel"])]))
        # stratified locked split, seeded
        rng = np.random.default_rng(seed)
        idx = {0: [i for i, r in enumerate(self.rows) if r["y"] == 0],
               1: [i for i, r in enumerate(self.rows) if r["y"] == 1]}
        locked = set()
        for lab, ii in idx.items():
            rng.shuffle(ii)
            locked.update(ii[:int(round(frac_locked * len(ii)))])
        for i, r in enumerate(self.rows):
            r["set"] = "locked" if i in locked else "dev"

    def _view(self, which):
        rs = [r for r in self.rows if r["set"] == which]
        X = np.array([r["X"] for r in rs], float)
        y = np.array([r["y"] for r in rs], int)
        if which == "dev":
            import featurize as F
            g = F.group_kfold_ids(np.arange(len(rs)), k=5, seed=0)   # 5-fold, calibrates the null
        else:
            g = np.arange(len(rs))
        names = [r["drug"] for r in rs]
        return X, y, g, names

    def dev(self):
        return self._view("dev")

    def locked(self):
        return self._view("locked")

    @staticmethod
    def pos_mask(y):
        return np.asarray(y) == 1


# --------------------------------------------------------------------------------------
# VariantBenchmark: the variant-conditioning test. Does variant-conditioning beat a healthy-cell model?
# --------------------------------------------------------------------------------------
VARIANT = os.path.join(DATA, "variant_susceptibility_results.csv")
_VBG = ["healthy", "LQT2_mild", "LQT2_mod", "LQT1"]


def _graded_risk(row, bg):
    """Continuous repolarization risk in one background, from the variant harness output.
    Collapse cases graded by dose-to-collapse (lower multiple = more dangerous); non-collapsing
    drugs keep their APD-prolongation fraction. Same shape as the CiPA harness graded metric."""
    band = row[bg + "_band"]
    if band == "collapse":
        mult = float(row[bg + "_collapse_mult"])
        return 2.0 + (4.0 - mult) / 4.0
    try:
        return float(row[bg])
    except (TypeError, ValueError):
        return 0.0


class VariantBenchmark:
    """The variant-conditioning premise, graded with the same firewall as everything else: does scoring each
    CiPA drug in disease-variant (long-QT) backgrounds add TdP-classification power a healthy-
    cell model does not have? Features = repolarization risk in {healthy, LQT2-mild, LQT2-mod,
    LQT1}; label = CiPA TdP class; baseline = healthy background alone. If variant-conditioning
    does not beat healthy on held-out drugs here, simulation alone cannot support the
    variant-conditioning claim. Reuses the CiPA 12/16 split and
    mechanism groups. Risks are precomputed by variant_susceptibility.py (no Myokit re-run)."""
    id = "variant_tdp"
    feature_names = ("risk_healthy", "risk_LQT2_mild", "risk_LQT2_mod", "risk_LQT1")

    def __init__(self):
        cipa = {r["drug"]: r for r in CiPABenchmark().rows}
        self.rows = []
        for r in csv.DictReader(open(VARIANT)):
            base = cipa.get(r["drug"])
            if base is None:
                continue
            self.rows.append(dict(
                drug=r["drug"], set=base["set"], y=base["y"], group=base["group"],
                X=[_graded_risk(r, bg) for bg in _VBG]))

    def _view(self, which):
        rs = [r for r in self.rows if r["set"] == which]
        X = np.array([r["X"] for r in rs], float)
        y = np.array([r["y"] for r in rs], int)
        g = np.array([r["group"] for r in rs])
        names = [r["drug"] for r in rs]
        return X, y, g, names

    def dev(self):
        return self._view("training")

    def locked(self):
        return self._view("validation")

    @staticmethod
    def pos_mask(y):
        return np.asarray(y) == 2


# --------------------------------------------------------------------------------------
# DICTrankBenchmark: the scale-up. 900+ FDA drugs, Morgan fingerprints, scaffold-split.
# --------------------------------------------------------------------------------------
DICTRANK_FULL = DICTRANK   # DICTrank_binarised.csv.gz (has Standardized_SMILES)


class DICTrankBenchmark:
    """Full-scale cardiotox benchmark: FDA DICTrank labels vs Morgan fingerprints, with a
    scaffold-disjoint dev/locked split and scaffold-grouped k-fold CV. This is the set with
    the statistical power the loop needs; CiPA-28 is too small to iterate on.

    Label: most/less concern -> 1, no -> 0, ambiguous dropped. Features: Morgan(radius=2)."""
    id = "dictrank_fp"

    def __init__(self, nbits=1024, radius=2, frac_locked=0.25, k_folds=5, seed=0,
                 split="scaffold"):
        import featurize as F
        self.nbits = nbits
        if split == "random":
            self.id = "dictrank_fp_random"
        names, smiles, labels = [], [], []
        with gzip.open(DICTRANK_FULL, "rt") as fh:
            for r in csv.DictReader(fh):
                concern = (r.get("DICT _ Concern") or "").strip().lower()
                smi = (r.get("Standardized_SMILES") or "").strip()
                if concern in ("most", "less"):
                    lab = 1
                elif concern == "no":
                    lab = 0
                else:
                    continue
                name = (r.get("Generic/Proper Name(s)") or "").strip()
                names.append(name)
                smiles.append(smi)
                labels.append(lab)
        X, keep_names, keep_y, scaffolds = [], [], [], []
        for nm, smi, lab in zip(names, smiles, labels):
            fp = F.morgan(smi, radius=radius, nbits=nbits)
            if fp is None:
                continue
            X.append(fp)
            keep_names.append(nm)
            keep_y.append(lab)
            scaffolds.append(F.scaffold(smi))
        self.X = np.array(X, float)
        self.y = np.array(keep_y, int)
        self.names = keep_names
        if split == "random":
            # matched n, frac_locked, k and seed; only the grouping changes. Each molecule is
            # its own CV group, so the folds are plain random k-fold like the published models.
            part = F.random_split(len(self.y), frac_locked=frac_locked, seed=seed)
            cv_groups = [f"__mol_{i}" for i in range(len(self.y))]
        else:
            part = F.scaffold_split(scaffolds, frac_locked=frac_locked, seed=seed)
            cv_groups = scaffolds
        self.part = part
        # grouped k-fold ids for the dev rows (locked rows get -1)
        dev_grp = [cv_groups[i] if part[i] == "dev" else None for i in range(len(part))]
        fold = F.group_kfold_ids([s for s in dev_grp if s is not None], k=k_folds, seed=seed)
        self.fold = np.full(len(part), -1)
        self.fold[part == "dev"] = fold

    def _view(self, which):
        mask = self.part == which
        X = self.X[mask]
        y = self.y[mask]
        g = self.fold[mask] if which == "dev" else np.arange(int(mask.sum()))
        names = [n for n, m in zip(self.names, mask) if m]
        return X, y, g, names

    def dev(self):
        return self._view("dev")

    def locked(self):
        return self._view("locked")

    @staticmethod
    def pos_mask(y):
        return np.asarray(y) == 1


# --------------------------------------------------------------------------------------
# DICTrank external cross-check
# --------------------------------------------------------------------------------------
def load_dictrank_labels():
    """Return {normalized_generic_name: 1/0} cardiotox concern. most/less -> 1, no -> 0,
    ambiguous/blank dropped. Curation choice, tagged in the report."""
    if not os.path.exists(DICTRANK):
        return {}
    out = {}
    with gzip.open(DICTRANK, "rt") as fh:
        for r in csv.DictReader(fh):
            concern = (r.get("DICT _ Concern") or "").strip().lower()
            name = (r.get("Generic/Proper Name(s)") or "").strip().lower()
            if not name:
                continue
            key = name.split()[0]           # first token: "dolasetron mesylate" -> "dolasetron"
            if concern in ("most", "less"):
                out[key] = 1
            elif concern in ("no",):
                out[key] = 0
    return out


MIN_MINORITY = 3   # need >=3 in the smaller class or the AUC is not interpretable


def dictrank_crosscheck(drug_names, scores):
    """AUC of `scores` separating DICTrank cardiotox-positive from negative on the name-matched
    subset. Independent of the CiPA TdP labels entirely.

    IMPORTANT: on the CiPA-28 overlap this is degenerate. CiPA drugs are selected for TdP
    relevance, so the matched set is ~12 positive / 1 negative and the AUC is meaningless
    (a single negative). Guarded to report `uninformative` rather than a misleading number.
    The real use of DICTrank is as a trained benchmark with both classes well populated;
    see DICTrankBenchmark."""
    labels = load_dictrank_labels()
    matched = [(s, labels[d.lower()]) for d, s in zip(drug_names, scores)
               if d.lower() in labels]
    ys = [l for _, l in matched]
    minority = min(ys.count(0), ys.count(1)) if ys else 0
    if minority < MIN_MINORITY:
        return dict(n_matched=len(matched), n_pos=sum(ys), auc=None, uninformative=True)
    from eval_core import rank_auc
    s = np.array([m[0] for m in matched], float)
    y = np.array([m[1] for m in matched], int)
    return dict(n_matched=len(matched), n_pos=int(y.sum()),
                auc=float(rank_auc(s, y == 1)), uninformative=False)
