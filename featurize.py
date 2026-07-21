#!/usr/bin/env python3
"""
SMILES featurization for the scale-up benchmarks (RDKit).

Two products per molecule:
  - Morgan fingerprint (ECFP-like), the feature vector the fingerprint models learn on.
  - Bemis-Murcko scaffold, the key for scaffold-disjoint train/test splits. Splitting by
    scaffold (not by molecule) is the standard leakage guard in cheminformatics: a near-clone
    of a training molecule must not sit in the test set, or the reported generalization is a
    memorization mirage.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import rdFingerprintGenerator
from rdkit.DataStructs import ConvertToNumpyArray

RDLogger.DisableLog("rdApp.*")   # silence per-molecule parse noise; we handle failures below
_GEN = {}


def _gen(radius, nbits):
    key = (radius, nbits)
    if key not in _GEN:
        _GEN[key] = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    return _GEN[key]


def morgan(smiles, radius=2, nbits=1024):
    """Bit vector as an int8 numpy array; None if the SMILES will not parse."""
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        return None
    arr = np.zeros(nbits, dtype=np.int8)
    ConvertToNumpyArray(_gen(radius, nbits).GetFingerprint(m), arr)
    return arr


def scaffold(smiles):
    """Canonical Bemis-Murcko scaffold SMILES; '' if it will not parse (grouped alone)."""
    m = Chem.MolFromSmiles(smiles) if smiles else None
    if m is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return ""


def scaffold_split(scaffolds, frac_locked=0.25, seed=0):
    """Assign each row to 'dev' or 'locked' so a scaffold never spans both. Greedy: shuffle
    scaffold groups, fill locked until it reaches frac_locked of rows, rest to dev.
    Molecules with an empty scaffold each form their own singleton group."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i, s in enumerate(scaffolds):
        key = s if s else f"__empty_{i}"      # empty scaffolds never merge
        groups.setdefault(key, []).append(i)
    keys = list(groups)
    rng.shuffle(keys)
    n = len(scaffolds)
    target = int(round(frac_locked * n))
    locked, filled = set(), 0
    for k in keys:
        if filled < target:
            locked.add(k)
            filled += len(groups[k])
    part = np.array(["locked" if k in locked else "dev"
                     for i in range(n) for k in [scaffolds[i] if scaffolds[i] else f"__empty_{i}"]])
    return part


def random_split(n, frac_locked=0.25, seed=0):
    """Molecule-level random dev/locked split: near-clones may span both sides. This is the
    split scheme most published DICT models report, so running the same features and folds
    under it measures what our scaffold split costs instead of assuming it."""
    rng = np.random.default_rng(seed)
    part = np.full(n, "dev", dtype="<U6")
    part[rng.permutation(n)[:int(round(frac_locked * n))]] = "locked"
    return part


def group_kfold_ids(groups, k=5, seed=0):
    """Map many groups onto k CV folds, keeping every group whole in one fold (GroupKFold).
    Returns a per-row fold id in [0, k). Used so leave-one-group-out over hundreds of
    scaffolds collapses to k fits per pipeline run, which keeps the permutation null tractable
    without breaking the scaffold-disjoint guarantee."""
    rng = np.random.default_rng(seed)
    # sort before shuffling: set/dict iteration order over string keys depends on the
    # per-process string hash seed, which made fold assignment (and every dev-side number
    # built on it) vary between interpreters despite the fixed seed.
    uniq = sorted({g for g in groups})
    rng.shuffle(uniq)
    fold_of = {g: i % k for i, g in enumerate(uniq)}
    return np.array([fold_of[g] for g in groups])
