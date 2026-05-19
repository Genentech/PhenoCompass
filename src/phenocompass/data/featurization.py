"""Molecular featurization utilities.

This module provides functions for converting molecules to graph representations
compatible with the pretrained PhenoCompass models. The atom feature encoding
produces 133-dimensional vectors matching the model checkpoint expectations.
"""

from typing import List, Union

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdmolfiles, rdmolops
from torch_geometric.data import Data

# Feature dimension constants - must match pretrained model expectations
MAX_ATOMIC_NUM = 100

ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Offset calculations for sparse encoding (each category includes +1 for unknown)
DEGREES_OFFSET = MAX_ATOMIC_NUM + 1  # 101
NUM_DEGREES = 6
CHARGE_OFFSET = DEGREES_OFFSET + NUM_DEGREES + 1  # 108
NUM_CHARGE_OPTIONS = 5
CHIRAL_OFFSET = CHARGE_OFFSET + NUM_CHARGE_OPTIONS + 1  # 114
NUM_CHIRAL_TAGS = 4
HS_OFFSET = CHIRAL_OFFSET + NUM_CHIRAL_TAGS + 1  # 119
NUM_HS_LIMIT = 5
HYBRIDIZATION_OFFSET = HS_OFFSET + NUM_HS_LIMIT + 1  # 125
NUM_HYBRIDIZATION = len(ATOM_FEATURES['hybridization'])  # 5
OTHER_OFFSET = HYBRIDIZATION_OFFSET + NUM_HYBRIDIZATION + 1  # 131
NUM_OTHER = 2  # aromatic + mass
NUM_TOTAL = OTHER_OFFSET + NUM_OTHER  # 133


def atom_features(
    atom: Chem.rdchem.Atom, functional_groups: List[int] = None
) -> List[Union[bool, int, float]]:
    """Compute atom features for a single atom.

    Builds a sparse one-hot encoding of atomic properties. Each categorical
    feature includes an "unknown" category for out-of-range values.

    Parameters
    ----------
    atom : rdkit.Chem.Atom
        RDKit atom object.
    functional_groups : list of int, optional
        A k-hot vector indicating the functional groups the atom belongs to.

    Returns
    -------
    list
        Feature vector of length NUM_TOTAL (133) plus any functional groups.
    """
    encoding = [0] * NUM_TOTAL

    # Atomic number (0-99 + unknown)
    value = atom.GetAtomicNum() - 1
    value = value if (value >= 0 and value < MAX_ATOMIC_NUM) else MAX_ATOMIC_NUM
    encoding[value] = 1

    # Degree (0-5 + unknown)
    value = atom.GetTotalDegree()
    value = value if (value >= 0 and value < NUM_DEGREES) else NUM_DEGREES
    encoding[value + DEGREES_OFFSET] = 1

    # Formal charge (-2 to +2 mapped to 0-4 + unknown)
    value = atom.GetFormalCharge() + 2
    value = value if (value >= 0 and value < NUM_CHARGE_OPTIONS) else NUM_CHARGE_OPTIONS
    encoding[value + CHARGE_OFFSET] = 1

    # Chiral tag (0-3 + unknown)
    value = int(atom.GetChiralTag())
    value = value if (value >= 0 and value < NUM_CHIRAL_TAGS) else NUM_CHIRAL_TAGS
    encoding[value + CHIRAL_OFFSET] = 1

    # Total number of hydrogens (0-4 + unknown)
    value = int(atom.GetTotalNumHs())
    value = value if (value >= 0 and value < NUM_HS_LIMIT) else NUM_HS_LIMIT
    encoding[value + HS_OFFSET] = 1

    # Hybridization (SP, SP2, SP3, SP3D, SP3D2 + unknown)
    value = int(atom.GetHybridization())
    choices = ATOM_FEATURES['hybridization']
    value = choices.index(value) if value in choices else NUM_HYBRIDIZATION
    encoding[value + HYBRIDIZATION_OFFSET] = 1

    # Aromaticity (binary)
    encoding[OTHER_OFFSET] = 1 if atom.GetIsAromatic() else 0

    # Scaled atomic mass
    encoding[OTHER_OFFSET + 1] = atom.GetMass() * 0.01

    if functional_groups is not None:
        encoding += functional_groups

    return encoding


def dgl_one_hot_encoding(x, allowable_set):
    """Create one-hot encoding as a list.

    Replacement for dgllife.utils.one_hot_encoding to avoid the dependency.

    Parameters
    ----------
    x : any
        Value to encode.
    allowable_set : list
        List of allowed values.

    Returns
    -------
    list of bool
        One-hot encoded list where True indicates match.
    """
    return list(map(lambda s: x == s, allowable_set))


def mol_to_data(notation, mode="smile", label=None, mol_features=None):
    """Convert a molecule to a PyTorch Geometric Data object.

    Parameters
    ----------
    notation : str
        Molecule as SMILES or InChI string.
    mode : str, default "smile"
        Molecule notation type ("smile" or "inchi").
    label : torch.Tensor, optional
        Label for the molecule.
    mol_features : torch.Tensor, optional
        Additional molecular features.

    Returns
    -------
    torch_geometric.data.Data
        Graph data object with node features (133-dim), edge indices,
        and edge attributes (12-dim).

    Raises
    ------
    ValueError
        If the molecule cannot be parsed.
    """
    if mode == "smile":
        mol = Chem.MolFromSmiles(notation)
    elif mode == "inchi":
        mol = Chem.inchi.MolFromInchi(notation)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mol is None:
        raise ValueError(f"Failed to parse molecule: {notation}")

    # Canonical atom reordering for reproducibility
    new_order = rdmolfiles.CanonicalRankAtoms(mol)
    mol = rdmolops.RenumberAtoms(mol, new_order)

    # Build edge lists
    src_list = []
    dst_list = []
    num_bonds = mol.GetNumBonds()
    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        src_list.extend([u, v])
        dst_list.extend([v, u])

    # Node features (133-dimensional)
    node_feature = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()])

    # Edge features (12-dimensional)
    edge_feature = []
    type_set = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    stereo_set = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS,
        Chem.rdchem.BondStereo.STEREOTRANS
    ]

    for i in range(num_bonds):
        bond = mol.GetBondWithIdx(i)
        # Bond type (4) + conjugated (1) + in_ring (1) + stereo (6) = 12 features
        feat = (
            dgl_one_hot_encoding(bond.GetBondType(), type_set) +
            [bond.GetIsConjugated(), bond.IsInRing()] +
            dgl_one_hot_encoding(bond.GetStereo(), stereo_set)
        )
        edge_feature.extend([feat, feat.copy()])

    edge_feature = torch.tensor(edge_feature, dtype=torch.float32)

    data = Data(
        x=node_feature,
        edge_attr=edge_feature,
        edge_index=torch.tensor([src_list, dst_list], dtype=torch.int64)
    )

    if label is not None:
        data.y = label
    if mol_features is not None:
        data.mol_features = mol_features

    return data


def mol_to_mf(mol, mf_featurizer=None, mode="smile", output="object"):
    """Convert a molecule to Morgan fingerprint.

    Parameters
    ----------
    mol : str
        Molecule as SMILES or InChI string.
    mf_featurizer : callable, optional
        Fingerprint generator function. If None, creates default Morgan generator.
    mode : str, default "smile"
        Molecule notation type ("smile" or "inchi").
    output : str, default "tensor"
        Output format ("tensor", "vector", or "object").

    Returns
    -------
    torch.Tensor, np.ndarray, or fingerprint object
        Morgan fingerprint in the specified format.

    Raises
    ------
    ValueError
        If the molecule cannot be parsed or output format is unknown.
    """
    if mode == "smile":
        mol_obj = Chem.MolFromSmiles(mol)
    elif mode == "inchi":
        mol_obj = Chem.inchi.MolFromInchi(mol)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if mol_obj is None:
        raise ValueError(f"Failed to parse molecule: {mol}")

    if mf_featurizer is None:
        fpgen = Chem.AllChem.GetMorganGenerator(radius=3, fpSize=2048)
        mf_featurizer = fpgen.GetCountFingerprint

    fp = mf_featurizer(mol_obj)

    if output == "object":
        return fp
    elif output == "vector":
        arr = np.zeros((2048,), dtype=np.float32)
        for idx, count in fp.GetNonzeroElements().items():
            arr[idx] = count
        return arr
    elif output == "tensor":
        arr = np.zeros((2048,), dtype=np.float32)
        for idx, count in fp.GetNonzeroElements().items():
            arr[idx] = count
        return torch.FloatTensor(arr)
    else:
        raise ValueError(f"Unknown output format: {output}")


# Module-level singleton for TautomerEnumerator. Lazily initialized so importing
# this module is cheap; the enumerator is only constructed when a caller needs it.
_TAUTOMER_ENUMERATOR = None


def get_tautomer_enumerator():
    """Return a cached TautomerEnumerator instance.

    The enumerator is constructed once per process and reused for all subsequent
    calls. Constructing one is non-trivial (loads SMARTS transform set), so the
    singleton avoids repeated initialization in tight loops.

    Returns
    -------
    rdkit.Chem.MolStandardize.rdMolStandardize.TautomerEnumerator
        Cached enumerator with default scoring (Sitzmann 2010 rules).
    """
    global _TAUTOMER_ENUMERATOR
    if _TAUTOMER_ENUMERATOR is None:
        from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator
        _TAUTOMER_ENUMERATOR = TautomerEnumerator()
    return _TAUTOMER_ENUMERATOR


def canonicalize_smiles(smiles, tautomer_enumerator=None):
    """Return the tautomer-canonical SMILES for an input SMILES.

    Uses RDKit's TautomerEnumerator with default scoring. If parsing or
    canonicalization fails for any reason, the original SMILES is returned
    unchanged so callers can rely on this never raising.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    tautomer_enumerator : TautomerEnumerator, optional
        Pre-constructed enumerator. If None, uses the module-level singleton
        from get_tautomer_enumerator().

    Returns
    -------
    str
        Canonical SMILES, or the input SMILES on failure.
    """
    if tautomer_enumerator is None:
        tautomer_enumerator = get_tautomer_enumerator()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    try:
        canon_mol = tautomer_enumerator.Canonicalize(mol)
        return Chem.MolToSmiles(canon_mol)
    except Exception:
        return smiles


def load_jump_canonical_map(path):
    """Load the precomputed JUMP tautomer-canonicalization parquet.

    The parquet has two columns: `original_smiles` and `canonical_smiles`,
    produced by `debugging/canonicalize_jump_smiles.py`.

    Parameters
    ----------
    path : str
        Filesystem path to the parquet file.

    Returns
    -------
    dict
        Mapping from original SMILES (as found in the JUMP H5 dataset) to
        tautomer-canonical SMILES.

    Raises
    ------
    FileNotFoundError
        If the parquet file does not exist; suggests the user run the
        canonicalization script.
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"JUMP canonicalization cache not found at {path}. "
            f"Run: python debugging/canonicalize_jump_smiles.py"
        )
    import pandas as pd
    df = pd.read_parquet(path)
    return dict(zip(df["original_smiles"], df["canonical_smiles"]))
