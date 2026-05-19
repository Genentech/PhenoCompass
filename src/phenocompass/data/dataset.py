"""Dataset classes for multimodal learning."""

import pickle
from collections import OrderedDict
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset, Subset

from phenocompass.data.featurization import mol_to_data, mol_to_mf


class H5Dataset(torch.utils.data.Dataset):
    """Base HDF5 dataset class.

    Parameters
    ----------
    data : str
        Path to HDF5 file.
    mods : list
        List of modality names.
    labels : torch.Tensor, optional
        Labels for supervised tasks.
    joint_as_input : bool, default False
        Whether to include joint modality as input.
    index_predetermined_file : str, optional
        Path to pickle file with predetermined split indices.
    """

    def __init__(
        self,
        data,
        mods,
        labels=None,
        joint_as_input=False,
        index_predetermined_file=None,
    ):
        self.f = h5py.File(data, "r")
        if "valid_smiles" in self.f.keys():
            self.mols = self.f["valid_mols"][...].astype(str)
            self.ids = self.f["valid_smiles"][...].astype(str)
        else:
            self.mols = self.f["valid_mols"][...].astype(str)
            self.ids = np.array(
                [Chem.MolToSmiles(Chem.inchi.MolFromInchi(i)) for i in self.mols]
            ).astype(str)
        self.mods = mods
        self.joint_as_input = joint_as_input

        self.y = labels
        if self.y is None:
            self.y = torch.Tensor([-1])

        if index_predetermined_file is not None:
            with open(index_predetermined_file, "rb") as rf:
                self.index_predetermined = pickle.load(rf)
        else:
            self.index_predetermined = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        ipt_mods = OrderedDict()
        for mod in self.mods:
            feat = self._get_mod_feat(idx, mod)
            ipt_mods.update({mod: feat})

        if self.joint_as_input:
            ipt_mods.update(
                {"joint": OrderedDict([(mod, ipt_mods[mod]) for mod in self.mods])}
            )

        return OrderedDict(
            [
                ("inputs", ipt_mods),
                ("labels", self.y if len(self.y) == 1 else self.y[idx]),
            ]
        )

    def _get_mod_feat(self, idx, mod):
        raise NotImplementedError


class FastH5DatasetJUMP(H5Dataset):
    """Optimized HDF5 dataset for JUMP data.

    Parameters
    ----------
    data : str
        Path to HDF5 file.
    mods : list, default ["struct", "morph"]
        Modalities to include.
    struct_ipt : str, default "graph"
        Structure input type ("graph", "mf", or "fixed").
    labels : torch.Tensor, optional
        Labels for supervised tasks.
    joint_as_input : bool, default True
        Whether to include joint modality.
    avg_morph_feat : bool, default True
        Whether to average morphology replicates.
    fixed_struct_data : str, optional
        Path to parquet with fixed structure embeddings.
    index_predetermined_file : str, optional
        Path to predetermined split indices.
    morph_meta_idx : int, default 1
        Index of metadata column for batch encoding.
    ohe_batch : bool, default True
        Whether to one-hot encode batch assignments.
    graph_data : str, optional
        Path to precomputed graph pickle.
    morph_aug : callable, optional
        Morphology augmentation function.
    """

    def __init__(
        self,
        data,
        mods=["struct", "morph"],
        struct_ipt="graph",
        labels=None,
        joint_as_input=True,
        avg_morph_feat=True,
        fixed_struct_data=None,
        index_predetermined_file=None,
        morph_meta_idx=1,
        ohe_batch=True,
        graph_data=None,
        morph_aug=None,
    ):
        super().__init__(
            data,
            mods,
            labels,
            joint_as_input,
            index_predetermined_file,
        )

        self.avg_morph_feat = avg_morph_feat
        self.morph_meta_idx = morph_meta_idx

        self.struct_ipt = struct_ipt
        if self.struct_ipt == "mf":
            fpgen = Chem.AllChem.GetMorganGenerator(radius=3, fpSize=2048)
            self.mf_featurizer = fpgen.GetCountFingerprint
        elif self.struct_ipt == "fixed":
            fixed_struct_df = pd.read_parquet(fixed_struct_data)
            self.fixed_struct_dict = {
                mol: vec
                for mol, vec in zip(
                    *[fixed_struct_df.index.values, fixed_struct_df.values]
                )
            }

        if graph_data is not None:
            with open(graph_data, "rb") as f:
                self.mol_graph_dict = pickle.load(f)
        else:
            self.mol_graph_dict = None

        self.data_morph = self.f["morph"]
        self.morph_idx = self.f["morph_index"]
        self.morph_meta = self.f["morph_metadata"]

        if ohe_batch:
            self._one_hot_encode_batch()
        else:
            self.batch_assignments = None

        self.morph_aug = morph_aug

    def _one_hot_encode_batch(self):
        ohe = OneHotEncoder()
        batch_ohe = ohe.fit_transform(self.morph_meta[:, [self.morph_meta_idx]])
        batch_assignments = []
        for start_idx, end_idx in self.morph_idx[:]:
            batch_assignments.append(
                np.array(batch_ohe[start_idx:end_idx].sum(0)).squeeze()
            )
        self.batch_assignments = np.array(batch_assignments)

    def _get_mod_feat(self, idx, mod):
        mol = self.ids[idx]
        if mod == "struct":
            if self.struct_ipt == "graph":
                if self.mol_graph_dict is None:
                    feat = mol_to_data(mol, mode="smile", label=None, mol_features=None)
                    feat.mols = mol
                else:
                    feat = self.mol_graph_dict[mol]
            elif self.struct_ipt == "mf":
                feat = mol_to_mf(mol, self.mf_featurizer, mode="smile", output="tensor")
            elif self.struct_ipt == "fixed":
                feat = torch.FloatTensor(self.fixed_struct_dict[mol])

        elif mod == "morph":
            mol_sidx, mol_eidx = self.morph_idx[idx]
            morph_feat_pool = self.data_morph[mol_sidx:mol_eidx, :]

            if self.morph_aug is not None:
                feat = self.morph_aug(morph_feat_pool)
            else:
                if self.avg_morph_feat:
                    feat = morph_feat_pool.mean(0)
                else:
                    num_replicates = morph_feat_pool.shape[0]
                    ix = np.random.choice(num_replicates, size=1, replace=False)
                    feat = torch.FloatTensor(morph_feat_pool[ix, :].flatten())
            feat = torch.FloatTensor(feat)
        else:
            raise ValueError(f"Modality {mod} not implemented")

        return feat

    def _get_metadata(self, idx):
        mol_sidx, mol_eidx = self.morph_idx[idx]
        return self.morph_meta[mol_sidx:mol_eidx, self.morph_meta_idx]


class CustomSubset(Subset):
    """Custom subset that preserves dataset attributes."""

    def __init__(self, dataset, indices, filter_source=None):
        super().__init__(dataset, indices)
        self.ids = self.dataset.ids[self.indices] if self.dataset.ids is not None else None

        if self.dataset.y.shape[0] == self.dataset.ids.shape[0]:
            self.y = self.dataset.y[self.indices]
        else:
            self.y = torch.Tensor([-1])
        self.filter_source = filter_source

        self.batch_assignments = (
            self.dataset.batch_assignments[self.indices]
            if self.dataset.batch_assignments is not None
            else None
        )

    def __getitem__(self, idx):
        self.dataset.filter_source = self.filter_source
        item = super().__getitem__(idx)
        return item

    def __len__(self):
        return len(self.ids)

    def __getattr__(self, item):
        return getattr(self.dataset, item)


class StructPredictionDataset(torch.utils.data.Dataset):
    """Dataset for structure-based prediction/inference."""

    def __init__(
        self,
        df,
        mol_col="Metadata_InChI",
        struct_ipt="graph",
        mol_notation="inchi",
        target_col=None,
        fixed_struct_df=None,
        device=None,
        seed=0,
    ):
        mols = df[mol_col].values
        targets = df[target_col].values if target_col is not None else [-1] * df.shape[0]

        self.struct_ipt = struct_ipt
        self.mol_notation = mol_notation
        self.ids = mols
        self.mols = mols
        self.y = torch.Tensor(targets)

        if self.struct_ipt == "mf":
            fpgen = Chem.AllChem.GetMorganGenerator(radius=3, fpSize=2048)
            self.mf_featurizer = fpgen.GetCountFingerprint
        elif self.struct_ipt == "fixed":
            self.fixed_struct_dict = {
                mol: vec
                for mol, vec in zip(
                    *[fixed_struct_df.index.values, fixed_struct_df.values]
                )
            }

        self.device = device
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __len__(self):
        return len(self.mols)

    @staticmethod
    def smile_to_data(smile, mode="smile", label=None, mol_features=None):
        return mol_to_data(smile, mode=mode, label=label, mol_features=mol_features)

    @staticmethod
    def inchi_to_data(inchi, mode="inchi", label=None, mol_features=None):
        return mol_to_data(inchi, mode=mode, label=label, mol_features=mol_features)

    def __getitem__(self, idx):
        mol = self.mols[idx]
        if self.struct_ipt == "graph":
            feat = mol_to_data(mol, mode=self.mol_notation, label=None, mol_features=None)
            feat.mols = mol
        elif self.struct_ipt == "mf":
            feat = mol_to_mf(
                mol, self.mf_featurizer, mode=self.mol_notation, output="tensor"
            )
        elif self.struct_ipt == "fixed":
            feat = torch.FloatTensor(self.fixed_struct_dict[mol])
        else:
            raise NotImplementedError

        return {
            "inputs": {
                "struct": feat,
            },
            "labels": self.y[idx],
        }


