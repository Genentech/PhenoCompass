"""Model ensemble and featurization utilities for inference."""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Avalon.pyAvalonTools import GetAvalonCountFP
from rdkit.Chem.rdReducedGraphs import (
    GenerateErGFingerprintForReducedGraph,
    GenerateMolExtendedReducedGraph,
)
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors
from descriptastorus.descriptors.rdDescriptors import RDKIT_PROPS

from phenocompass.data.dataset import StructPredictionDataset
from phenocompass.data.featurization import mol_to_mf
from phenocompass.models.components import GNEpropGIN
from phenocompass.models.utils import move_batch_input_to_device


class MoleculeFeaturizer:
    """Standalone molecule featurizer.

    Computes multi-view molecular features including GNEprop embeddings,
    ECFP, Avalon, ERG, and RDKit descriptors.

    Parameters
    ----------
    pretrained_gneprop_ckpt_path : str
        Path to pretrained GNEprop GIN checkpoint.
    use_normalized_rdkit : bool, default True
        Whether to use normalized RDKit 2D descriptors.
    """

    def __init__(self, pretrained_gneprop_ckpt_path, use_normalized_rdkit=True):
        self.use_normalized_rdkit = use_normalized_rdkit
        self.load_pretrained_gneprop(pretrained_gneprop_ckpt_path)
        self.initialize_mf_featurizer()
        self.initialize_rdkit_featurizer()

    def load_pretrained_gneprop(self, pretrained_gneprop_ckpt_path):
        """Load pretrained GNEprop GIN model."""
        in_channels = 133
        edge_dim = 12
        hidden_channels = 500
        ffn_hidden_channels = None
        num_layers = 5
        out_channels = 1
        num_readout_layers = 2
        mol_features_size = 0
        dropout = 0.0

        self.gneprop = GNEpropGIN(
            in_channels,
            edge_dim,
            hidden_channels,
            ffn_hidden_channels,
            num_layers,
            out_channels,
            num_readout_layers,
            dropout,
            mol_features_size,
        )

        self.gneprop.load_pretrained(pretrain_path=pretrained_gneprop_ckpt_path)
        self.gneprop.eval()

    def initialize_mf_featurizer(self):
        """Initialize Morgan fingerprint generator."""
        fpgen = AllChem.GetMorganGenerator(radius=3, fpSize=2048)
        self.mf_featurizer = fpgen.GetCountFingerprint

    def initialize_rdkit_featurizer(self):
        """Initialize RDKit descriptor generator."""
        self.rdkit_feature_names = RDKIT_PROPS["1.0.0"]
        if self.use_normalized_rdkit:
            self.rdkit_feature_generator = rdNormalizedDescriptors.RDKit2DNormalized()
        else:
            self.rdkit_feature_generator = rdDescriptors.RDKit2D()

    def ecfp_featurizer(self, mol):
        """Compute ECFP fingerprint for a molecule."""
        return mol_to_mf(mol, self.mf_featurizer, output="vector")

    def erg_featurizer(self, mol, mol_notation="smiles"):
        """Compute extended reduced graph fingerprint."""
        return np.array(
            GenerateErGFingerprintForReducedGraph(
                GenerateMolExtendedReducedGraph(Chem.MolFromSmiles(mol))
            )
        )

    def avalon_featurizer(self, mol, mol_notation="smiles", nBits=1024):
        """Compute Avalon fingerprint."""
        if mol_notation == "smiles":
            return np.array(
                GetAvalonCountFP(Chem.MolFromSmiles(mol), nBits=nBits).ToList()
            )

    def rdkit_featurizer(self, mol):
        """Compute RDKit 2D descriptors."""
        return np.array(self.rdkit_feature_generator.process(mol)[1:])

    def gneprop_compute_representations_dataloader(
        self, mols, batch_size=5096, mol_notation="smile"
    ):
        """Compute GNEprop embeddings for a list of molecules."""
        mol_df = pd.DataFrame(mols, columns=[mol_notation])
        dataset = StructPredictionDataset(
            mol_df, mol_col=mol_notation, mol_notation=mol_notation
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        combined_embeddings = []
        for batch in dataloader:
            gneprop_embeddings = self.gneprop(batch["inputs"]["struct"])
            combined_embeddings.append(gneprop_embeddings)

        return np.concatenate(combined_embeddings)

    def compute_representations_dataloader(
        self,
        mols,
        batch_size=5096,
        mol_notation="smile",
        feature_list=None,
    ):
        """Compute combined molecular representations.

        Parameters
        ----------
        mols : list
            List of SMILES strings.
        batch_size : int, default 5096
            Batch size for processing.
        mol_notation : str, default "smile"
            Molecule notation type.
        feature_list : list, optional
            Features to compute. Default: ["gneprop", "ecfp", "avalon", "erg", "rdkit"]

        Returns
        -------
        np.ndarray
            Combined feature matrix.
        """
        if feature_list is None:
            feature_list = ["gneprop", "ecfp", "avalon", "erg", "rdkit"]

        mol_df = pd.DataFrame(mols, columns=[mol_notation])
        dataset = StructPredictionDataset(
            mol_df, mol_col=mol_notation, mol_notation=mol_notation
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        combined_embeddings = []
        for batch in dataloader:
            embeddings_list = []
            for feature in feature_list:
                if feature == "gneprop":
                    embeddings = self.gneprop(batch["inputs"]["struct"]).cpu().data.numpy()
                elif feature == "ecfp":
                    embeddings = np.array(
                        [self.ecfp_featurizer(mol) for mol in batch["inputs"]["struct"].mols]
                    )
                elif feature == "avalon":
                    embeddings = np.array(
                        [self.avalon_featurizer(mol) for mol in batch["inputs"]["struct"].mols]
                    )
                elif feature == "erg":
                    embeddings = np.array(
                        [self.erg_featurizer(mol) for mol in batch["inputs"]["struct"].mols]
                    )
                elif feature == "rdkit":
                    embeddings = np.array(
                        [self.rdkit_featurizer(mol) for mol in batch["inputs"]["struct"].mols]
                    )
                else:
                    raise NotImplementedError(f"Unknown feature: {feature}")

                embeddings_list.append(embeddings)
            combined_embeddings.append(np.concatenate(embeddings_list, axis=1))

        return np.concatenate(combined_embeddings)


class EnsembleOfModels:
    """Ensemble of contrastive models for inference.

    Wraps multiple TripleContrastive_PL models and a shared MoleculeFeaturizer
    for computing embeddings across the ensemble.

    Parameters
    ----------
    model_list : list
        List of TripleContrastive_PL model instances.
    model_class_list : list
        List of model identifiers (e.g., "combined_fp_0", "consistency_reg_1").
    gneprop_ckpt_path : str
        Path to pretrained GNEprop checkpoint.
    device : str or torch.device, optional
        Device for computation.
    """

    def __init__(self, model_list, model_class_list, gneprop_ckpt_path, device=None):
        self.model_list = model_list
        self.model_class_list = model_class_list
        self.mol_featurizer = MoleculeFeaturizer(gneprop_ckpt_path)
        self.precompute_mol_features = any(
            ["combined_fp" in m for m in model_class_list]
        )
        self.device = self.model_list[0].device if device is None else device
        for model in self.model_list:
            model.to(self.device)

    def compute_representations(
        self,
        x_dict: Dict[str, Any],
        mod_name: str = "struct",
        batch_size: int = 512,
    ) -> np.ndarray:
        """Compute embeddings for the specified modality.

        Parameters
        ----------
        x_dict : dict
            Dictionary of inputs by modality:
            - "struct": List[str] of SMILES
            - "morph": np.ndarray of shape (n_samples, feature_dim)
            For "joint", provide both "struct" and "morph"
        mod_name : str
            Modality to compute: "struct", "morph", or "joint"
        batch_size : int
            Batch size for processing

        Returns
        -------
        np.ndarray
            Shape: (n_models, n_samples, latent_dim)
        """
        # Validate required keys based on mod_name
        if mod_name == "struct":
            if "struct" not in x_dict:
                raise ValueError("x_dict must contain 'struct' key for struct modality")
        elif mod_name == "morph":
            if "morph" not in x_dict:
                raise ValueError("x_dict must contain 'morph' key for morph modality")
        elif mod_name == "joint":
            if "struct" not in x_dict or "morph" not in x_dict:
                raise ValueError(
                    "x_dict must contain both 'struct' and 'morph' keys for joint modality"
                )
        else:
            raise ValueError(f"Unknown mod_name: {mod_name}. Use 'struct', 'morph', or 'joint'")

        # Dispatch to appropriate implementation
        if mod_name == "morph":
            return self._compute_morph_representations(x_dict["morph"], batch_size)
        else:
            # struct or joint
            morph_data = x_dict.get("morph") if mod_name == "joint" else None
            return self._compute_struct_representations(
                x_dict["struct"], batch_size, mod_name, morph_data
            )

    def _compute_struct_representations(
        self,
        mols: List[str],
        batch_size: int,
        mod_name: str,
        morph_data: Optional[np.ndarray],
    ) -> np.ndarray:
        """Internal method to compute struct or joint embeddings."""
        mol_notation = "smile"
        mol_df = pd.DataFrame(mols, columns=[mol_notation])
        dataset = StructPredictionDataset(
            mol_df, mol_col=mol_notation, mol_notation=mol_notation
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        rep_list = []
        batch_start_idx = 0
        for batch in tqdm(dataloader):
            batch_size_actual = len(batch["inputs"]["struct"].mols)
            batch_end_idx = batch_start_idx + batch_size_actual

            if self.precompute_mol_features:
                mols_features = self.mol_featurizer.compute_representations_dataloader(
                    batch["inputs"]["struct"].mols
                )
                mols_features = torch.nan_to_num(
                    torch.FloatTensor(mols_features).to(self.device)
                )

            model_reps_list = []
            for model_idx, model in enumerate(self.model_list):
                model_class = self.model_class_list[model_idx]

                if "combined_fp" in model_class:
                    batch_x_dict = {"struct": mols_features}
                else:
                    batch_x_dict = dict(batch["inputs"])

                if morph_data is not None:
                    # Slice morph_data to match current batch
                    batch_morph = morph_data[batch_start_idx:batch_end_idx]
                    batch_x_dict["morph"] = torch.FloatTensor(batch_morph)

                if mod_name == "joint":
                    batch_x_dict["joint"] = {
                        k: batch_x_dict[k] for k in ["struct", "morph"]
                    }

                batch_x_dict = move_batch_input_to_device(batch_x_dict, device=self.device)
                rep = model.compute_representations(batch_x_dict, mod_name=mod_name)

                model_reps_list.append(rep)
            rep_list.append(torch.stack(model_reps_list).cpu().data.numpy())
            batch_start_idx = batch_end_idx

        return np.concatenate(rep_list, axis=1)

    def _compute_morph_representations(
        self,
        morph_data: np.ndarray,
        batch_size: int,
    ) -> np.ndarray:
        """Internal method to compute morphology embeddings."""
        n_samples = morph_data.shape[0]
        rep_list = []

        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_morph = morph_data[start_idx:end_idx]

            model_reps_list = []
            for model in self.model_list:
                batch_x_dict = {"morph": torch.FloatTensor(batch_morph)}
                batch_x_dict = move_batch_input_to_device(batch_x_dict, device=self.device)

                with torch.no_grad():
                    rep = model.compute_representations(batch_x_dict, mod_name="morph")
                model_reps_list.append(rep.cpu().numpy())

            # Stack: (n_models, batch_size, latent_dim)
            rep_list.append(np.stack(model_reps_list, axis=0))

        # Concatenate along sample dimension: (n_models, n_samples, latent_dim)
        return np.concatenate(rep_list, axis=1)


class EnsembleWithPrecomputedRDKitFeatures(EnsembleOfModels):
    """Ensemble with separate RDKit feature computation.

    Extends EnsembleOfModels to support precomputed RDKit features, which can
    be useful when RDKit descriptors need to be computed separately (e.g.,
    for large-scale screening where RDKit computation is a bottleneck).

    Parameters
    ----------
    model_list : list
        List of TripleContrastive_PL model instances.
    model_class_list : list
        List of model identifiers.
    gneprop_ckpt_path : str
        Path to pretrained GNEprop checkpoint.
    device : str or torch.device, optional
        Device for computation.
    """

    def __init__(self, model_list, model_class_list, gneprop_ckpt_path, device=None):
        super().__init__(model_list, model_class_list, gneprop_ckpt_path, device=device)
        self.mol_features_to_compute = ["gneprop", "ecfp", "avalon", "erg"]

    def compute_representations_dataloader(
        self,
        mols,
        rdkit_features_df,
        batch_size=512,
        mol_notation="smile",
        mod_name="struct",
        morph_data=None,
    ):
        """Compute embeddings using precomputed RDKit features.

        Parameters
        ----------
        mols : list
            List of SMILES strings.
        rdkit_features_df : pd.DataFrame
            DataFrame with precomputed RDKit features, indexed by SMILES.
        batch_size : int, default 512
            Batch size for processing.
        mol_notation : str, default "smile"
            Molecule notation type.
        mod_name : str, default "struct"
            Modality to compute.
        morph_data : np.ndarray, optional
            Pre-computed morphology features.

        Returns
        -------
        np.ndarray
            Shape: (n_models, n_molecules, embedding_dim)
        """
        mol_df = pd.DataFrame(mols, columns=[mol_notation])
        dataset = StructPredictionDataset(
            mol_df, mol_col=mol_notation, mol_notation=mol_notation
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        rep_list = []
        for batch in dataloader:
            mols_graph = batch["inputs"]["struct"]
            batch_mols = batch["inputs"]["struct"].mols

            batch_rdkit_features = rdkit_features_df.loc[batch_mols].values

            batch_mols_features = self.mol_featurizer.compute_representations_dataloader(
                batch["inputs"]["struct"].mols,
                feature_list=self.mol_features_to_compute,
            )
            mols_features = np.concatenate(
                [batch_mols_features, batch_rdkit_features], axis=1
            )
            mols_features = torch.nan_to_num(
                torch.FloatTensor(mols_features).to(self.device)
            )

            model_reps_list = []
            for model_idx, model in enumerate(self.model_list):
                model = self.model_list[model_idx]
                model_class = self.model_class_list[model_idx]

                x_dict = {"struct": mols_features}

                if morph_data is not None:
                    x_dict["morph"] = torch.FloatTensor(morph_data)

                if mod_name == "joint":
                    x_dict.update({"joint": {k: x_dict[k] for k in ["struct", "morph"]}})

                x_dict = move_batch_input_to_device(x_dict, device=self.device)
                rep = model.compute_representations(x_dict, mod_name=mod_name)

                model_reps_list.append(rep)
            rep_list.append(torch.stack(model_reps_list).cpu().data.numpy())

        return np.concatenate(rep_list, axis=1)
