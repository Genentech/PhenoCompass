"""PhenoCompass: phenotypic virtual screening via multimodal contrastive embeddings.

Embeds query SMILES with a 12-model ensemble (two model families x six seeds)
and scores them against precomputed MoA anchor embeddings using cosine similarity.

Expected layout of ``final_model_dir``::

    final_model/
    ├── model_ckpts/
    │   ├── gneprop/epoch=26-step=2973239.ckpt
    │   ├── <combined_fp_prefix>_{0..5}/epoch=XX-best-h5_batch_match.ckpt
    │   └── <consistency_reg_prefix>_{0..5}/epoch=XX-best-h5_batch_match.ckpt
    └── anchors/
        └── anchor_embeddings/  (parquet per model/seed/cluster/modality)

``final_model_dir`` resolution: constructor arg > env var
``PHENOCOMPASS_FINAL_MODEL_DIR`` > ``__file__``-relative default.
"""

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from phenocompass.models.contrastive import TripleContrastive_PL
from phenocompass.inference.ensemble import EnsembleOfModels
from phenocompass.evaluation.metrics import compute_fewshot_similarities

# Path resolution: inference/ -> phenocompass/ (pkg) -> src/ -> phenocompass/ (repo root)
_PACKAGE_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_FINAL_MODEL_DIR = os.environ.get(
    "PHENOCOMPASS_FINAL_MODEL_DIR",
    str(_PACKAGE_ROOT / "final_model"),
)

# Model-prefix strings shared between anchor loading and checkpoint enumeration.
_MODEL_PREFIXES = {
    "combined_fp": (
        "jump_mapv3a_dino1_tvn_gmc_combined_fp_gneprop_"
        "final_index_predetermined_bs128_0327"
    ),
    "consistency_reg": (
        "alphaNone_drop0.1_encode256_latent256_gmc_"
        "consist_alpha10_temp0.5_final"
    ),
}

# Best-epoch checkpoint filenames per seed (0-5).
_BEST_CKPT_FILENAMES = {
    "combined_fp": [
        "epoch=37-best-h5_batch_match.ckpt",
        "epoch=40-best-h5_batch_match.ckpt",
        "epoch=48-best-h5_batch_match.ckpt",
        "epoch=33-best-h5_batch_match.ckpt",
        "epoch=37-best-h5_batch_match.ckpt",
        "epoch=33-best-h5_batch_match.ckpt",
    ],
    "consistency_reg": [
        "epoch=122-best-h5_batch_match.ckpt",
        "epoch=168-best-h5_batch_match.ckpt",
        "epoch=139-best-h5_batch_match.ckpt",
        "epoch=97-best-h5_batch_match.ckpt",
        "epoch=87-best-h5_batch_match.ckpt",
        "epoch=71-best-h5_batch_match.ckpt",
    ],
}

_GNEPROP_CKPT_SUBPATH = os.path.join("gneprop", "epoch=26-step=2973239.ckpt")

_CLUSTER_IDX_TO_MOA = {
    3: "mtor_pi3k",
    6: "hsp90",
    7: "jak_rock",
    8: "hdac",
    10: "mapk",
    11: "cdk",
}


class PhenoCompass:
    """Embed and score molecules against MoA anchors using a 12-model ensemble.

    Parameters
    ----------
    anchor_modality : str
        Anchor embedding space to use: ``"joint"``, ``"struct"``, or ``"morph"``.
    device : str
        Torch device for inference.
    verbose : bool
        Print progress during loading.
    final_model_dir : str or None
        Root directory with ``model_ckpts/`` and ``anchors/``.
        Defaults via env var or ``__file__``-relative path (see module docstring).

    Examples
    --------
    >>> from phenocompass import PhenoCompass
    >>> model = PhenoCompass(device="cuda")
    >>> scores = model.score_against_anchors(["CCO", "c1ccccc1"])
    >>> struct_emb = model.compute_struct_embeddings(["CCO", "c1ccccc1"])
    """

    N_SEEDS = 6

    def __init__(
        self,
        anchor_modality="joint",
        device="cpu",
        verbose=False,
        final_model_dir=None,
    ):
        self.final_model_dir = final_model_dir or _DEFAULT_FINAL_MODEL_DIR
        self.device = device
        self.verbose = verbose
        self._anchor_modality = anchor_modality

        self._validate_model_dir()
        self.load_anchor_molecules_data(modality=anchor_modality)
        self.load_model_ensemble()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_models(self) -> int:
        """Number of models in the ensemble."""
        return len(self.ensemble.model_list)

    @property
    def model_keys(self) -> List[str]:
        """Model identifiers in the ensemble (e.g., 'combined_fp_0')."""
        return list(self.ensemble.model_class_list)

    @property
    def latent_dim(self) -> int:
        """Dimensionality of the latent embedding space."""
        # Get from first model's encoder output dimension
        first_model = self.ensemble.model_list[0]
        return first_model.hparams.get("latent_dim", 256)

    # ------------------------------------------------------------------
    # Directory validation
    # ------------------------------------------------------------------

    def _validate_model_dir(self):
        """Check that required subdirectories exist; raise FileNotFoundError if not."""
        required_subpaths = [
            "model_ckpts",
            os.path.join("model_ckpts", "gneprop"),
            os.path.join("anchors", "anchor_embeddings"),
        ]
        for subpath in required_subpaths:
            full = os.path.join(self.final_model_dir, subpath)
            if not os.path.isdir(full):
                raise FileNotFoundError(
                    f"Required subdirectory '{subpath}' not found inside "
                    f"final_model_dir={self.final_model_dir!r}.  Verify the "
                    f"directory layout or set PHENOCOMPASS_FINAL_MODEL_DIR."
                )

    # ------------------------------------------------------------------
    # Anchor embeddings
    # ------------------------------------------------------------------

    def load_anchor_molecules_data(self, modality="joint"):
        """Load precomputed anchor embeddings for every MoA cluster and model.

        Populates ``self.anchor_embeddings_dict[moa]["{family}_{seed}"]``
        with DataFrames (SMILES index, latent-dim columns) read from parquet.
        """
        if self.verbose:
            print("Loading anchor molecules data...")

        anchor_embeddings_dir = os.path.join(
            self.final_model_dir, "anchors", "anchor_embeddings"
        )

        self.anchor_embeddings_dict = defaultdict(dict)
        for cluster_idx, moa in _CLUSTER_IDX_TO_MOA.items():
            self.anchor_embeddings_dict[moa] = {}

            for model_family, prefix in _MODEL_PREFIXES.items():
                for seed in range(self.N_SEEDS):
                    fname = (
                        f"anchor_embeddings.cluster{cluster_idx}"
                        f".{prefix}.{seed}.{modality}.parquet"
                    )
                    path = os.path.join(anchor_embeddings_dir, fname)
                    embeddings = pd.read_parquet(path)

                    key = f"{model_family}_{seed}"
                    self.anchor_embeddings_dict[moa][key] = embeddings

    def get_anchor_moas(self) -> List[str]:
        """Get list of available MoA anchor categories.

        Returns
        -------
        List[str]
            MoA identifiers (e.g., ["mtor_pi3k", "hsp90", "jak_rock", ...])
        """
        return list(_CLUSTER_IDX_TO_MOA.values())

    def get_anchor_embeddings(
        self,
        moa: str,
        model_key: Optional[str] = None,
    ) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
        """Get precomputed anchor embeddings for a specific MoA.

        Parameters
        ----------
        moa : str
            MoA identifier (e.g., "mtor_pi3k", "hsp90").
        model_key : str, optional
            Specific model key (e.g., "combined_fp_0"). If None, returns
            dict of all models.

        Returns
        -------
        Union[Dict[str, pd.DataFrame], pd.DataFrame]
            If model_key is None: dict mapping model_key -> DataFrame
            If model_key provided: single DataFrame with SMILES index
        """
        if moa not in self.anchor_embeddings_dict:
            raise ValueError(
                f"Unknown MoA '{moa}'. Available: {self.get_anchor_moas()}"
            )

        if model_key is None:
            return dict(self.anchor_embeddings_dict[moa])
        else:
            if model_key not in self.anchor_embeddings_dict[moa]:
                raise ValueError(
                    f"Unknown model_key '{model_key}'. Available: {self.model_keys}"
                )
            return self.anchor_embeddings_dict[moa][model_key]

    # ------------------------------------------------------------------
    # Ensemble loading
    # ------------------------------------------------------------------

    def load_model_ensemble(self):
        """Load 12 PhenoCompass model checkpoints (combined_fp 0-5, then
        consistency_reg 0-5) plus the GNEprop encoder into an EnsembleOfModels.
        """
        if self.verbose:
            print("Loading ensemble of models...")

        ckpts_dir = os.path.join(self.final_model_dir, "model_ckpts")

        ckpt_paths = []
        model_class_list = []
        for family in ("combined_fp", "consistency_reg"):
            prefix = _MODEL_PREFIXES[family]
            for seed, ckpt_fname in enumerate(_BEST_CKPT_FILENAMES[family]):
                subdir = f"{prefix}_{seed}"
                ckpt_paths.append(os.path.join(ckpts_dir, subdir, ckpt_fname))
                model_class_list.append(f"{family}_{seed}")

        model_list = [
            TripleContrastive_PL.load_from_checkpoint(p) for p in ckpt_paths
        ]

        gneprop_ckpt_path = os.path.join(ckpts_dir, _GNEPROP_CKPT_SUBPATH)
        self.ensemble = EnsembleOfModels(
            model_list, model_class_list, gneprop_ckpt_path, device=self.device
        )

    # ------------------------------------------------------------------
    # Embedding computation
    # ------------------------------------------------------------------

    def compute_struct_embeddings(
        self,
        smiles: List[str],
        batch_size: int = 512,
    ) -> np.ndarray:
        """Compute structure embeddings for query molecules.

        Parameters
        ----------
        smiles : List[str]
            List of SMILES strings for query compounds.
        batch_size : int, default=512
            Batch size for processing.

        Returns
        -------
        np.ndarray
            Shape (n_models, n_molecules, latent_dim)
        """
        return self.ensemble.compute_representations(
            {"struct": smiles}, mod_name="struct", batch_size=batch_size
        )

    def compute_morph_embeddings(
        self,
        morph_features: np.ndarray,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Compute morphology embeddings from Cell Painting features.

        Parameters
        ----------
        morph_features : np.ndarray
            Raw Cell Painting features, shape (n_samples, feature_dim).
        batch_size : int, default=512
            Batch size for processing.

        Returns
        -------
        np.ndarray
            Shape (n_models, n_samples, latent_dim)
        """
        return self.ensemble.compute_representations(
            {"morph": morph_features}, mod_name="morph", batch_size=batch_size
        )

    def compute_joint_embeddings(
        self,
        smiles: List[str],
        morph_features: np.ndarray,
        batch_size: int = 512,
    ) -> np.ndarray:
        """Compute joint embeddings from both structure and morphology.

        Parameters
        ----------
        smiles : List[str]
            List of SMILES strings for query compounds.
        morph_features : np.ndarray
            Raw Cell Painting features, shape (n_samples, feature_dim).
            Must have same length as smiles.
        batch_size : int, default=512
            Batch size for processing.

        Returns
        -------
        np.ndarray
            Shape (n_models, n_samples, latent_dim)
        """
        if len(smiles) != morph_features.shape[0]:
            raise ValueError(
                f"smiles length ({len(smiles)}) must match "
                f"morph_features rows ({morph_features.shape[0]})"
            )

        return self.ensemble.compute_representations(
            {"struct": smiles, "morph": morph_features},
            mod_name="joint",
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score_against_anchors(
        self,
        query_mols: List[str],
        moa_list: Optional[List[str]] = None,
        metric: str = "cosine",
        mode: str = "mean",
    ) -> pd.DataFrame:
        """Score molecules against MoA anchor sets.

        Embeds queries via the ensemble, computes per-model similarity to
        anchors, and averages across models.

        Parameters
        ----------
        query_mols : List[str]
            List of SMILES strings for query compounds.
        moa_list : List[str], optional
            MoA categories to score against. Default: all available MoAs.
        metric : str, default="cosine"
            Similarity metric: "cosine" or "tanimoto".
        mode : str, default="mean"
            Aggregation mode for anchor similarity: "mean" or "max".

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by query_mols with columns "<moa>_score"
            for each MoA.
        """
        if moa_list is None:
            moa_list = ["mtor_pi3k", "hsp90", "jak_rock", "hdac", "mapk", "cdk"]

        query_embeddings = self.compute_struct_embeddings(query_mols)

        score_df = pd.DataFrame(index=query_mols)
        for moa in moa_list:
            per_model_scores = np.array([
                compute_fewshot_similarities(
                    self.anchor_embeddings_dict[moa][model_class],
                    query_embeddings[i],
                    metric=metric,
                    mode=mode,
                )
                for i, model_class in enumerate(self.ensemble.model_class_list)
            ])
            score_df[f"{moa}_score"] = per_model_scores.mean(axis=0)

        return score_df

    def score_against_reference_groups(
        self,
        query_smiles: List[str],
        group_to_smiles: Dict[str, List[str]],
        reference_morph_df: Optional[pd.DataFrame] = None,
        metric: str = "cosine",
        aggregation: str = "mean",
        batch_size: int = 512,
    ) -> pd.DataFrame:
        """Score query compounds against user-defined reference groups.

        Supports many-to-many molecule-to-group membership: the same SMILES
        can appear in multiple groups.  All unique reference molecules are
        embedded once and reused across groups for efficiency.

        Parameters
        ----------
        query_smiles : List[str]
            SMILES strings for query compounds (embedded as structure).
        group_to_smiles : Dict[str, List[str]]
            Mapping from group name (e.g., MoA) to a list of SMILES that
            belong to that group.  A SMILES may appear in multiple groups.
        reference_morph_df : Optional[pd.DataFrame]
            DataFrame indexed by SMILES with morphology feature columns.
            When provided, joint (structure + morphology) embeddings are
            used for references; otherwise structure-only embeddings are
            used.  Must contain rows for every unique SMILES present in
            ``group_to_smiles``.
        metric : str, default="cosine"
            Similarity metric: ``"cosine"`` or ``"tanimoto"``.
        aggregation : str, default="mean"
            Within-group aggregation: ``"mean"`` or ``"max"``.
        batch_size : int, default=512
            Batch size for embedding computation.

        Returns
        -------
        pd.DataFrame
            Shape ``(n_queries, n_groups)`` with query SMILES as index and
            group names as columns.  Each value is the ensemble-averaged,
            within-group-aggregated similarity score.

        Raises
        ------
        ValueError
            If ``group_to_smiles`` is empty, or if ``reference_morph_df``
            is provided but is missing rows for some reference SMILES.
        """
        if not group_to_smiles:
            raise ValueError("group_to_smiles must be a non-empty dict")

        # Deduplicate while preserving a stable order
        unique_smiles_list: List[str] = list(
            dict.fromkeys(
                smi for smi_list in group_to_smiles.values() for smi in smi_list
            )
        )
        smiles_to_idx = {smi: i for i, smi in enumerate(unique_smiles_list)}

        # --- Compute reference embeddings once for all unique SMILES ---
        if reference_morph_df is not None:
            missing = set(unique_smiles_list) - set(reference_morph_df.index)
            if missing:
                raise ValueError(
                    f"reference_morph_df is missing rows for {len(missing)} "
                    f"reference SMILES.  First few: {list(missing)[:5]}"
                )
            morph_features = reference_morph_df.loc[unique_smiles_list].values
            ref_embeddings = self.compute_joint_embeddings(
                unique_smiles_list, morph_features, batch_size=batch_size
            )
        else:
            ref_embeddings = self.compute_struct_embeddings(
                unique_smiles_list, batch_size=batch_size
            )

        # --- Compute query structure embeddings ---
        query_embeddings = self.compute_struct_embeddings(
            query_smiles, batch_size=batch_size
        )

        n_models = query_embeddings.shape[0]
        n_queries = len(query_smiles)
        group_names = list(group_to_smiles.keys())

        # Pre-compute index arrays for each group
        group_indices = {
            group: np.array([smiles_to_idx[s] for s in smi_list if s in smiles_to_idx])
            for group, smi_list in group_to_smiles.items()
        }

        # --- Score per model, per group ---
        per_model_group_scores = np.zeros((n_models, n_queries, len(group_names)))

        for model_idx in range(n_models):
            query_emb = query_embeddings[model_idx]
            ref_emb = ref_embeddings[model_idx]

            for group_idx, group in enumerate(group_names):
                idx = group_indices[group]
                if len(idx) == 0:
                    continue
                group_ref_emb = ref_emb[idx]
                per_model_group_scores[model_idx, :, group_idx] = (
                    compute_fewshot_similarities(
                        group_ref_emb, query_emb, metric=metric, mode=aggregation,
                    )
                )

        avg_group_scores = per_model_group_scores.mean(axis=0)

        return pd.DataFrame(
            avg_group_scores, index=query_smiles, columns=group_names,
        )

    # ------------------------------------------------------------------
    # Cross-modal similarity search
    # ------------------------------------------------------------------

    def compute_crossmodal_similarity(
        self,
        query_smiles: List[str],
        reference_morph: Optional[np.ndarray] = None,
        reference_smiles: Optional[List[str]] = None,
        reference_ids: Optional[List[str]] = None,
        group_labels: Optional[Union[List[str], np.ndarray]] = None,
        aggregation: str = "mean",
        batch_size: int = 512,
    ) -> pd.DataFrame:
        """Score query compounds against reference profiles.

        Embeds query SMILES using the 12-model ensemble to obtain structure
        embeddings. Reference embeddings are computed based on provided inputs:
        - Structure-only: when only `reference_smiles` is provided
        - Morphology-only: when only `reference_morph` is provided
        - Joint (structure + morphology): when both are provided

        Parameters
        ----------
        query_smiles : List[str]
            List of SMILES strings for query compounds.
        reference_morph : Optional[np.ndarray]
            Raw Cell Painting features, shape (n_refs, feature_dim).
            Required for morphology-only or joint reference embeddings.
        reference_smiles : Optional[List[str]]
            SMILES strings for reference compounds.
            Required for structure-only or joint reference embeddings.
        reference_ids : Optional[List[str]]
            Identifiers for reference profiles. If None, uses integer indices.
        group_labels : Optional[Union[List[str], np.ndarray]]
            Group identifiers for reference profiles. When provided, returns
            group-averaged similarity scores instead of per-profile scores.
        aggregation : str, default="mean"
            Aggregation method for group-level similarity: "mean" or "max".
        batch_size : int, default=512
            Batch size for embedding computation.

        Returns
        -------
        pd.DataFrame
            If group_labels is None: shape (n_queries, n_refs)
            If group_labels provided: shape (n_queries, n_groups)

        Raises
        ------
        ValueError
            If neither reference_morph nor reference_smiles is provided, or if
            their lengths don't match when both are provided.
        """
        # Validate at least one reference input is provided
        if reference_morph is None and reference_smiles is None:
            raise ValueError(
                "At least one of reference_morph or reference_smiles must be provided"
            )

        # Determine n_refs from available input
        if reference_morph is not None:
            n_refs = reference_morph.shape[0]
        else:
            n_refs = len(reference_smiles)

        # Validate consistency when both inputs provided
        if reference_morph is not None and reference_smiles is not None:
            if len(reference_smiles) != reference_morph.shape[0]:
                raise ValueError(
                    f"reference_smiles length ({len(reference_smiles)}) must match "
                    f"reference_morph rows ({reference_morph.shape[0]}) for joint embeddings"
                )

        # Validate reference_ids length
        if reference_ids is not None and len(reference_ids) != n_refs:
            raise ValueError(
                f"reference_ids length ({len(reference_ids)}) must match "
                f"number of references ({n_refs})"
            )

        # Validate group_labels length
        if group_labels is not None and len(group_labels) != n_refs:
            raise ValueError(
                f"group_labels length ({len(group_labels)}) must match "
                f"number of references ({n_refs})"
            )

        # Set default reference IDs if not provided
        if reference_ids is None:
            reference_ids = [f"ref_{i}" for i in range(n_refs)]

        # Compute query structure embeddings: (n_models, n_queries, latent_dim)
        query_embeddings = self.compute_struct_embeddings(
            query_smiles, batch_size=batch_size
        )

        # Compute reference embeddings: (n_models, n_refs, latent_dim)
        if reference_morph is not None and reference_smiles is not None:
            # Joint embeddings when both are provided
            ref_embeddings = self.compute_joint_embeddings(
                reference_smiles, reference_morph, batch_size=batch_size
            )
        elif reference_morph is not None:
            # Morphology-only embeddings
            ref_embeddings = self.compute_morph_embeddings(
                reference_morph, batch_size=batch_size
            )
        else:
            # Structure-only embeddings
            ref_embeddings = self.compute_struct_embeddings(
                reference_smiles, batch_size=batch_size
            )

        n_models = query_embeddings.shape[0]
        n_queries = len(query_smiles)

        if group_labels is None:
            # Profile-level similarity: compute full similarity matrix per model
            per_model_similarities = []
            for model_idx in range(n_models):
                query_emb = query_embeddings[model_idx]  # (n_queries, latent_dim)
                ref_emb = ref_embeddings[model_idx]  # (n_refs, latent_dim)
                sim_matrix = cosine_similarity(query_emb, ref_emb)
                per_model_similarities.append(sim_matrix)

            # Average across models: (n_queries, n_refs)
            avg_similarities = np.mean(per_model_similarities, axis=0)

            return pd.DataFrame(
                avg_similarities,
                index=query_smiles,
                columns=reference_ids,
            )

        else:
            # Group-level similarity: use compute_fewshot_similarities per group
            group_labels = np.array(group_labels)
            unique_groups = list(dict.fromkeys(group_labels))  # Preserve order
            n_groups = len(unique_groups)

            per_model_group_scores = []
            for model_idx in range(n_models):
                query_emb = query_embeddings[model_idx]  # (n_queries, latent_dim)
                ref_emb = ref_embeddings[model_idx]  # (n_refs, latent_dim)

                group_scores = np.zeros((n_queries, n_groups))
                for group_idx, group in enumerate(unique_groups):
                    group_mask = group_labels == group
                    group_ref_emb = ref_emb[group_mask]

                    # compute_fewshot_similarities expects (train, test) and returns (n_test,)
                    scores = compute_fewshot_similarities(
                        group_ref_emb,
                        query_emb,
                        metric="cosine",
                        mode=aggregation,
                    )
                    group_scores[:, group_idx] = scores

                per_model_group_scores.append(group_scores)

            # Average across models: (n_queries, n_groups)
            avg_group_scores = np.mean(per_model_group_scores, axis=0)

            return pd.DataFrame(
                avg_group_scores,
                index=query_smiles,
                columns=unique_groups,
            )

    def compute_enrichment_analysis(
        self,
        query_smiles: List[str],
        reference_morph: Optional[np.ndarray] = None,
        reference_smiles: Optional[List[str]] = None,
        group_annotations: Optional[Union[List[str], np.ndarray]] = None,
        k: int = 10,
        min_group_size: int = 1,
        correction_method: Optional[str] = "fdr_bh",
        batch_size: int = 512,
    ) -> pd.DataFrame:
        """Compute enrichment analysis for query compounds against annotated reference groups.

        Convenience method that combines similarity computation and enrichment testing.
        Computes cross-modal similarities between queries and references, then tests
        for hypergeometric enrichment of each group among top-k neighbors.

        Parameters
        ----------
        query_smiles : List[str]
            SMILES strings for query compounds.
        reference_morph : Optional[np.ndarray]
            Raw Cell Painting features for references, shape (n_refs, feature_dim).
        reference_smiles : Optional[List[str]]
            SMILES for reference compounds.
        group_annotations : Union[List[str], np.ndarray]
            Group label for each reference (e.g., MoA). Length must equal n_refs.
        k : int, default=10
            Number of top neighbors to test for enrichment.
        min_group_size : int, default=1
            Minimum group size to include.
        correction_method : Optional[str], default="fdr_bh"
            Multiple testing correction: "fdr_bh", "bonferroni", or None.
        batch_size : int, default=512
            Batch size for embedding computation.

        Returns
        -------
        pd.DataFrame
            Enrichment results with one row per (query, group) pair. Columns:
            - "query": Query identifier
            - "group": Group name being tested
            - "k": Number of neighbors considered
            - "hits_in_group": Count of top-k neighbors in this group
            - "group_size": Total references in this group
            - "n_refs": Total number of references
            - "expected_hits": Expected hits under null
            - "fold_enrichment": hits_in_group / expected_hits
            - "p_value": Raw hypergeometric p-value
            - "p_adjusted": Corrected p-value

        See Also
        --------
        compute_crossmodal_similarity : Compute raw similarity scores.
        phenocompass.evaluation.compute_enrichment_analysis : Standalone function.
        """
        from phenocompass.evaluation.enrichment import (
            compute_enrichment_analysis as _enrichment,
        )

        # Compute similarities
        similarity_scores = self.compute_crossmodal_similarity(
            query_smiles=query_smiles,
            reference_morph=reference_morph,
            reference_smiles=reference_smiles,
            batch_size=batch_size,
        )

        # Run enrichment analysis
        return _enrichment(
            similarity_scores=similarity_scores,
            group_annotations=group_annotations,
            k=k,
            min_group_size=min_group_size,
            correction_method=correction_method,
        )
