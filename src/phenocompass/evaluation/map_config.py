"""Configuration dataclasses and constants for JUMP map evaluation.

This module defines metadata field mappings, aggregation levels, and control
compound definitions for the JUMP Cell Painting dataset, as well as
Cell Painting validation pipeline configuration.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd


# Path to compound annotation TSV files
# evaluation/ -> phenocompass(pkg)/ -> src/ -> project_root/
_ANNOTATIONS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
    "final_data", "compound_annotations"
))


def load_negative_controls(annotations_dir=None):
    """Load JUMP negative controls from TSV -> dict[pert_type] -> list[pert_id]."""
    path = os.path.join(annotations_dir or _ANNOTATIONS_DIR, "jump_negative_controls.tsv")
    df = pd.read_csv(path, sep="\t")
    return df.groupby("perturbation_type")["perturbation_id"].apply(list).to_dict()


def load_cp_validation_controls(annotations_dir=None):
    """Load CP validation controls from TSV -> dict[control_type] -> list[compound_id]."""
    path = os.path.join(annotations_dir or _ANNOTATIONS_DIR, "map_evaluation_controls.tsv")
    df = pd.read_csv(path, sep="\t")
    return df.groupby("control_type")["compound_id"].apply(list).to_dict()


def load_dmso_jcp_id(annotations_dir=None):
    """Load DMSO JCP ID from negative controls TSV."""
    path = os.path.join(annotations_dir or _ANNOTATIONS_DIR, "jump_negative_controls.tsv")
    df = pd.read_csv(path, sep="\t")
    return df.loc[df["description"] == "DMSO", "perturbation_id"].iloc[0]


@dataclass
class JUMPFields:
    """Metadata field names for JUMP Cell Painting dataset.

    These correspond to the column names in JUMP parquet files.
    """
    source: str = "Metadata_Source"
    batch: str = "Metadata_Batch"
    plate: str = "Metadata_Plate"
    well: str = "Metadata_Well"
    perturbation: str = "Metadata_JCP2022"
    perturbation_type: str = "Metadata_PerturbationType"
    perturbation_name: str = "Metadata_Perturbation"
    plate_type: str = "Metadata_PlateType"
    row: str = "Metadata_Row"
    column: str = "Metadata_Column"
    cell_count: str = "Metadata_Cell_Count"
    image_count: str = "Metadata_Image_Count"


# Control compound definitions by perturbation type (loaded from TSV)
NEGATIVE_CONTROLS: Dict[str, List[str]] = load_negative_controls()

# DMSO JCP identifier (loaded from TSV)
DMSO_JCP_ID: str = load_dmso_jcp_id()

# Default sources to include (excluding known problematic sources 4, 7, 9, 13)
DEFAULT_INCLUDED_SOURCES: List[str] = [
    "source_1",
    "source_2",
    "source_3",
    "source_5",
    "source_6",
    "source_8",
    "source_10",
    "source_11",
]


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    # Sample filtering
    drop_unknown_perturbations: bool = True
    drop_plates_without_negcon: bool = True
    min_cell_count: Optional[int] = None
    min_image_count: Optional[int] = None

    # Feature filtering
    column_max_nan_ratio: float = 0.01
    drop_low_var_min: float = 0.0
    drop_low_var_max: float = 5.0

    # Normalization
    normalize_method: str = "platewise_median_mad"
    normalize_groupby: List[str] = field(default_factory=lambda: [
        "Metadata_Source", "Metadata_Batch", "Metadata_Plate"
    ])


@dataclass
class TVNConfig:
    """Configuration for TVN (Typical Variation Normalization) batch correction."""
    groupby: List[str] = field(default_factory=lambda: [
        "Metadata_Source", "Metadata_Batch", "Metadata_Plate"
    ])
    reg_param: float = 1e-3
    oas: bool = True
    recolor: bool = True
    use_negcon_only: bool = True


@dataclass
class EvalConfig:
    """Configuration for evaluation metrics."""
    # kNN evaluation
    knn_k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 50])
    knn_metric: str = "cosine"

    # Batch effect evaluation
    batch_effect_class_columns: List[str] = field(default_factory=lambda: [
        "Metadata_Source", "Metadata_Batch", "Metadata_Plate"
    ])

    # Reproducibility evaluation
    reproducibility_n_samples: int = 10000
    reproducibility_variables: List[str] = field(default_factory=lambda: [
        "Metadata_Source", "Metadata_Batch", "Metadata_JCP2022"
    ])

    # Multivariate evaluation
    run_multivariate: bool = True
    multivariate_k_values: List[int] = field(default_factory=lambda: [5, 10, 25, 50])


# =============================================================================
# Cell Painting Validation Controls (from nuvisan/map-lib DefaultSubsets)
# =============================================================================

CP_VALIDATION_CONTROLS: Dict[str, List[str]] = load_cp_validation_controls()


@dataclass
class CPValidationConfig:
    """Configuration for Cell Painting validation pipeline.

    Swept parameters control the hyperparameter grid search; fixed parameters
    match the nuvisan/map-lib defaults.
    """
    # Swept parameters
    zscore_spread_fn: str = "mad"
    pca_n_components: int = 128
    batch_correction: str = "tvn"  # "global_sphere", "tvn", or "none"

    # Fixed parameters
    column_max_nan_ratio: float = 0.01
    drop_low_std_min: float = 0.1
    drop_low_std_max: float = 5.0
    clip_value: float = 5.0
    sphere_reg_param: float = 0.001
    tvn_reg_param: float = 1.0
    tvn_oas: bool = False           # map-lib uses False
    tvn_recolor: bool = False       # map-lib uses False
    tvn_sphere_first: bool = True   # map-lib applies global sphere before TVN

    # Column names
    plate_col: str = "Metadata_Plate"
    perturbation_col: str = "Metadata_Perturbation"
    well_col: str = "Metadata_WellID"
    groupby: List[str] = field(default_factory=lambda: ["Metadata_Plate"])


# =============================================================================
# Way et al. 2019/2021 Feature Blacklist
# =============================================================================

# Source: map_lib/data/loaders/jump/_jump_specific.py:332-406
# Reference: doi:10.6084/m9.figshare.10255811.v3
WAY2019_FEATURE_BLACKLIST: List[str] = [
    "Nuclei_Correlation_Manders_AGP_DNA",
    "Nuclei_Correlation_Manders_AGP_ER",
    "Nuclei_Correlation_Manders_AGP_Mito",
    "Nuclei_Correlation_Manders_AGP_RNA",
    "Nuclei_Correlation_Manders_DNA_AGP",
    "Nuclei_Correlation_Manders_DNA_ER",
    "Nuclei_Correlation_Manders_DNA_Mito",
    "Nuclei_Correlation_Manders_DNA_RNA",
    "Nuclei_Correlation_Manders_ER_AGP",
    "Nuclei_Correlation_Manders_ER_DNA",
    "Nuclei_Correlation_Manders_ER_Mito",
    "Nuclei_Correlation_Manders_ER_RNA",
    "Nuclei_Correlation_Manders_Mito_AGP",
    "Nuclei_Correlation_Manders_Mito_DNA",
    "Nuclei_Correlation_Manders_Mito_ER",
    "Nuclei_Correlation_Manders_Mito_RNA",
    "Nuclei_Correlation_Manders_RNA_AGP",
    "Nuclei_Correlation_Manders_RNA_DNA",
    "Nuclei_Correlation_Manders_RNA_ER",
    "Nuclei_Correlation_Manders_RNA_Mito",
    "Nuclei_Correlation_RWC_AGP_DNA",
    "Nuclei_Correlation_RWC_AGP_ER",
    "Nuclei_Correlation_RWC_AGP_Mito",
    "Nuclei_Correlation_RWC_AGP_RNA",
    "Nuclei_Correlation_RWC_DNA_AGP",
    "Nuclei_Correlation_RWC_DNA_ER",
    "Nuclei_Correlation_RWC_DNA_Mito",
    "Nuclei_Correlation_RWC_DNA_RNA",
    "Nuclei_Correlation_RWC_ER_AGP",
    "Nuclei_Correlation_RWC_ER_DNA",
    "Nuclei_Correlation_RWC_ER_Mito",
    "Nuclei_Correlation_RWC_ER_RNA",
    "Nuclei_Correlation_RWC_Mito_AGP",
    "Nuclei_Correlation_RWC_Mito_DNA",
    "Nuclei_Correlation_RWC_Mito_ER",
    "Nuclei_Correlation_RWC_Mito_RNA",
    "Nuclei_Correlation_RWC_RNA_AGP",
    "Nuclei_Correlation_RWC_RNA_DNA",
    "Nuclei_Correlation_RWC_RNA_ER",
    "Nuclei_Correlation_RWC_RNA_Mito",
    "Nuclei_Granularity_14_AGP",
    "Nuclei_Granularity_14_DNA",
    "Nuclei_Granularity_14_ER",
    "Nuclei_Granularity_14_Mito",
    "Nuclei_Granularity_14_RNA",
    "Nuclei_Granularity_15_AGP",
    "Nuclei_Granularity_15_DNA",
    "Nuclei_Granularity_15_ER",
    "Nuclei_Granularity_15_Mito",
    "Nuclei_Granularity_15_RNA",
    "Nuclei_Granularity_16_AGP",
    "Nuclei_Granularity_16_DNA",
    "Nuclei_Granularity_16_ER",
    "Nuclei_Granularity_16_Mito",
    "Nuclei_Granularity_16_RNA",
]
