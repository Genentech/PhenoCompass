"""Preprocessing utilities for Cell Painting validation pipeline.

This module replicates the preprocessing pipeline from map-lib's cp_sweep.sh,
providing functions for normalization, batch correction (TVN/sphering),
dimensionality reduction, and feature filtering.

Reference: debugging/maplib_cp_sweep_reference.md
"""

import math
from typing import List, Optional, Tuple

import numba
import numpy as np
import pandas as pd
from scipy.stats import median_abs_deviation
from sklearn.covariance import OAS
from tqdm import tqdm


# =============================================================================
# Numba-optimized Distance Functions
# =============================================================================


@numba.njit
def cosine_similarity_numba(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between arrays x and y.

    Data points are the last dimension of x and y.

    Parameters
    ----------
    x : np.ndarray
        First array of shape (..., n_features).
    y : np.ndarray
        Second array of shape (..., n_features).

    Returns
    -------
    np.ndarray
        Cosine similarity values.
    """
    return np.sum(x * y, axis=-1) / (
        np.sqrt(np.sum(x**2, axis=-1)) * np.sqrt(np.sum(y**2, axis=-1))
    )


@numba.njit
def euclidean_distance_numba(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate squared Euclidean distance between arrays x and y.

    Data points are the last dimension of x and y.

    Parameters
    ----------
    x : np.ndarray
        First array of shape (..., n_features).
    y : np.ndarray
        Second array of shape (..., n_features).

    Returns
    -------
    np.ndarray
        Squared Euclidean distance values.
    """
    return np.sum((x - y) ** 2, axis=-1)


def get_random_index_pairs(num_perts: int, sample_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random pairs of indices, excluding the diagonal.

    Parameters
    ----------
    num_perts : int
        Number of perturbations (matrix size).
    sample_size : int
        Number of pairs to sample.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Two arrays of indices representing pairs.
    """
    idx = np.random.choice(num_perts * (num_perts - 1), sample_size, replace=True)
    idx1, idx2 = np.unravel_index(idx, (num_perts, num_perts - 1))
    idx2 = idx2 + (idx2 >= idx1).astype(idx2.dtype)
    return idx1, idx2


# =============================================================================
# Data Utilities
# =============================================================================


def get_negcon_perturbation_ids(
    df: pd.DataFrame,
    perturbation_type_col: str = "Metadata_PerturbationType",
    perturbation_col: str = "Metadata_JCP2022",
) -> List[str]:
    """Get negative control perturbation IDs present in the DataFrame.

    Used by JUMP evaluation metrics to exclude controls from analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    perturbation_type_col : str
        Column containing perturbation type.
    perturbation_col : str
        Column containing perturbation ID.

    Returns
    -------
    List[str]
        List of negative control perturbation IDs.
    """
    from phenocompass.evaluation.map_config import NEGATIVE_CONTROLS

    negcon_ids = []

    if perturbation_type_col in df.columns:
        for pert_type, controls in NEGATIVE_CONTROLS.items():
            mask = df[perturbation_type_col].str.lower() == pert_type.lower()
            present_ids = df.loc[mask, perturbation_col].unique()
            negcon_ids.extend([pid for pid in controls if pid in present_ids])
    else:
        all_controls = [c for controls in NEGATIVE_CONTROLS.values() for c in controls]
        present_ids = df[perturbation_col].unique()
        negcon_ids = [pid for pid in all_controls if pid in present_ids]

    return list(set(negcon_ids))


def get_metadata_columns(df: pd.DataFrame, prefix: str = "Metadata_") -> List[str]:
    """Get all metadata columns from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    prefix : str, default "Metadata_"
        Prefix for metadata columns.

    Returns
    -------
    List[str]
        List of metadata column names.
    """
    return [col for col in df.columns if col.startswith(prefix)]


def get_feature_columns(
    df: pd.DataFrame,
    metadata_columns: Optional[List[str]] = None,
    prefix: str = "Metadata_",
) -> List[str]:
    """Get all feature (non-metadata) columns from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    metadata_columns : List[str], optional
        Explicit list of metadata columns to exclude.
    prefix : str, default "Metadata_"
        Prefix for metadata columns (used if metadata_columns not provided).

    Returns
    -------
    List[str]
        List of feature column names.
    """
    if metadata_columns is None:
        metadata_columns = get_metadata_columns(df, prefix=prefix)
    return [col for col in df.columns if col not in metadata_columns]


# =============================================================================
# Preprocessing: Feature Filtering
# =============================================================================


def drop_way2019_blacklist(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop Way 2019 blacklisted features from CellPaint data.

    Removes 55 Nuclei_Correlation_Manders_*, Nuclei_Correlation_RWC_*, and
    Nuclei_Granularity_14-16_* features identified by Way et al. 2019/2021.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_columns : List[str]
        Current feature column names.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with blacklisted columns removed, and updated feature list.
    """
    from phenocompass.evaluation.map_config import WAY2019_FEATURE_BLACKLIST

    to_drop = [c for c in WAY2019_FEATURE_BLACKLIST if c in feature_columns]
    remaining = [c for c in feature_columns if c not in set(to_drop)]
    if to_drop:
        df = df.drop(columns=to_drop)
        print(f"Way 2019 blacklist: dropped {len(to_drop)} features")
    return df, remaining


def drop_low_var_columns(
    df: pd.DataFrame,
    feature_columns: List[str],
    metric_fn: str = "std",
    min_value: float = 0.1,
    max_value: float = 5.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns with spread metric outside specified range.

    Matches map-lib's drop_low_var_columns behavior: drops features where
    the specified metric is < min_value or > max_value.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_columns : List[str]
        Feature columns to check.
    metric_fn : str
        Spread metric to use: "std", "var", "mad", or "iqr".
    min_value : float
        Minimum threshold (columns below this are dropped).
    max_value : float
        Maximum threshold (columns above this are dropped).

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Filtered DataFrame and list of remaining feature columns.
    """
    feat_df = df[feature_columns]

    if metric_fn == "std":
        stats = feat_df.std()
    elif metric_fn == "var":
        stats = feat_df.var()
    elif metric_fn == "mad":
        stats = feat_df.apply(lambda x: median_abs_deviation(x, nan_policy="omit", scale=1 / 1.4826))
    elif metric_fn == "iqr":
        stats = feat_df.quantile(0.75) - feat_df.quantile(0.25)
    else:
        raise ValueError(f"Unknown metric_fn: {metric_fn}")

    mask = (stats >= min_value) & (stats <= max_value)
    cols_to_keep = stats[mask].index.tolist()
    cols_to_drop = [c for c in feature_columns if c not in cols_to_keep]

    if cols_to_drop:
        print(f"Dropped {len(cols_to_drop)} columns with {metric_fn} outside [{min_value}, {max_value}]")
        df = df.drop(columns=cols_to_drop)

    return df, cols_to_keep


# =============================================================================
# Preprocessing: NaN/Inf Handling
# =============================================================================


def inf_to_nan(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:
    """Replace inf/-inf values with NaN in feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_columns : List[str]
        Feature columns to process.

    Returns
    -------
    pd.DataFrame
        DataFrame with inf values replaced by NaN.
    """
    df = df.copy()
    df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan)
    return df


def drop_nan(
    df: pd.DataFrame,
    feature_columns: List[str],
    column_max_nan_ratio: float = 0.01,
) -> Tuple[pd.DataFrame, List[str]]:
    """Drop columns with high NaN ratio, then drop rows with remaining NaN.

    Matches map-lib's drop_nan behavior: when 0 < column_max_nan_ratio < 1,
    first drops columns where NaN ratio >= threshold, then drops any rows
    that still contain NaN values in the remaining feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_columns : List[str]
        Feature columns to check.
    column_max_nan_ratio : float
        Maximum allowed ratio of NaN values per column.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Filtered DataFrame and updated list of feature columns.
    """
    # Step 1: Drop columns with NaN ratio >= threshold
    nan_ratio = df[feature_columns].isna().mean()
    cols_to_keep = nan_ratio[nan_ratio < column_max_nan_ratio].index.tolist()
    cols_to_drop = [c for c in feature_columns if c not in cols_to_keep]

    if cols_to_drop:
        print(f"Dropped {len(cols_to_drop)} columns with NaN ratio >= {column_max_nan_ratio}")
        df = df.drop(columns=cols_to_drop)

    # Step 2: Drop rows with any remaining NaN in feature columns
    mask = df[cols_to_keep].isna().any(axis=1)
    n_dropped = mask.sum()
    if n_dropped > 0:
        print(f"Dropped {n_dropped} rows with NaN values")
        df = df.loc[~mask].copy()

    return df, cols_to_keep


# =============================================================================
# Preprocessing: Clipping
# =============================================================================


def clip_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    lower: float = -5.0,
    upper: float = 5.0,
) -> pd.DataFrame:
    """Clip feature values to a specified range.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_columns : List[str]
        Feature columns to clip.
    lower : float
        Lower bound.
    upper : float
        Upper bound.

    Returns
    -------
    pd.DataFrame
        DataFrame with clipped feature values.
    """
    df = df.copy()
    df[feature_columns] = df[feature_columns].clip(lower=lower, upper=upper)
    return df


# =============================================================================
# Normalization: Platewise Z-score
# =============================================================================


def platewise_zscore(
    df: pd.DataFrame,
    feature_columns: List[str],
    groupby: List[str],
    center_fn: str = "median",
    spread_fn: str = "mad",
    control_ids: Optional[List[str]] = None,
    perturbation_col: str = "Metadata_Perturbation",
) -> pd.DataFrame:
    """Apply plate-wise z-score normalization.

    Matches map-lib's zscore_deprecated with groupby and subset parameters.
    Statistics are computed from the control subset within each group.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_columns : List[str]
        Feature columns to normalize.
    groupby : List[str]
        Columns to group by (typically ["Metadata_Plate"]).
    center_fn : str, default "median"
        Centering function ("mean" or "median").
    spread_fn : str, default "mad"
        Spread function ("std", "mad", or "iqr").
    control_ids : List[str], optional
        Perturbation IDs to use for computing statistics.
        If None, all rows are used.
    perturbation_col : str
        Column containing perturbation IDs.

    Returns
    -------
    pd.DataFrame
        Normalized DataFrame.
    """
    df = df.copy()

    # Resolve center function
    if center_fn == "median":
        _center = lambda x: np.nanmedian(x, axis=0)
    elif center_fn == "mean":
        _center = lambda x: np.nanmean(x, axis=0)
    else:
        raise ValueError(f"Unknown center_fn: {center_fn}")

    # Resolve spread function
    if spread_fn == "mad":
        _spread = lambda x: median_abs_deviation(x, axis=0, nan_policy="omit", scale=1 / 1.4826)
    elif spread_fn == "std":
        _spread = lambda x: np.nanstd(x, axis=0, ddof=1)
    elif spread_fn == "iqr":
        _spread = lambda x: np.nanpercentile(x, 75, axis=0) - np.nanpercentile(x, 25, axis=0)
    else:
        raise ValueError(f"Unknown spread_fn: {spread_fn}")

    # Normalize per group
    for _, group_idx in df.groupby(groupby).groups.items():
        group_df = df.loc[group_idx]

        if control_ids is not None:
            control_mask = group_df[perturbation_col].isin(control_ids)
            control_df = group_df.loc[control_mask]
        else:
            control_df = group_df

        if len(control_df) == 0:
            continue

        center = _center(control_df[feature_columns].values)
        spread = _spread(control_df[feature_columns].values)

        # Match map-lib behavior: do NOT replace spread==0.
        # Division by zero produces inf, caught by post_normalize (inf_to_nan + drop_nan).
        df.loc[group_idx, feature_columns] = (
            df.loc[group_idx, feature_columns].values - center
        ) / spread

    return df


# =============================================================================
# Normalization: Z-score by Controls (Centering Step)
# =============================================================================


def zscore_by_controls(
    df: pd.DataFrame,
    feature_columns: List[str],
    control_ids: List[str],
    perturbation_col: str = "Metadata_Perturbation",
    center_fn: str = "mean",
    spread_fn: str = "std",
) -> pd.DataFrame:
    """Global z-score normalization using control compounds.

    Matches map-lib's zscore_deprecated with subset=negcon and no groupby.
    Computes center and spread from control rows, applies to all rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_columns : List[str]
        Feature columns to normalize.
    control_ids : List[str]
        Control compound IDs to compute statistics from.
    perturbation_col : str
        Column containing perturbation IDs.
    center_fn : str, default "mean"
        Centering function ("mean" or "median").
    spread_fn : str, default "std"
        Spread function ("std", "mad", or "iqr").

    Returns
    -------
    pd.DataFrame
        Z-score normalized DataFrame.
    """
    df = df.copy()
    control_mask = df[perturbation_col].isin(control_ids)
    control_vals = df.loc[control_mask, feature_columns].values

    # Compute center
    if center_fn == "mean":
        center = np.nanmean(control_vals, axis=0)
    elif center_fn == "median":
        center = np.nanmedian(control_vals, axis=0)
    else:
        raise ValueError(f"Unknown center_fn: {center_fn}")

    # Compute spread
    if spread_fn == "std":
        spread = np.nanstd(control_vals, axis=0, ddof=1)
    elif spread_fn == "mad":
        spread = median_abs_deviation(control_vals, axis=0, nan_policy="omit", scale=1 / 1.4826)
    elif spread_fn == "iqr":
        spread = np.nanpercentile(control_vals, 75, axis=0) - np.nanpercentile(control_vals, 25, axis=0)
    else:
        raise ValueError(f"Unknown spread_fn: {spread_fn}")

    # Match map-lib: no special handling for spread==0
    df[feature_columns] = (df[feature_columns].values - center) / spread

    return df


# =============================================================================
# Dimensionality Reduction: PCA
# =============================================================================


def apply_pca(
    df: pd.DataFrame,
    feature_columns: List[str],
    n_components: int,
    random_state: int = 42,
    svd_solver: str = "auto",
) -> Tuple[pd.DataFrame, List[str]]:
    """Apply PCA dimensionality reduction to feature columns.

    Fits PCA on all data (no train/test split), matching map-lib behavior.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_columns : List[str]
        Feature columns to transform.
    n_components : int
        Number of PCA components.
    random_state : int
        Random state for reproducibility.
    svd_solver : str
        SVD solver to use (default "auto" matches map-lib).

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        DataFrame with PCA features replacing originals, and new column names.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components, svd_solver=svd_solver, random_state=random_state)
    X = df[feature_columns].values
    X_pca = pca.fit_transform(X)

    new_cols = [f"PC_{i}" for i in range(n_components)]
    meta_cols = [c for c in df.columns if c not in feature_columns]

    df_meta = df[meta_cols].reset_index(drop=True)
    df_pca = pd.DataFrame(X_pca, columns=new_cols)
    df_out = pd.concat([df_meta, df_pca], axis=1)

    print(f"PCA: {len(feature_columns)} features -> {n_components} components "
          f"({pca.explained_variance_ratio_.sum():.2%} variance explained)")

    return df_out, new_cols


# =============================================================================
# Batch Correction: Covariance Estimators and Sphering Transform
# =============================================================================


def empirical_covariance(X: np.ndarray) -> np.ndarray:
    """Compute empirical covariance matrix with Bessel's correction.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Covariance matrix (n_features, n_features).
    """
    X_centered = X - X.mean(axis=0)
    return (X_centered.T @ X_centered) / (X.shape[0] - 1)


def oas_covariance(X: np.ndarray) -> np.ndarray:
    """Compute Oracle Approximating Shrinkage covariance estimate.

    Parameters
    ----------
    X : np.ndarray
        Data matrix (n_samples, n_features).

    Returns
    -------
    np.ndarray
        Shrinkage covariance matrix (n_features, n_features).
    """
    oas = OAS()
    oas.fit(X)
    return oas.covariance_


def whitening_transform(
    cov: np.ndarray,
    reg_param: float = 1e-6,
    rotate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute whitening transform from covariance matrix.

    Parameters
    ----------
    cov : np.ndarray
        Covariance matrix.
    reg_param : float
        Regularization parameter for numerical stability.
    rotate : bool
        If True, use ZCA whitening (W = V @ D^(-1/2) @ V^T) which preserves
        the original data orientation. If False, use PCA whitening
        (W = V @ D^(-1/2)).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (W, W_inv) whitening transform and its inverse.
    """
    # Add regularization
    cov_reg = cov + reg_param * np.eye(cov.shape[0])

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_reg)

    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues))
    D_sqrt = np.diag(np.sqrt(eigenvalues))

    if rotate:
        # ZCA whitening: W = V @ D^(-1/2) @ V^T
        W = eigenvectors @ D_inv_sqrt @ eigenvectors.T
        W_inv = eigenvectors @ D_sqrt @ eigenvectors.T
    else:
        # PCA whitening: W = V @ D^(-1/2)
        W = eigenvectors @ D_inv_sqrt
        W_inv = eigenvectors @ D_sqrt

    return W, W_inv


class SpheringTransform:
    """Whitening/sphering transform using empirical or OAS covariance estimation.

    This implements the sphering step used in TVN batch correction and
    global sphering. Default rotate=True matches map-lib behavior (ZCA whitening).
    """

    def __init__(
        self,
        controls: np.ndarray,
        reg_param: float = 1e-6,
        oas: bool = False,
        rotate: bool = True,
    ):
        """Initialize sphering transform from control samples.

        Parameters
        ----------
        controls : np.ndarray
            Control sample embeddings (n_samples, n_features).
        reg_param : float
            Regularization parameter.
        oas : bool
            If True, use OAS shrinkage estimator.
        rotate : bool
            If True, use ZCA whitening (default, matches map-lib).
        """
        self.reg_param = reg_param
        self.oas = oas

        # Center using control mean
        self.mean = controls.mean(axis=0)

        # Compute covariance
        if oas:
            self.cov = oas_covariance(controls)
        else:
            self.cov = empirical_covariance(controls)

        # Compute whitening transform and its inverse
        self.W, self.W_inv = whitening_transform(self.cov, reg_param, rotate=rotate)

    def normalize(self, X: np.ndarray) -> np.ndarray:
        """Apply whitening transform.

        Parameters
        ----------
        X : np.ndarray
            Data matrix (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Whitened data.
        """
        X_centered = X - self.mean
        return X_centered @ self.W

    def recolor(self, X: np.ndarray, target_transform: "SpheringTransform") -> np.ndarray:
        """Apply recoloring transform using target distribution.

        Parameters
        ----------
        X : np.ndarray
            Whitened data (n_samples, n_features).
        target_transform : SpheringTransform
            Target distribution transform.

        Returns
        -------
        np.ndarray
            Recolored data.
        """
        return X @ target_transform.W_inv + target_transform.mean


# =============================================================================
# Batch Correction: Global Sphering
# =============================================================================


def apply_global_sphering(
    df: pd.DataFrame,
    feature_columns: List[str],
    reg_param: float = 0.001,
    oas: bool = False,
    rotate: bool = True,
    subset_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Apply global sphering (whitening) transformation.

    By default uses ALL rows for covariance estimation, matching
    map-lib's global_sphere config (subset=None, condition_column=None).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_columns : List[str]
        Feature columns to transform.
    reg_param : float
        Regularization parameter.
    oas : bool
        If True, use OAS shrinkage estimator.
    rotate : bool
        If True, use ZCA whitening (default matches map-lib).
    subset_mask : np.ndarray, optional
        Boolean mask selecting rows to use for covariance estimation.
        If None, all rows are used.

    Returns
    -------
    pd.DataFrame
        Sphered DataFrame.
    """
    df = df.copy()
    X = df[feature_columns].values

    if subset_mask is not None:
        controls = X[subset_mask]
    else:
        controls = X

    transform = SpheringTransform(controls, reg_param=reg_param, oas=oas, rotate=rotate)
    df[feature_columns] = transform.normalize(X)
    return df


# =============================================================================
# Batch Correction: TVN (Typical Variation Normalization)
# =============================================================================


def apply_tvn(
    df: pd.DataFrame,
    feature_columns: List[str],
    groupby: List[str],
    reg_param: float = 1.0,
    oas: bool = False,
    recolor: bool = False,
    rotate: bool = True,
    subset_mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Apply TVN (Typical Variation Normalization) batch correction.

    Matches map-lib's batch_correct_tvn: per-group whitening with optional
    recoloring to global distribution.

    With recolor=False (cp_sweep.sh default), this performs per-plate
    whitening only. With recolor=True, whitened data is recolored to
    match the global control distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_columns : List[str]
        Feature columns to transform.
    groupby : List[str]
        Columns defining batches (typically ["Metadata_Plate"]).
    reg_param : float
        Regularization parameter for per-batch covariance.
    oas : bool
        If True, use OAS shrinkage estimator.
    recolor : bool
        If True, recolor to global distribution after whitening.
    rotate : bool
        If True, use ZCA whitening (default matches map-lib).
    subset_mask : np.ndarray, optional
        Boolean mask selecting rows to use for covariance estimation.
        If None, all rows per group are used (matches map-lib query=None).

    Returns
    -------
    pd.DataFrame
        Batch-corrected DataFrame.
    """
    df = df.copy()

    # Compute global transform for recoloring (if needed)
    if recolor:
        if subset_mask is not None:
            global_controls = df.loc[subset_mask, feature_columns].values
        else:
            global_controls = df[feature_columns].values
        global_transform = SpheringTransform(
            global_controls, reg_param=reg_param, oas=oas, rotate=rotate
        )

    # Process each batch
    for group_key, group_idx in tqdm(
        df.groupby(groupby).groups.items(),
        desc="TVN batch correction",
    ):
        group_df = df.loc[group_idx]

        if subset_mask is not None:
            batch_subset = subset_mask[group_idx] if isinstance(subset_mask, pd.Series) else \
                np.isin(group_idx, df.index[subset_mask])
            batch_controls = group_df.loc[batch_subset, feature_columns].values
        else:
            batch_controls = group_df[feature_columns].values

        if len(batch_controls) < 2:
            print(f"Warning: Batch {group_key} has < 2 samples for covariance, skipping")
            continue

        # Compute batch transform
        batch_transform = SpheringTransform(
            batch_controls, reg_param=reg_param, oas=oas, rotate=rotate
        )

        # Whiten batch data
        batch_data = group_df[feature_columns].values
        whitened = batch_transform.normalize(batch_data)

        if recolor:
            # Recolor using global distribution
            corrected = batch_transform.recolor(whitened, global_transform)
        else:
            corrected = whitened

        df.loc[group_idx, feature_columns] = corrected

    return df
