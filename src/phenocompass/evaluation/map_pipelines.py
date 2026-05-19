"""Pipeline runners for JUMP map building evaluation.

This module contains the preprocessing pipeline runner, evaluation pipeline
runner, and Cell Painting validation pipeline.
"""

import itertools
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from phenocompass.evaluation.map_config import (
    CP_VALIDATION_CONTROLS,
    CPValidationConfig,
    JUMPFields,
    PreprocessingConfig,
)
from phenocompass.evaluation.map_evaluation import (
    compute_per_term_metrics,
    knn_vote_accuracy,
    multivariate_metric_all,
    reproducibility_ks,
    univariate_self_consistency,
)
from phenocompass.evaluation.map_preprocessing import (
    apply_global_sphering,
    apply_pca,
    apply_tvn,
    clip_features,
    drop_low_var_columns,
    drop_nan,
    drop_way2019_blacklist,
    get_feature_columns,
    get_metadata_columns,
    inf_to_nan,
    platewise_zscore,
    zscore_by_controls,
)


# =============================================================================
# Pipeline Runner (JUMP data)
# =============================================================================


def run_preprocessing_pipeline(
    df: pd.DataFrame,
    config: PreprocessingConfig,
    fields: JUMPFields = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run complete preprocessing pipeline for JUMP data.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    config : PreprocessingConfig
        Preprocessing configuration.
    fields : JUMPFields, optional
        JUMP field names.

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Preprocessed DataFrame and feature columns.
    """
    if fields is None:
        fields = JUMPFields()

    metadata_cols = get_metadata_columns(df)
    feature_cols = get_feature_columns(df, metadata_cols)

    print(f"Starting with {len(df)} samples, {len(feature_cols)} features")

    # Drop NaN columns then rows
    df, feature_cols = drop_nan(df, feature_cols, config.column_max_nan_ratio)

    # Normalize
    if config.normalize_method == "platewise_median_mad":
        groupby = [c for c in config.normalize_groupby if c in df.columns]
        df = platewise_zscore(
            df,
            feature_columns=feature_cols,
            groupby=groupby,
            center_fn="median",
            spread_fn="mad",
            perturbation_col=fields.perturbation,
        )

    # Drop low variance columns
    df, feature_cols = drop_low_var_columns(
        df, feature_cols,
        metric_fn="std",
        min_value=config.drop_low_var_min,
        max_value=config.drop_low_var_max,
    )

    print(f"After preprocessing: {len(df)} samples, {len(feature_cols)} features")

    return df, feature_cols


def run_all_evaluations(
    df: pd.DataFrame,
    feature_columns: List[str],
    fields: JUMPFields = None,
    config: "EvalConfig" = None,
) -> Dict[str, pd.DataFrame]:
    """Run all evaluation metrics.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame.
    feature_columns : List[str]
        Feature columns.
    fields : JUMPFields, optional
        JUMP field names.
    config : EvalConfig, optional
        Evaluation configuration.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of evaluation results.
    """
    from phenocompass.evaluation.map_config import EvalConfig

    if fields is None:
        fields = JUMPFields()
    if config is None:
        config = EvalConfig()

    results = {}

    # Batch effect evaluation (kNN accuracy)
    print("Running batch effect evaluation...")
    class_cols = [c for c in config.batch_effect_class_columns if c in df.columns]
    if class_cols:
        results["batch_effect_knn"] = knn_vote_accuracy(
            df, feature_columns, class_cols,
            k_values=config.knn_k_values,
            metric=config.knn_metric,
        )

    # Reproducibility evaluation
    print("Running reproducibility evaluation...")
    results["reproducibility_ks"] = reproducibility_ks(
        df, feature_columns,
        perturbation_col=fields.perturbation,
        n_samples=config.reproducibility_n_samples,
    )

    # Univariate self-consistency
    print("Running univariate self-consistency...")
    results["univariate_consistency"] = univariate_self_consistency(
        df, feature_columns,
        perturbation_col=fields.perturbation,
        perturbation_type_col=fields.perturbation_type,
    )

    return results


# =============================================================================
# Utility Functions
# =============================================================================


def aggregate_by_perturbation(
    df: pd.DataFrame,
    feature_columns: List[str],
    perturbation_col: str,
) -> pd.DataFrame:
    """Median-aggregate features per perturbation.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (well-level).
    feature_columns : List[str]
        Feature columns to aggregate.
    perturbation_col : str
        Column containing perturbation IDs.

    Returns
    -------
    pd.DataFrame
        Perturbation-level DataFrame with median-aggregated features.
    """
    return df.groupby(perturbation_col, observed=True)[feature_columns].median().reset_index()


def load_gprobe_ground_truth(csv_path: str) -> Dict[str, List[str]]:
    """Load GProbe ground truth gene sets from CSV.

    Expects columns containing compound identifiers (e.g., "id", "number")
    and a MOA / class / term label column.
    Builds a dict mapping each MOA term to the list of compound IDs in that term.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with ground truth annotations.

    Returns
    -------
    Dict[str, List[str]]
        Mapping of MOA term names to lists of compound IDs.
    """
    gt_df = pd.read_csv(csv_path)

    # Auto-detect columns
    id_cols = [c for c in gt_df.columns if "number" in c.lower() or "id" in c.lower()]
    moa_cols = [c for c in gt_df.columns if "moa" in c.lower() or "class" in c.lower() or "term" in c.lower()]

    if not id_cols or not moa_cols:
        # Fallback: assume first two are IDs, third is MOA
        cols = gt_df.columns.tolist()
        if len(cols) >= 3:
            id_cols = cols[:2]
            moa_cols = [cols[2]]
        else:
            raise ValueError(
                f"Cannot auto-detect columns. Found: {cols}. "
                "Expected columns containing 'id' and 'MOA'/'class'/'term'."
            )

    moa_col = moa_cols[0]

    gt_sets: Dict[str, List[str]] = {}
    for moa, group in gt_df.groupby(moa_col):
        ids = set()
        for col in id_cols:
            ids.update(group[col].dropna().unique())
        gt_sets[str(moa)] = sorted(ids)

    return gt_sets


# =============================================================================
# Cell Painting Validation Pipeline
# =============================================================================


def run_cp_validation_pipeline(
    df: pd.DataFrame,
    zscore_spread_fn: str = "mad",
    pca_n_components: int = 128,
    batch_correction: str = "tvn",
    posnegcon_ids: Optional[List[str]] = None,
    negcon_ids: Optional[List[str]] = None,
    perturbation_col: str = "Metadata_COMPOUND_ID",
    plate_col: str = "Metadata_Plate",
    output_path: Optional[str] = None,
    random_state: int = 42,
    sphere_reg_param: float = 0.001,
    tvn_reg_param: float = 1.0,
    tvn_oas: bool = False,
    tvn_recolor: bool = False,
    column_max_nan_ratio: float = 0.01,
    drop_low_std_min: float = 0.1,
    drop_low_std_max: float = 5.0,
    clip_value: float = 5.0,
) -> Tuple[pd.DataFrame, List[str]]:
    """Run the Cell Painting validation preprocessing pipeline.

    Replicates the map-lib cp_sweep.sh pipeline exactly:

    1. prefilter_features: drop Way2019 blacklist
    2. pre_normalize: inf->NaN, drop NaN columns/rows
    3. normalize: platewise z-score (median center, mad/iqr spread, posnegcon subset)
    4. postfilter_features: drop features with std outside [0.1, 5.0]
    5. post_normalize: inf->NaN, drop NaN columns/rows, clip [-5, 5]
    6. dim_red: PCA
    7. batch_correction: global_sphere / sphere+TVN / none
    8. center: z-score by negcon (mean center + std spread)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (well-level Cell Painting features).
    zscore_spread_fn : str
        Spread function for platewise z-score ("mad" or "iqr").
    pca_n_components : int
        Number of PCA components (64, 128, or 256).
    batch_correction : str
        Batch correction method ("global_sphere", "tvn", or "none").
    posnegcon_ids : List[str], optional
        Positive + negative control IDs for z-score normalization.
        Defaults to CP_VALIDATION_CONTROLS["posnegcon"].
    negcon_ids : List[str], optional
        Negative control IDs for centering.
        Defaults to CP_VALIDATION_CONTROLS["negcon"].
    perturbation_col : str
        Column containing perturbation IDs.
    plate_col : str
        Column containing plate identifiers.
    output_path : str, optional
        If provided, save centered.parquet to this path.
    random_state : int
        Random state for PCA reproducibility.
    sphere_reg_param : float
        Regularization for global sphering (default 0.001).
    tvn_reg_param : float
        Regularization for TVN per-plate whitening (default 1.0).
    tvn_oas : bool
        Whether to use OAS shrinkage in TVN (default False).
    tvn_recolor : bool
        Whether to recolor in TVN (default False).
    column_max_nan_ratio : float
        Max NaN ratio for dropping columns (default 0.01).
    drop_low_std_min : float
        Min std threshold for feature retention (default 0.1).
    drop_low_std_max : float
        Max std threshold for feature retention (default 5.0).
    clip_value : float
        Clipping bound (default 5.0, clips to [-5, 5]).

    Returns
    -------
    Tuple[pd.DataFrame, List[str]]
        Processed DataFrame and list of feature column names.
    """
    if posnegcon_ids is None:
        posnegcon_ids = CP_VALIDATION_CONTROLS["posnegcon"]
    if negcon_ids is None:
        negcon_ids = CP_VALIDATION_CONTROLS["negcon"]

    groupby = [plate_col]
    feature_columns = get_feature_columns(df)

    print(f"Starting CP validation pipeline: {len(df)} samples, {len(feature_columns)} features")
    print(f"  zscore_spread_fn={zscore_spread_fn}, pca={pca_n_components}, batch_correction={batch_correction}")

    # Step 1: prefilter_features — drop Way2019 blacklist
    df, feature_columns = drop_way2019_blacklist(df, feature_columns)

    # Step 2: pre_normalize — inf->NaN, drop NaN columns (>1%), drop NaN rows
    df = inf_to_nan(df, feature_columns)
    df, feature_columns = drop_nan(df, feature_columns, column_max_nan_ratio)

    # Step 3: normalize — platewise z-score using posnegcon controls
    df = platewise_zscore(
        df,
        feature_columns=feature_columns,
        groupby=groupby,
        center_fn="median",
        spread_fn=zscore_spread_fn,
        control_ids=posnegcon_ids,
        perturbation_col=perturbation_col,
    )

    # Step 4: postfilter_features — drop features with std outside [min, max]
    df, feature_columns = drop_low_var_columns(
        df, feature_columns,
        metric_fn="std",
        min_value=drop_low_std_min,
        max_value=drop_low_std_max,
    )

    # Step 5: post_normalize — inf->NaN, drop NaN columns/rows, clip
    df = inf_to_nan(df, feature_columns)
    df, feature_columns = drop_nan(df, feature_columns, column_max_nan_ratio)
    df = clip_features(df, feature_columns, lower=-clip_value, upper=clip_value)

    # Step 6: dim_red — PCA
    df, feature_columns = apply_pca(
        df, feature_columns, pca_n_components,
        random_state=random_state, svd_solver="auto",
    )

    # Step 7: batch_correction
    if batch_correction == "global_sphere":
        df = apply_global_sphering(
            df, feature_columns,
            reg_param=sphere_reg_param,
            oas=False, rotate=True,
        )
    elif batch_correction == "tvn":
        # Map-lib applies global sphere BEFORE per-plate TVN
        df = apply_global_sphering(
            df, feature_columns,
            reg_param=sphere_reg_param,
            oas=False, rotate=True,
        )
        df = apply_tvn(
            df, feature_columns,
            groupby=groupby,
            reg_param=tvn_reg_param,
            oas=tvn_oas,
            recolor=tvn_recolor,
            rotate=True,
        )
    elif batch_correction == "none":
        pass
    else:
        raise ValueError(f"Unknown batch_correction: {batch_correction}")

    # Step 8: center — z-score by negcon (mean + std)
    df = zscore_by_controls(
        df, feature_columns,
        control_ids=negcon_ids,
        perturbation_col=perturbation_col,
        center_fn="mean",
        spread_fn="std",
    )

    # Optionally save centered.parquet
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"Saved: {output_path}")

    print(f"CP validation pipeline complete: {len(df)} samples, {len(feature_columns)} features")

    return df, feature_columns


def run_cp_validation_sweep(
    df: pd.DataFrame,
    output_dir: str,
    posnegcon_ids: Optional[List[str]] = None,
    negcon_ids: Optional[List[str]] = None,
    sweep_params: Optional[Dict[str, List[Any]]] = None,
    random_state: int = 42,
    perturbation_col: str = "Metadata_COMPOUND_ID",
    plate_col: str = "Metadata_Plate",
    run_evaluations: bool = False,
    gt_dir: Optional[str] = None,
    gt_files: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Run a hyperparameter sweep over the CP validation pipeline.

    Default sweep: 2 normalizations x 3 PCA dims x 3 batch corrections = 18 runs.
    Each run saves a centered.parquet file. Optionally runs reproducibility KS
    and multivariate biological recall evaluations.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (well-level Cell Painting features).
    output_dir : str
        Directory to save sweep outputs. Each run writes to
        ``{output_dir}/{run_name}/centered.parquet``.
    posnegcon_ids : List[str], optional
        Positive + negative control IDs. Defaults to CP_VALIDATION_CONTROLS["posnegcon"].
    negcon_ids : List[str], optional
        Negative control IDs. Defaults to CP_VALIDATION_CONTROLS["negcon"].
    sweep_params : Dict[str, List[Any]], optional
        Hyperparameter grid. Defaults to standard 18-combination sweep.
    random_state : int
        Random state for PCA reproducibility.
    perturbation_col : str
        Column containing perturbation IDs.
    plate_col : str
        Column containing plate identifiers.
    run_evaluations : bool
        If True, run reproducibility KS and multivariate biological recall
        after each preprocessing run. Requires gt_dir and gt_files.
    gt_dir : str, optional
        Directory containing ground truth files for multivariate evaluation.
    gt_files : Dict[str, str], optional
        Mapping of label to ground truth file name within gt_dir
        (e.g., ``{"moa": "opentarget_moa.graph.pkl", "target": "opentarget_target.graph.pkl"}``).

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with columns: run_index, run_name, zscore_spread_fn,
        pca_n_components, batch_correction, output_path, n_samples, n_features,
        status. When run_evaluations=True, also includes ks_plate, ks_perturbation,
        and recall_at_95_<label> for each ground truth dataset.
    """
    if posnegcon_ids is None:
        posnegcon_ids = CP_VALIDATION_CONTROLS["posnegcon"]
    if negcon_ids is None:
        negcon_ids = CP_VALIDATION_CONTROLS["negcon"]

    if sweep_params is None:
        sweep_params = {
            "zscore_spread_fn": ["mad", "iqr"],
            "pca_n_components": [64, 128, 256],
            "batch_correction": ["global_sphere", "tvn", "none"],
        }

    # Build parameter grid
    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    grid = list(itertools.product(*param_values))

    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    for run_idx, combo in enumerate(grid):
        params = dict(zip(param_names, combo))

        # Create descriptive run name matching map-lib convention
        run_name = (
            f"norm_{params.get('zscore_spread_fn', 'mad')}"
            f"_pca{params.get('pca_n_components', 128)}"
            f"_bc_{params.get('batch_correction', 'none')}"
        )

        run_output_dir = os.path.join(output_dir, run_name)
        output_path = os.path.join(run_output_dir, "centered.parquet")

        print(f"\n{'='*60}")
        print(f"Run {run_idx}/{len(grid)-1}: {run_name}")
        print(f"  {params}")
        print(f"{'='*60}")

        row = {
            "run_index": run_idx,
            "run_name": run_name,
            **params,
            "output_path": output_path,
        }

        try:
            df_processed, feat_cols = run_cp_validation_pipeline(
                df,
                zscore_spread_fn=params.get("zscore_spread_fn", "mad"),
                pca_n_components=params.get("pca_n_components", 128),
                batch_correction=params.get("batch_correction", "none"),
                posnegcon_ids=posnegcon_ids,
                negcon_ids=negcon_ids,
                perturbation_col=perturbation_col,
                plate_col=plate_col,
                output_path=output_path,
                random_state=random_state,
            )

            row["n_samples"] = len(df_processed)
            row["n_features"] = len(feat_cols)
            row["status"] = "success"

            # Run evaluations if requested
            if run_evaluations and gt_dir is not None and gt_files is not None:
                try:
                    print(f"  Running evaluations...")

                    # Reproducibility KS (well-level)
                    ks_results = reproducibility_ks(
                        df_processed,
                        variables=[plate_col, perturbation_col],
                        n_samples=10000,
                        columns=feat_cols,
                        seed=42,
                    )
                    ks_path = os.path.join(run_output_dir, "reproducibility_ks.csv")
                    ks_results.to_csv(ks_path, index=False)
                    print(f"  Saved: {ks_path}")

                    # Extract KS statistics for summary
                    for _, ks_row in ks_results.iterrows():
                        var_name = ks_row["variable"]
                        short_name = var_name.replace("Metadata_", "").lower()
                        row[f"ks_{short_name}"] = ks_row["statistic"]

                    # Multivariate biological recall (perturbation-level)
                    df_agg = df_processed.groupby(perturbation_col, observed=True)[feat_cols].mean().reset_index()

                    # Build label lookup from gt_files dict
                    file_to_label = {
                        v.replace(".pkl", "").replace(".csv", ""): k
                        for k, v in gt_files.items()
                    }

                    multi_results = multivariate_metric_all(
                        df_agg,
                        pert_field=perturbation_col,
                        data_dir=gt_dir,
                        eval_datasets=list(gt_files.values()),
                        columns=feat_cols,
                    )

                    # Save and extract recall@95 per GT dataset
                    for dataset_key, perf_df in multi_results.items():
                        if dataset_key.endswith("__per_term"):
                            base_key = dataset_key.replace("__per_term", "")
                            label = file_to_label.get(base_key, base_key)
                            per_term_path = os.path.join(
                                run_output_dir,
                                f"per_term_metrics.{label}.csv",
                            )
                            perf_df.to_csv(per_term_path, index=False)
                            print(f"  Saved: {per_term_path}")
                            continue

                        label = file_to_label.get(dataset_key, dataset_key)
                        multi_path = os.path.join(
                            run_output_dir,
                            f"multivariate_metrics.{label}.csv",
                        )
                        perf_df.to_csv(multi_path, index=False)
                        print(f"  Saved: {multi_path}")

                        # Extract recall@95 (global, one-sided)
                        row_95 = perf_df[perf_df["percentile"] == 95]
                        if len(row_95) > 0:
                            row[f"recall_at_95_{label}"] = row_95["global_recall_one-sided"].values[0]

                    print(f"  Evaluations complete.")

                except Exception as eval_e:
                    print(f"  Evaluation error: {eval_e}")
                    row["eval_error"] = str(eval_e)

        except Exception as e:
            print(f"Error in run {run_idx} ({run_name}): {e}")
            row["n_samples"] = np.nan
            row["n_features"] = np.nan
            row["status"] = f"error: {e}"

        all_results.append(row)

    results_df = pd.DataFrame(all_results)

    # Save sweep summary
    summary_path = os.path.join(output_dir, "sweep_summary.parquet")
    results_df.to_parquet(summary_path, index=False)
    print(f"\nSweep complete: {len(results_df)} runs. Summary saved to {summary_path}")

    return results_df
