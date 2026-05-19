#!/usr/bin/env python
"""Run Cell Painting validation sweep using phenocompass evaluation pipeline.

This script replicates the cp_sweep.sh pipeline that runs via Hydra/map-lib,
but uses the pure Python implementation in phenocompass. It sweeps over 18
hyperparameter combinations (2 normalizations x 3 PCA dims x 3 batch
corrections) and saves preprocessed embeddings as centered.parquet files.

Usage:
    python scripts/run_cp_validation_sweep.py

    # Custom data path:
    python scripts/run_cp_validation_sweep.py \
        --data-path /path/to/well_aggregated.parquet

    # Custom output directory:
    python scripts/run_cp_validation_sweep.py \
        --output-dir debugging/phenocompass_script/

    # Subset of sweep parameters:
    python scripts/run_cp_validation_sweep.py \
        --spread-fns mad \
        --pca-dims 128 \
        --batch-corrections global_sphere tvn none
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from phenocompass.evaluation.map_config import CP_VALIDATION_CONTROLS
from phenocompass.evaluation.map_pipelines import run_cp_validation_sweep

# Default paths (relative to repo root)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_FINAL_DATA = os.path.join(_REPO_ROOT, "final_data")

DEFAULT_DATA_PATH = os.path.join(
    _FINAL_DATA, "validation", "cell_painting_validation.cellprofiler_features.parquet"
)

DEFAULT_OUTPUT_DIR = os.path.join(_FINAL_DATA, "validation", "validation_cp_sweep")

DEFAULT_GT_DIR = os.path.join(_FINAL_DATA, "compound_annotations")

DEFAULT_GT_FILES = {
    "moa": "cp_validation_opentarget_annotations.pkl",
}


def main():
    parser = argparse.ArgumentParser(
        description="Run Cell Painting validation hyperparameter sweep",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_DATA_PATH,
        help="Path to well-aggregated Cell Painting parquet file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for centered.parquet files and sweep summary",
    )
    parser.add_argument(
        "--spread-fns",
        nargs="+",
        default=["mad", "iqr"],
        choices=["mad", "iqr"],
        help="Z-score spread functions to sweep",
    )
    parser.add_argument(
        "--pca-dims",
        nargs="+",
        type=int,
        default=[64, 128, 256],
        help="PCA component counts to sweep",
    )
    parser.add_argument(
        "--batch-corrections",
        nargs="+",
        default=["global_sphere", "tvn", "none"],
        choices=["global_sphere", "tvn", "none"],
        help="Batch correction methods to sweep",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for PCA reproducibility",
    )
    parser.add_argument(
        "--run-evaluations",
        action="store_true",
        help="Run reproducibility KS and multivariate biological recall after preprocessing",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=DEFAULT_GT_DIR,
        help="Directory containing ground truth files for evaluation",
    )
    parser.add_argument(
        "--gt-files", nargs="+",
        default=list(DEFAULT_GT_FILES.values()),
        help="Ground truth file names within gt-dir",
    )
    parser.add_argument(
        "--gt-labels", nargs="+",
        default=list(DEFAULT_GT_FILES.keys()),
        help="Labels for ground truth files (must match --gt-files length)",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}")
        sys.exit(1)

    # Load data
    print("=" * 60)
    print("Cell Painting Validation Sweep")
    print("=" * 60)
    print(f"Data: {args.data_path}")
    print(f"Output: {args.output_dir}")
    print()

    print("Loading data...")
    df = pd.read_parquet(args.data_path)
    print(f"Loaded {len(df)} wells, {len([c for c in df.columns if not c.startswith('Metadata_')])} features")

    # Build sweep parameters
    sweep_params = {
        "zscore_spread_fn": args.spread_fns,
        "pca_n_components": args.pca_dims,
        "batch_correction": args.batch_corrections,
    }

    n_combos = 1
    for v in sweep_params.values():
        n_combos *= len(v)
    print(f"Sweep grid: {n_combos} combinations")
    for k, v in sweep_params.items():
        print(f"  {k}: {v}")
    print()

    # Run sweep (preprocessing + optional evaluation)
    results = run_cp_validation_sweep(
        df=df,
        output_dir=args.output_dir,
        posnegcon_ids=CP_VALIDATION_CONTROLS["posnegcon"],
        negcon_ids=CP_VALIDATION_CONTROLS["negcon"],
        sweep_params=sweep_params,
        random_state=args.random_state,
        run_evaluations=args.run_evaluations,
        gt_dir=args.gt_dir,
        gt_files=dict(zip(args.gt_labels, args.gt_files)),
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    summary_cols = ["run_index", "run_name", "n_samples", "n_features", "status"]
    recall_cols = [c for c in results.columns if c.startswith("recall_at_95")]
    eval_cols = [c for c in ["ks_plate", "ks_perturbation"] if c in results.columns]
    eval_cols.extend(sorted(recall_cols))
    summary_cols.extend(eval_cols)
    print(results[summary_cols].to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
