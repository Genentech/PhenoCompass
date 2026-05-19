#!/bin/bash
# Run the Cell Painting validation sweep that produces inputs for Fig5
# (FOXO3A validation analysis). Activate the phenocompass environment
# before running this script.
#
# Usage:
#   bash scripts/run_cp_validation_sweep.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/scripts"

python "${SCRIPT_DIR}/run_cp_validation_sweep.py" \
    --data-path "${REPO_ROOT}/final_data/validation/cell_painting_validation.cellprofiler_features.parquet" \
    --output-dir "${REPO_ROOT}/final_data/validation/validation_cp_sweep" \
    --run-evaluations
