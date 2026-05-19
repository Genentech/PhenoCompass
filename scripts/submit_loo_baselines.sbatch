#!/bin/bash
#SBATCH --job-name=loo_baselines
#SBATCH --array=0-287
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/loo_baselines/output_%a.out
#SBATCH --error=logs/loo_baselines/errors_%a.err
#
# Submit all 288 LOO baseline worker jobs in parallel via SLURM.
#
# Array mapping: task_id = config_idx * 36 + seed * 6 + moa_idx
#   8 configs x 6 seeds x 6 MoAs = 288 tasks
#
# Submit from the repository root. Activate the phenocompass environment
# before submitting (conda/venv activation is intentionally left to the
# caller). Output partials are consumed by
# scripts/preprocess_loo_comparisons.py.
#
# Usage:
#   cd <repo-root>
#   mkdir -p logs/loo_baselines
#   sbatch scripts/submit_loo_baselines.sbatch
set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/scripts"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/notebooks:${PYTHONPATH:-}"

# ── Configuration tables ─────────────────────────────────────────────
# 8 job configurations: (embeddings_file, method_name, display_name, model_type, mode, knn_metric)
EMBEDDINGS=(
    "gneprop_embeddings.parquet"
    "gneprop_embeddings.parquet"
    "gneprop_embeddings.parquet"
    "gneprop_embeddings.parquet"
    "morgan_fingerprints.parquet"
    "morgan_fingerprints.parquet"
    "morgan_fingerprints.parquet"
    "morgan_fingerprints.parquet"
)
METHOD_NAMES=(gneprop gneprop gneprop gneprop mf mf mf mf)
DISPLAY_NAMES=(
    "GNEprop Embeddings"
    "GNEprop Embeddings"
    "GNEprop Embeddings"
    "GNEprop Embeddings"
    "Morgan Fingerprints"
    "Morgan Fingerprints"
    "Morgan Fingerprints"
    "Morgan Fingerprints"
)
MODEL_TYPES=(linear_regression knn_regression logistic_regression knn_classifier linear_regression knn_regression logistic_regression knn_classifier)
MODES=(regression regression classifier classifier regression regression classifier classifier)
KNN_METRICS=("" euclidean "" "" "" cosine "" "")

MOAS=(pi3k_mtor hsp90 jak_rock hdac mek cdk)
SEEDS=(0 1 2 3 4 5)

# ── Decode task ID ───────────────────────────────────────────────────
TASK_ID=${SLURM_ARRAY_TASK_ID}
CONFIG_IDX=$((TASK_ID / 36))
REMAINDER=$((TASK_ID % 36))
SEED_IDX=$((REMAINDER / 6))
MOA_IDX=$((REMAINDER % 6))

EMBEDDING="${EMBEDDINGS[$CONFIG_IDX]}"
METHOD_NAME="${METHOD_NAMES[$CONFIG_IDX]}"
DISPLAY_NAME="${DISPLAY_NAMES[$CONFIG_IDX]}"
MODEL_TYPE="${MODEL_TYPES[$CONFIG_IDX]}"
MODE="${MODES[$CONFIG_IDX]}"
KNN_METRIC="${KNN_METRICS[$CONFIG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
MOA="${MOAS[$MOA_IDX]}"

EMBEDDINGS_PATH="${REPO_ROOT}/final_data/embeddings/structure/${EMBEDDING}"
DATA_DIR="${REPO_ROOT}/data/loo_comparisons_cache"
OUTPUT_DIR="${DATA_DIR}/partials"
NEIGHBOR_DIR="${REPO_ROOT}/final_data/library_similarities/neighbor_indices"
mkdir -p "$OUTPUT_DIR"

# ── Skip if output already exists ────────────────────────────────────
OUT_FILE="${OUTPUT_DIR}/${MODE}_${METHOD_NAME}_${MODEL_TYPE}_seed${SEED}_${MOA}.parquet"
if [[ -f "$OUT_FILE" ]]; then
    echo "Output already exists, skipping: ${OUT_FILE}"
    exit 0
fi

# ── Build command ────────────────────────────────────────────────────
echo "Task ${TASK_ID}: config=${CONFIG_IDX} seed=${SEED} moa=${MOA} model=${MODEL_TYPE} method=${METHOD_NAME}"

CMD=(
    python -u "$SCRIPT_DIR/loo_baseline_worker.py"
    --mode "$MODE"
    --seed "$SEED"
    --model-type "$MODEL_TYPE"
    --moa "$MOA"
    --embeddings-path "$EMBEDDINGS_PATH"
    --method-name "$METHOD_NAME"
    --method-display-name "$DISPLAY_NAME"
    --data-dir "$DATA_DIR"
    --output-dir "$OUTPUT_DIR"
    --neighbor-dir "$NEIGHBOR_DIR"
    --use-tautomer-canonicalization
)

if [[ -n "$KNN_METRIC" ]]; then
    CMD+=(--knn-metric "$KNN_METRIC")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
