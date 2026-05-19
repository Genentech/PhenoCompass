#!/bin/bash
# Run LOO baseline evaluation locally (no SLURM).
# Iterates over all (seed, moa) combinations for a given model_type,
# skipping combinations whose partial output file already exists.
# Output partials are consumed by scripts/preprocess_loo_comparisons.py.
#
# Activate the phenocompass environment before running this script
# (conda/venv activation is intentionally left to the caller).
#
# Usage:
#   bash scripts/run_loo_baselines_local.sh <model_type> \
#       --embeddings-path <path> \
#       --method-name <name> \
#       --method-display-name <display_name> \
#       --data-dir <dir> \
#       --neighbor-dir <dir> \
#       [--neighbor-prefix <prefix>] \
#       [--knn-metric <metric>] \
#       [--use-tautomer-canonicalization]
#
# Example:
#   bash scripts/run_loo_baselines_local.sh logistic_regression \
#       --embeddings-path final_data/embeddings/structure/gneprop_embeddings.parquet \
#       --method-name gneprop \
#       --method-display-name "GNEprop Embeddings" \
#       --data-dir data/loo_comparisons_cache \
#       --neighbor-dir final_data/library_similarities/neighbor_indices \
#       --use-tautomer-canonicalization
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/scripts"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/notebooks:${PYTHONPATH:-}"

# ── Parse model type ──────────────────────────────────────────────────
MODEL_TYPE="${1:?Usage: $0 <model_type> --embeddings-path ... --method-name ... --method-display-name ... --data-dir ... --neighbor-dir ...}"
shift

case "$MODEL_TYPE" in
    linear_regression|knn_regression)
        MODE="regression" ;;
    logistic_regression|knn_classifier)
        MODE="classifier" ;;
    *)
        echo "Error: unknown model type '$MODEL_TYPE'." >&2
        echo "Choose: logistic_regression, knn_classifier, linear_regression, knn_regression" >&2
        exit 1 ;;
esac

# ── Parse remaining arguments ─────────────────────────────────────────
EMBEDDINGS_PATH=""
METHOD_NAME=""
METHOD_DISPLAY_NAME=""
DATA_DIR=""
NEIGHBOR_DIR=""
NEIGHBOR_PREFIX=""
KNN_METRIC=""
USE_TAUTOMER_CANON=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --embeddings-path) EMBEDDINGS_PATH="$2"; shift 2 ;;
        --method-name) METHOD_NAME="$2"; shift 2 ;;
        --method-display-name) METHOD_DISPLAY_NAME="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --neighbor-dir) NEIGHBOR_DIR="$2"; shift 2 ;;
        --neighbor-prefix) NEIGHBOR_PREFIX="$2"; shift 2 ;;
        --knn-metric) KNN_METRIC="$2"; shift 2 ;;
        --use-tautomer-canonicalization) USE_TAUTOMER_CANON=true; shift ;;
        *) echo "Error: unknown argument '$1'" >&2; exit 1 ;;
    esac
done

# Validate required arguments
if [[ -z "$EMBEDDINGS_PATH" ]]; then echo "Error: --embeddings-path is required" >&2; exit 1; fi
if [[ -z "$METHOD_NAME" ]]; then echo "Error: --method-name is required" >&2; exit 1; fi
if [[ -z "$METHOD_DISPLAY_NAME" ]]; then echo "Error: --method-display-name is required" >&2; exit 1; fi
if [[ -z "$DATA_DIR" ]]; then echo "Error: --data-dir is required" >&2; exit 1; fi
if [[ -z "$NEIGHBOR_DIR" ]]; then echo "Error: --neighbor-dir is required" >&2; exit 1; fi

OUTPUT_DIR="${DATA_DIR}/partials"
mkdir -p "$OUTPUT_DIR"

# ── Build worker arguments ────────────────────────────────────────────
MOA_KEYS=(pi3k_mtor hsp90 jak_rock hdac mek cdk)
SEEDS=(0 1 2 3 4 5)

WORKER_ARGS=(
    --embeddings-path "$EMBEDDINGS_PATH"
    --method-name "$METHOD_NAME"
    --method-display-name "$METHOD_DISPLAY_NAME"
    --data-dir "$DATA_DIR"
    --output-dir "$OUTPUT_DIR"
    --neighbor-dir "$NEIGHBOR_DIR"
)
if [[ -n "$NEIGHBOR_PREFIX" ]]; then
    WORKER_ARGS+=(--neighbor-prefix "$NEIGHBOR_PREFIX")
fi
if [[ -n "$KNN_METRIC" ]]; then
    WORKER_ARGS+=(--knn-metric "$KNN_METRIC")
fi
if [[ -n "$USE_TAUTOMER_CANON" ]]; then
    WORKER_ARGS+=(--use-tautomer-canonicalization)
fi

# ── Run jobs ──────────────────────────────────────────────────────────
total=0
skip=0
for seed in "${SEEDS[@]}"; do
    for moa in "${MOA_KEYS[@]}"; do
        total=$((total + 1))
        out_file="${OUTPUT_DIR}/${MODE}_${METHOD_NAME}_${MODEL_TYPE}_seed${seed}_${moa}.parquet"
        if [[ -f "$out_file" ]]; then
            skip=$((skip + 1))
        fi
    done
done

remaining=$((total - skip))
echo ""
echo "=== ${MODE} ${MODEL_TYPE}: ${remaining}/${total} jobs to run (${skip} already complete) ==="
echo ""

if [[ "$remaining" -eq 0 ]]; then
    echo "All jobs already complete — nothing to do."
    exit 0
fi

current=0
for seed in "${SEEDS[@]}"; do
    for moa in "${MOA_KEYS[@]}"; do
        out_file="${OUTPUT_DIR}/${MODE}_${METHOD_NAME}_${MODEL_TYPE}_seed${seed}_${moa}.parquet"
        if [[ -f "$out_file" ]]; then
            continue
        fi

        current=$((current + 1))
        echo "────────────────────────────────────────────────────────"
        echo "[${current}/${remaining}] ${MODE} ${MODEL_TYPE} seed=${seed} moa=${moa}"
        echo "────────────────────────────────────────────────────────"

        python "$SCRIPT_DIR/loo_baseline_worker.py" \
            --mode "$MODE" \
            --seed "$seed" \
            --model-type "$MODEL_TYPE" \
            --moa "$moa" \
            "${WORKER_ARGS[@]}"
    done
done

echo ""
echo "=== ${MODE} ${MODEL_TYPE} complete: ${current} jobs ran ==="
echo "Output directory: ${OUTPUT_DIR}"
