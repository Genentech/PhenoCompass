#!/usr/bin/env python
"""Compute pairwise nearest neighbor indices for a set of compound embeddings.

Generates chunked CSV files compatible with kNNPredictorFromNeighborsTable
in loo_baseline_worker.py.

Usage (sequential, all chunks):
    python scripts/compute_neighbor_indices.py \
        --embeddings-path final_data/embeddings/structure/gneprop_embeddings.parquet \
        --metric euclidean \
        --output-dir final_data/library_similarities/neighbor_indices \
        --output-prefix pairwise_gneprop_euclidean

Usage (single chunk, for parallelization across an array job):
    python scripts/compute_neighbor_indices.py \
        --embeddings-path final_data/embeddings/structure/gneprop_embeddings.parquet \
        --metric euclidean \
        --output-dir final_data/library_similarities/neighbor_indices \
        --output-prefix pairwise_gneprop_euclidean \
        --task-id $TASK_ID
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from phenocompass.evaluation.metrics import compute_pairwise_tanimoto_similarity


def compute_chunk(embeddings_df, chunk_id, batch_size, metric, top_k):
    """Compute top-k neighbors for one chunk of compounds."""
    all_ids = embeddings_df.index.tolist()
    n_compounds = len(all_ids)

    start = chunk_id * batch_size
    end = min(start + batch_size, n_compounds)
    chunk_ids = all_ids[start:end]

    if metric == "tanimoto":
        raise ValueError(
            "Tanimoto metric requires fingerprint objects, not numeric embeddings. "
            "Use --metric cosine or euclidean for parquet-based embeddings."
        )

    query = embeddings_df.iloc[start:end].values
    reference = embeddings_df.values

    if metric == "euclidean":
        distances = pairwise_distances(query, reference, metric="euclidean")
        sorted_indices = np.argsort(distances, axis=1)
        top_indices = sorted_indices[:, 1:top_k + 1]
        top_similarities = np.take_along_axis(-distances, sorted_indices[:, 1:top_k + 1], axis=1)
    elif metric == "cosine":
        distances = pairwise_distances(query, reference, metric="cosine")
        sorted_indices = np.argsort(distances, axis=1)
        top_indices = sorted_indices[:, 1:top_k + 1]
        top_similarities = 1.0 - np.take_along_axis(distances, sorted_indices[:, 1:top_k + 1], axis=1)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'euclidean' or 'cosine'.")

    indices_df = pd.DataFrame(top_indices, index=chunk_ids, columns=range(top_k))
    similarities_df = pd.DataFrame(top_similarities, index=chunk_ids, columns=range(top_k))

    return indices_df, similarities_df


def main():
    parser = argparse.ArgumentParser(
        description="Compute pairwise nearest neighbor indices for compound embeddings."
    )
    parser.add_argument("--embeddings-path", type=str, required=True,
                        help="Parquet file with embeddings (index=SMILES, columns=features)")
    parser.add_argument("--metric", type=str, required=True, choices=["euclidean", "cosine"],
                        help="Distance metric for neighbor computation")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Directory for output CSV files")
    parser.add_argument("--output-prefix", type=str, required=True,
                        help="Filename prefix (e.g., 'pairwise_gneprop_euclidean')")
    parser.add_argument("--top-k", type=int, default=500,
                        help="Number of nearest neighbors to store (default: 500)")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Compounds per chunk (default: 512)")
    parser.add_argument("--task-id", type=int, default=None,
                        help="Process only this chunk (for SLURM). If omitted, process all.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading embeddings from {args.embeddings_path}...")
    embeddings_df = pd.read_parquet(args.embeddings_path)
    print(f"  Shape: {embeddings_df.shape}")

    n_compounds = len(embeddings_df)
    n_chunks = (n_compounds + args.batch_size - 1) // args.batch_size
    print(f"  Chunks: {n_chunks} (batch_size={args.batch_size})")

    if args.task_id is not None:
        chunks_to_process = [args.task_id]
    else:
        chunks_to_process = range(n_chunks)

    for chunk_id in chunks_to_process:
        indices_path = os.path.join(args.output_dir, f"{args.output_prefix}.{chunk_id}.indices.csv")
        sims_path = os.path.join(args.output_dir, f"{args.output_prefix}.{chunk_id}.similarities.csv")

        if os.path.exists(indices_path) and os.path.exists(sims_path):
            print(f"  Chunk {chunk_id}/{n_chunks - 1}: already exists, skipping")
            continue

        print(f"  Chunk {chunk_id}/{n_chunks - 1}: computing...", end="", flush=True)
        indices_df, sims_df = compute_chunk(
            embeddings_df, chunk_id, args.batch_size, args.metric, args.top_k
        )
        indices_df.to_csv(indices_path)
        sims_df.to_csv(sims_path)
        print(f" saved ({len(indices_df)} compounds)")

    if args.task_id is None:
        print(f"\nDone. {n_chunks} chunks saved to {args.output_dir}/")
    else:
        print(f"\nChunk {args.task_id} complete.")

    print(f"\nFor SLURM parallelization, use: --array=0-{n_chunks - 1}")


if __name__ == "__main__":
    main()
