"""Hypergeometric enrichment analysis for similarity-based group testing."""

from collections import defaultdict
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from statsmodels.stats.multitest import multipletests


def compute_enrichment_analysis(
    similarity_scores: pd.DataFrame,
    group_annotations: Union[List[str], np.ndarray, pd.Series],
    k: int = 10,
    min_group_size: int = 1,
    correction_method: Optional[str] = "fdr_bh",
) -> pd.DataFrame:
    """Compute hypergeometric enrichment for query compounds against annotated groups.

    For each query, identifies the top-k most similar references and tests whether
    any group is over-represented among these neighbors using the hypergeometric test.

    Parameters
    ----------
    similarity_scores : pd.DataFrame
        Similarity matrix from compute_crossmodal_similarity().
        Shape: (n_queries, n_refs). Index = query IDs, columns = reference IDs.
    group_annotations : Union[List[str], np.ndarray, pd.Series]
        Group label for each reference compound, length = n_refs.
        Order must match similarity_scores columns.
    k : int, default=10
        Number of top neighbors to consider as "hits" for each query.
    min_group_size : int, default=1
        Minimum group size to include in analysis. Groups smaller than this
        are excluded from results.
    correction_method : Optional[str], default="fdr_bh"
        Multiple testing correction method passed to statsmodels.multipletests.
        Common options: "fdr_bh" (Benjamini-Hochberg), "bonferroni", or None.

    Returns
    -------
    pd.DataFrame
        Enrichment results with one row per (query, group) pair. Columns:
        - "query": Query identifier
        - "group": Group name being tested
        - "k": Number of neighbors considered (N in hypergeom notation)
        - "hits_in_group": Count of top-k neighbors in this group (x)
        - "group_size": Total references in this group (n)
        - "n_refs": Total number of references (M)
        - "expected_hits": Expected hits under null (k * group_size / n_refs)
        - "fold_enrichment": hits_in_group / expected_hits
        - "p_value": Raw hypergeometric p-value
        - "p_adjusted": Corrected p-value (if correction_method specified)

    Notes
    -----
    Uses the same statistical approach as the Cell Painting validation notebook:
        p = hypergeom.sf(x - 1, M, n, N)
    where M = population size, n = group size, N = draws, x = observed hits.

    Examples
    --------
    >>> from phenocompass.evaluation import compute_enrichment_analysis
    >>> enrichment = compute_enrichment_analysis(similarity_scores, moa_labels, k=10)
    >>> significant = enrichment[enrichment["p_adjusted"] < 0.05]
    """
    # Validate inputs
    n_queries, n_refs = similarity_scores.shape
    group_annotations = np.asarray(group_annotations)

    if len(group_annotations) != n_refs:
        raise ValueError(
            f"group_annotations length ({len(group_annotations)}) must match "
            f"number of reference columns ({n_refs})"
        )

    if k > n_refs:
        raise ValueError(f"k ({k}) cannot exceed number of references ({n_refs})")

    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")

    # Get unique groups and filter by minimum size
    unique_groups, group_counts = np.unique(group_annotations, return_counts=True)
    group_sizes = dict(zip(unique_groups, group_counts))
    valid_groups = [g for g, size in group_sizes.items() if size >= min_group_size]

    if len(valid_groups) == 0:
        raise ValueError(
            f"No groups meet min_group_size={min_group_size}. "
            f"Group sizes: {group_sizes}"
        )

    # Build results using the notebook pattern
    enrichment_dict = defaultdict(list)
    query_ids = similarity_scores.index.tolist()

    for query_idx, query_id in enumerate(query_ids):
        # Get top-k reference indices for this query
        row_similarities = similarity_scores.iloc[query_idx].values
        top_k_indices = np.argsort(row_similarities)[-k:][::-1]
        top_k_groups = group_annotations[top_k_indices]

        # Count hits per group among top-k
        for group in valid_groups:
            x = np.sum(top_k_groups == group)  # Observed hits
            n = group_sizes[group]  # Group size in population
            M = n_refs  # Total population
            N = k  # Number of draws

            # Hypergeometric p-value: P(X >= x)
            pval = hypergeom.sf(x - 1, M, n, N)

            # Expected hits and fold enrichment
            expected = N * n / M
            fold_enrichment = x / expected if expected > 0 else 0.0

            enrichment_dict["query"].append(query_id)
            enrichment_dict["group"].append(group)
            enrichment_dict["k"].append(N)
            enrichment_dict["hits_in_group"].append(x)
            enrichment_dict["group_size"].append(n)
            enrichment_dict["n_refs"].append(M)
            enrichment_dict["expected_hits"].append(expected)
            enrichment_dict["fold_enrichment"].append(fold_enrichment)
            enrichment_dict["p_value"].append(pval)

    results_df = pd.DataFrame(enrichment_dict)

    # Apply multiple testing correction using statsmodels
    if correction_method is not None and len(results_df) > 0:
        _, pvals_corrected, _, _ = multipletests(
            results_df["p_value"].values, alpha=0.05, method=correction_method
        )
        results_df["p_adjusted"] = pvals_corrected
    elif len(results_df) > 0:
        results_df["p_adjusted"] = results_df["p_value"]

    return results_df
