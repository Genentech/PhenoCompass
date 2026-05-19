"""Evaluation metrics for JUMP map building evaluation.

This module contains evaluation metrics, multivariate metrics (graph-based
biological recall), per-term metrics, and phenosimilarity scoring.
"""

import itertools
import math
import pathlib
import pickle
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
from statsmodels.stats.multitest import fdrcorrection

from phenocompass.evaluation.map_config import NEGATIVE_CONTROLS
from phenocompass.evaluation.map_preprocessing import (
    cosine_similarity_numba,
    euclidean_distance_numba,
    get_feature_columns,
    get_metadata_columns,
    get_negcon_perturbation_ids,
    get_random_index_pairs,
)
from phenocompass.evaluation.metrics import compute_fewshot_similarities

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


# =============================================================================
# Evaluation Metrics
# =============================================================================


def knn_vote_accuracy(
    df: pd.DataFrame,
    feature_columns: List[str],
    class_columns: List[str],
    k_values: List[int] = [1, 5, 10, 50],
    metric: str = "cosine",
) -> pd.DataFrame:
    """Compute kNN voting accuracy for batch effect evaluation.

    For each sample, predict class labels based on k nearest neighbors
    and compute accuracy.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with features and class labels.
    feature_columns : List[str]
        Feature columns to use for kNN.
    class_columns : List[str]
        Class label columns (e.g., source, batch, plate).
    k_values : List[int]
        k values for kNN.
    metric : str
        Distance metric.

    Returns
    -------
    pd.DataFrame
        DataFrame with accuracy for each k and class column.
    """
    X = df[feature_columns].values

    # Fit kNN
    max_k = max(k_values)
    nn = NearestNeighbors(n_neighbors=max_k + 1, metric=metric)
    nn.fit(X)

    # Get neighbor indices (excluding self)
    distances, indices = nn.kneighbors(X)
    neighbor_indices = indices[:, 1:]  # Exclude self

    results = []
    for class_col in class_columns:
        labels = df[class_col].values

        for k in k_values:
            # Get labels of k nearest neighbors
            neighbor_labels = labels[neighbor_indices[:, :k]]

            # Majority vote prediction
            predictions = np.array([
                np.argmax(np.bincount(row.astype(int))) if row.dtype in [np.int32, np.int64, int]
                else pd.Series(row).mode().iloc[0]
                for row in neighbor_labels
            ])

            # Compute accuracy
            accuracy = (predictions == labels).mean()

            results.append({
                "class_column": class_col,
                "k": k,
                "accuracy": accuracy,
            })

    return pd.DataFrame(results)


def reproducibility_ks(
    df: pd.DataFrame,
    variables: List[str],
    n_samples: int = 10000,
    columns: Optional[List[str]] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Evaluate reproducibility using Kolmogorov-Smirnov test on multiple variables.

    For each variable, samples positive pairs (same group) and negative pairs
    (random) and computes KS statistic on cosine similarity distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with features and metadata.
    variables : List[str]
        List of metadata column names to evaluate (e.g., ["Metadata_Perturbation"]).
    n_samples : int
        Number of sample pairs for positive and negative distributions.
    columns : List[str], optional
        Feature columns to use. If None, uses all non-Metadata columns.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: statistic, pvalue, statistic_location,
        statistic_sign, variable.
    """
    np.random.seed(seed)

    if columns is None:
        columns = [c for c in df.columns if not c.startswith("Metadata_")]

    features = df[columns].values

    summary_rows = []
    for variable in variables:
        if variable not in df.columns:
            print(f"Warning: Variable {variable} not found in DataFrame")
            continue

        # Filter to groups with more than one replicate
        counts = df[variable].value_counts()
        counts_filt = counts[counts > 1]

        if len(counts_filt) == 0:
            print(f"Warning: No replicated groups for {variable}")
            continue

        # Sample groups (with replacement if needed)
        replace = n_samples > len(counts_filt)
        counts_sample = counts_filt.sample(n=n_samples, replace=replace, random_state=seed)

        # Get count distribution for sampled groups
        var_counts = counts_sample.index.value_counts()
        var_df = df[df[variable].isin(var_counts.index)]

        # Assemble paired indices for positive samples
        idx1, idx2 = [], []
        for v, df_group in var_df.groupby(variable):
            n_v = var_counts.get(v, 0)
            for _ in range(n_v):
                if len(df_group) >= 2:
                    sample_idx = df_group.sample(n=2, replace=False).index.tolist()
                    idx1.append(sample_idx[0])
                    idx2.append(sample_idx[1])

        if len(idx1) == 0:
            print(f"Warning: Could not sample positive pairs for {variable}")
            continue

        # Get feature indices from df index
        df_index_to_pos = {idx: i for i, idx in enumerate(df.index)}
        pos_idx1 = np.array([df_index_to_pos[i] for i in idx1])
        pos_idx2 = np.array([df_index_to_pos[i] for i in idx2])

        features1 = features[pos_idx1]
        features2 = features[pos_idx2]
        positives = cosine_similarity_numba(features1, features2)

        # Sample random indices for negatives
        neg_idx1, neg_idx2 = get_random_index_pairs(len(df), sample_size=n_samples)
        negatives = cosine_similarity_numba(
            features[neg_idx1], features[neg_idx2]
        )

        # KS test
        ks_result = ks_2samp(positives, negatives)

        summary_rows.append({
            "statistic": ks_result.statistic,
            "pvalue": ks_result.pvalue,
            "statistic_location": ks_result.statistic_location,
            "statistic_sign": ks_result.statistic_sign,
            "variable": variable,
        })

    return pd.DataFrame(summary_rows)


def univariate_self_consistency(
    df: pd.DataFrame,
    feature_columns: List[str],
    perturbation_col: str = "Metadata_JCP2022",
    perturbation_type_col: str = "Metadata_PerturbationType",
    n_random: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute univariate self-consistency metrics.

    For each perturbation, compute correlation between replicate profiles
    and compare to random correlations.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    feature_columns : List[str]
        Feature columns.
    perturbation_col : str
        Column containing perturbation IDs.
    perturbation_type_col : str
        Column containing perturbation types.
    n_random : int
        Number of random comparisons.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame
        DataFrame with self-consistency metrics.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    np.random.seed(seed)

    # Get perturbations with 2+ replicates (excluding controls)
    negcon_ids = get_negcon_perturbation_ids(df, perturbation_type_col, perturbation_col)

    pert_counts = df[perturbation_col].value_counts()
    valid_perts = [
        p for p in pert_counts[pert_counts >= 2].index
        if p not in negcon_ids
    ]

    if len(valid_perts) < 10:
        print("Warning: Too few perturbations with replicates")
        return pd.DataFrame()

    X = df[feature_columns].values
    perturbations = df[perturbation_col].values

    # Compute replicate correlations
    replicate_sims = []
    for pert in valid_perts:
        pert_idx = np.where(perturbations == pert)[0]
        if len(pert_idx) >= 2:
            # Average pairwise similarity among replicates
            pert_X = X[pert_idx]
            sim_matrix = cosine_similarity(pert_X)
            upper_tri = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
            replicate_sims.append(np.mean(upper_tri))

    # Compute random correlations
    random_sims = []
    for _ in range(n_random):
        i, j = np.random.choice(len(X), 2, replace=False)
        if perturbations[i] != perturbations[j]:
            sim = cosine_similarity(X[i:i+1], X[j:j+1])[0, 0]
            random_sims.append(sim)

    return pd.DataFrame({
        "metric": ["replicate_sim", "random_sim", "enrichment"],
        "mean": [np.mean(replicate_sims), np.mean(random_sims),
                 np.mean(replicate_sims) - np.mean(random_sims)],
        "std": [np.std(replicate_sims), np.std(random_sims), np.nan],
        "n": [len(replicate_sims), len(random_sims), np.nan],
    })


# =============================================================================
# Univariate Metrics (Leave-One-Out Cosine Similarity)
# =============================================================================


def _loo_cossim(mat: np.ndarray) -> float:
    """Calculate leave-one-out cosine similarity over array mat.

    For each row, computes the cosine similarity between that row and the
    mean of the remaining rows, then averages across all rows.

    Parameters
    ----------
    mat : np.ndarray
        Feature matrix of shape (n_samples, n_features).

    Returns
    -------
    float
        Mean leave-one-out cosine similarity.
    """
    weights = np.ones(mat.shape[:-1])
    mat_weighted = mat * np.expand_dims(weights, -1)
    sum_rep = np.expand_dims(np.sum(mat_weighted, axis=-2), -2)
    total_weights = np.expand_dims(np.sum(weights, axis=-1), -1)
    loo_weights = np.expand_dims((total_weights - weights), -1)
    loo = (sum_rep - mat_weighted) / loo_weights
    sim = cosine_similarity_numba(loo, mat)
    return np.sum(sim, axis=-1) / sim.shape[-1]


def _score_against_null(val: float, null_vals: np.ndarray) -> float:
    """Score test statistic against empirical null to obtain a p-value.

    Assumes one-sided test such that larger values yield lower p-values.

    Parameters
    ----------
    val : float
        Observed test statistic.
    null_vals : np.ndarray
        Null distribution values.

    Returns
    -------
    float
        Empirical p-value.
    """
    return (np.sum(null_vals >= val) + 1) / (len(null_vals) + 1)


def _summarize_univariate(
    df: pd.DataFrame,
    perturbation_type_field: str,
) -> pd.DataFrame:
    """Summarize univariate metrics by perturbation type.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``significant`` and perturbation type columns.
    perturbation_type_field : str
        Column containing perturbation type labels.

    Returns
    -------
    pd.DataFrame
        Summary with columns: perturbation_type_field, significant,
        not_significant, significant_percent.
    """
    groupby = [perturbation_type_field]

    out = df.groupby(groupby, dropna=False)["significant"].value_counts().unstack()
    out.columns.name = None
    out.rename(columns={True: "significant", False: "not_significant"}, inplace=True)

    # Ensure both columns exist
    for col in ["significant", "not_significant"]:
        if col not in out.columns:
            out[col] = 0

    # Add perturbation_type=all row
    alls = out[["significant", "not_significant"]].sum().to_frame().T
    alls[perturbation_type_field] = "all"
    alls.set_index(groupby, inplace=True)
    out = pd.concat([out, alls])

    out.sort_values(groupby, inplace=True)
    out.reset_index(inplace=True)

    out["not_significant"] = out["not_significant"].astype(int)
    out["significant"] = out["significant"].astype(int)
    out["significant_percent"] = out["significant"] / out[["significant", "not_significant"]].sum(axis=1)

    return out


def univariate_metric_all_random(
    df: pd.DataFrame,
    columns: List[str],
    perturbation_name_field: str,
    perturbation_id_field: str,
    perturbation_type_field: str,
    n_samples: int = 9999,
    significance_level: float = 0.05,
    correct_pvalues: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Calculate univariate metrics for all perturbations using leave-one-out
    cosine similarity against an empirical null distribution.

    For each perturbation with 2+ replicates, computes the leave-one-out
    cosine similarity and tests it against a null distribution built by
    randomly sampling the same number of profiles.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with features and metadata.
    columns : List[str]
        Feature columns to use for similarity computation.
    perturbation_name_field : str
        Column containing perturbation names (e.g., SMILES).
    perturbation_id_field : str
        Column containing perturbation IDs (e.g., JCP2022).
    perturbation_type_field : str
        Column containing perturbation type labels.
    n_samples : int, default 9999
        Number of null samples to draw for each replicate count.
    significance_level : float, default 0.05
        P-value threshold for significance (applied to corrected p-values
        when ``correct_pvalues=True``).
    correct_pvalues : bool, default True
        If True, apply Benjamini-Hochberg FDR correction to raw p-values
        and determine significance from the corrected values.

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary with keys:
        - ``"metrics"``: per-perturbation results with columns perturbation,
          p-value, p-value_corrected, significant, plus metadata columns
          from merge.
        - ``"summary"``: summary by perturbation type with columns
          perturbation_type_field, significant, not_significant,
          significant_percent.
    """
    # Get all replicate counts that we need to build nulls for
    ns = [n for n in df[perturbation_id_field].value_counts().unique() if n >= 2]

    # Build null distributions
    mat = df[columns].values
    nulls = {}
    for n in tqdm(ns, desc="Building null distributions"):
        null = np.zeros(n_samples)
        for i in range(n_samples):
            idx = np.random.choice(mat.shape[0], replace=False, size=(n,))
            null[i] = _loo_cossim(mat[idx, :])
        nulls[n] = null

    # Calculate test statistics per perturbation
    ret = []
    for perturbation, idx in df.reset_index().groupby(perturbation_id_field, observed=True).groups.items():
        if idx.shape[0] <= 1:
            ret.append([perturbation, np.nan])
            continue
        pval = _score_against_null(_loo_cossim(mat[idx, :]), nulls[idx.shape[0]])
        ret.append([perturbation, pval])

    ret = pd.DataFrame.from_records(ret, columns=["perturbation", "p-value"])

    # FDR correction and significance determination
    mask = ret["p-value"].notna()
    if correct_pvalues and mask.any():
        _, ret.loc[mask, "p-value_corrected"] = fdrcorrection(
            ret.loc[mask, "p-value"], alpha=significance_level
        )
    else:
        ret["p-value_corrected"] = ret["p-value"]
    ret["significant"] = False
    ret.loc[mask, "significant"] = ret.loc[mask, "p-value_corrected"] <= significance_level

    # Merge metadata
    metadata_df = df[
        list(set([perturbation_id_field, perturbation_name_field, perturbation_type_field]))
    ].drop_duplicates()
    ret = ret.merge(metadata_df, how="left", left_on="perturbation", right_on=perturbation_id_field)
    ret["significant"] = ret["significant"].astype("category").cat.set_categories([True, False])

    # Build summary
    summary_df = _summarize_univariate(ret, perturbation_type_field=perturbation_type_field)

    return {"metrics": ret, "summary": summary_df}


# =============================================================================
# Multivariate Metrics (Graph-based Biological Recall)
# =============================================================================


def _opentarget_to_mol_graph(
    graph_path: pathlib.Path,
    molecules: np.ndarray,
) -> Tuple[nx.Graph, Dict[Any, Any]]:
    """Load OpenTarget graph and return subgraph containing only specified molecules.

    Parameters
    ----------
    graph_path : pathlib.Path
        Path to the pickled graph file.
    molecules : np.ndarray
        Array of molecule identifiers to include.

    Returns
    -------
    Tuple[nx.Graph, Dict[Any, Any]]
        Subgraph and sets dictionary.
    """
    with open(graph_path, "rb") as f:
        data = pickle.load(f)
    if "compound2gene" in str(graph_path):
        return data, {}
    graph = data["graph"].subgraph(molecules)
    sets = {k: list(set(v) & set(molecules)) for k, v in data["sets"].items()}
    return graph, sets


def _enrichr_to_gene_graph(
    db_path: pathlib.Path,
    perturbations: np.ndarray,
) -> Tuple[nx.Graph, Dict[Any, Any]]:
    """Load Enrichr gene set database and return a graph of genes.

    Parameters
    ----------
    db_path : pathlib.Path
        Path to the Enrichr text file.
    perturbations : np.ndarray
        Array of perturbation identifiers.

    Returns
    -------
    Tuple[nx.Graph, Dict[Any, Any]]
        Gene graph and gene sets dictionary.
    """
    with open(db_path) as f:
        data = f.readlines()
    gene_sets = {
        x.split("\t\t")[0]: [
            gene for gene in x.split("\t\t")[1].split("\t")
            if gene in perturbations
        ]
        for x in data
    }
    unique_genes = set(itertools.chain.from_iterable(gene_sets.values()))
    gene_graph = nx.Graph()
    gene_graph.add_nodes_from(unique_genes)
    for genes in gene_sets.values():
        gene_graph.add_edges_from(itertools.combinations(genes, 2))
    return gene_graph, gene_sets


def _weighted_ecdf(sample: np.ndarray, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute weighted empirical cumulative distribution function.

    Parameters
    ----------
    sample : np.ndarray
        Sample values.
    weights : np.ndarray
        Sample weights.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Sorted sample and cumulative weights.
    """
    sorted_indices = np.argsort(sample)
    sorted_sample = sample[sorted_indices]
    sorted_weights = weights[sorted_indices]
    normalized_weights = sorted_weights / sorted_weights.sum()
    cum_weights = np.cumsum(normalized_weights)
    return sorted_sample, cum_weights


def _get_weighted_ks_stat(
    data1: np.ndarray,
    data2: np.ndarray,
    weights1: np.ndarray,
    weights2: np.ndarray,
) -> float:
    """Compute Kolmogorov-Smirnov statistic on 2 weighted samples.

    Parameters
    ----------
    data1 : np.ndarray
        First sample values.
    data2 : np.ndarray
        Second sample values.
    weights1 : np.ndarray
        Weights for first sample.
    weights2 : np.ndarray
        Weights for second sample.

    Returns
    -------
    float
        KS statistic.
    """
    ecdf1_x, ecdf1_y = _weighted_ecdf(data1, weights1)
    ecdf2_x, ecdf2_y = _weighted_ecdf(data2, weights2)

    data_all = np.concatenate([ecdf1_x, ecdf2_x])
    data_unique = np.unique(data_all)

    ecdf1_values = interp1d(
        ecdf1_x, ecdf1_y, kind="previous", fill_value=(0, 1), bounds_error=False
    )(data_unique)
    ecdf2_values = interp1d(
        ecdf2_x, ecdf2_y, kind="previous", fill_value=(0, 1), bounds_error=False
    )(data_unique)

    ecdf_diff = ecdf1_values - ecdf2_values
    ks_statistic = np.max(ecdf_diff)
    return ks_statistic


def _get_ks_stat(positive_distances: np.ndarray, negative_distances: np.ndarray) -> float:
    """Compute KS statistic between positive and negative distance distributions.

    Parameters
    ----------
    positive_distances : np.ndarray
        Distances for positive (ground truth) pairs.
    negative_distances : np.ndarray
        Distances for negative (random) pairs.

    Returns
    -------
    float
        KS statistic.
    """
    positive_weights = np.ones_like(positive_distances)
    negative_weights = np.ones_like(negative_distances)
    return _get_weighted_ks_stat(
        positive_distances, negative_distances, positive_weights, negative_weights
    )


def _calculate_weights(
    ground_truth: nx.Graph,
    keys1: List[Any],
    keys2: List[Any],
) -> np.ndarray:
    """Calculate weights for edges based on node degrees.

    Parameters
    ----------
    ground_truth : nx.Graph
        Ground truth graph.
    keys1 : List[Any]
        First node in each edge.
    keys2 : List[Any]
        Second node in each edge.

    Returns
    -------
    np.ndarray
        Edge weights.
    """
    degree1 = np.array([ground_truth.degree[k] for k in keys1])
    degree2 = np.array([ground_truth.degree[k] for k in keys2])
    weights = 1 / (degree1 * (degree2 + 1) + degree2 * (degree1 + 1))
    return weights


def _distance_function(
    features1: np.ndarray,
    features2: np.ndarray,
    distance_metric: str,
) -> np.ndarray:
    """Calculate distance between two sets of features.

    Parameters
    ----------
    features1 : np.ndarray
        First feature array.
    features2 : np.ndarray
        Second feature array.
    distance_metric : str
        Distance metric ("cosine" or "sqeuclidean").

    Returns
    -------
    np.ndarray
        Distance values.
    """
    if distance_metric == "cosine":
        return 1 - cosine_similarity_numba(features1, features2)
    elif distance_metric == "sqeuclidean":
        return euclidean_distance_numba(features1, features2)
    else:
        raise NotImplementedError(
            "distance metric needs to be one of ['cosine', 'sqeuclidean']"
        )


def _get_negative_distances_global(
    features: np.ndarray,
    sample_size: int,
    distance_metric: str,
) -> np.ndarray:
    """Compute pairwise distances, optionally randomly subsampled.

    Parameters
    ----------
    features : np.ndarray
        Feature matrix.
    sample_size : int
        Number of pairs to sample.
    distance_metric : str
        Distance metric.

    Returns
    -------
    np.ndarray
        Distance values.
    """
    num_perts = len(features)
    if sample_size < math.comb(num_perts, 2):
        idx1, idx2 = get_random_index_pairs(len(features), sample_size)
    else:
        idx1, idx2 = np.triu_indices(n=num_perts, k=1)
    features1, features2 = features[idx1], features[idx2]
    return _distance_function(features1, features2, distance_metric)


def get_negative_distances_local(
    keys: np.ndarray,
    key2idx: Dict[Any, int],
    features: np.ndarray,
    gt_keys: List[Any],
    sample_size: int,
    distance_metric: str,
) -> np.ndarray:
    """Compute pairwise distances for each node to all other nodes.

    Parameters
    ----------
    keys : np.ndarray
        Array of perturbation keys.
    key2idx : Dict[Any, int]
        Mapping from key to index.
    features : np.ndarray
        Feature matrix.
    gt_keys : List[Any]
        Ground truth keys.
    sample_size : int
        Number of pairs to sample per node.
    distance_metric : str
        Distance metric.

    Returns
    -------
    np.ndarray
        Distance matrix.
    """
    ret = []
    for node in gt_keys:
        node_idx = key2idx[node]
        if len(keys) - 1 <= sample_size:
            idx = np.arange(len(keys) - 1)
        else:
            idx = np.random.choice(len(keys) - 1, sample_size, replace=False)
        # Adjust to exclude diagonal
        idx = idx.copy()
        idx[idx >= node_idx] += 1
        ret.append(
            cdist(
                features[node_idx][np.newaxis, ...],
                features[idx],
                metric=distance_metric,
            )
        )
    return np.vstack(ret)


def _get_positive_distances(
    key2idx: Dict[Any, int],
    features: np.ndarray,
    gt_graph: nx.Graph,
    distance_metric: str,
) -> Tuple[List[Any], List[Any], np.ndarray]:
    """Compute pairwise distances for all edges in ground truth graph.

    Parameters
    ----------
    key2idx : Dict[Any, int]
        Mapping from key to index.
    features : np.ndarray
        Feature matrix.
    gt_graph : nx.Graph
        Ground truth graph.
    distance_metric : str
        Distance metric.

    Returns
    -------
    Tuple[List[Any], List[Any], np.ndarray]
        Keys for first node, keys for second node, and distances.
    """
    idx1, idx2 = [], []
    keys1, keys2 = [], []
    for v1, v2, _ in nx.to_edgelist(gt_graph):
        keys1.append(v1)
        keys2.append(v2)
        idx1.append(key2idx[v1])
        idx2.append(key2idx[v2])

        keys1.append(v2)
        keys2.append(v1)
        idx1.append(key2idx[v2])
        idx2.append(key2idx[v1])
    idx1, idx2 = np.array(idx1), np.array(idx2)

    gt_distances = _distance_function(features[idx1], features[idx2], distance_metric)
    return keys1, keys2, gt_distances


def get_performance_metrics(
    keys: np.ndarray,
    features: np.ndarray,
    universe: np.ndarray,
    gt_graph: nx.Graph,
    sample_size_global: int = 10000,
    sample_size_local: int = 100,
    distance_metric: str = "cosine",
    percentiles: np.ndarray = None,
) -> pd.DataFrame:
    """Calculate performance metrics for graph-based predictions.

    Parameters
    ----------
    keys : np.ndarray
        Perturbation keys (must be unique).
    features : np.ndarray
        Feature matrix.
    universe : np.ndarray
        Universe of perturbations to consider.
    gt_graph : nx.Graph
        Ground truth graph.
    sample_size_global : int
        Sample size for global negative distances.
    sample_size_local : int
        Sample size for local negative distances.
    distance_metric : str
        Distance metric ("cosine" or "sqeuclidean").
    percentiles : np.ndarray, optional
        Percentiles for recall computation.

    Returns
    -------
    pd.DataFrame
        DataFrame with performance metrics.
    """
    if percentiles is None:
        percentiles = np.arange(75, 101, 1)

    # Check for non-unique keys
    if np.any(np.unique(keys, return_counts=True)[1] != 1):
        raise ValueError(
            "`keys` should contain unique elements. Please aggregate to perturbation level."
        )

    # Subset graph to unique keys
    gt_graph = gt_graph.subgraph(np.unique(keys))
    num_nodes = gt_graph.number_of_nodes()
    num_edges = gt_graph.number_of_edges()
    num_universe = len(universe)

    # Initialize returns
    recalls, weighted_recalls = {}, {}

    # Handle edge cases
    if num_universe <= 1 or num_nodes <= 1:
        for metric_flag in ["global", "local"]:
            for cutoff_type in ["one-sided", "two-sided"]:
                metric_key = f"{metric_flag}_{cutoff_type}"
                recalls[metric_key] = np.zeros_like(percentiles, dtype=float)
                weighted_recalls[metric_key] = np.zeros_like(percentiles, dtype=float)
        ks_stat = 0.0
        weighted_ks_stat = 0.0
    else:
        # Initial lookup table
        key2idx = {k: i for i, k in enumerate(keys)}

        # Filter to universe
        indices = [key2idx[key] for key in universe]
        universe_features = features[indices]
        universe_keys = keys[indices]

        # Update lookup table
        key2idx = {k: i for i, k in enumerate(universe_keys)}

        # Calculate global negative distances
        negative_distances_dict = {}
        negative_distances_dict["global"] = _get_negative_distances_global(
            universe_features, sample_size_global, distance_metric
        ).reshape(1, -1)

        # Calculate local negative distances
        gt_keys = list(gt_graph)
        keys2idx_local = {k: i for i, k in enumerate(gt_keys)}

        negative_distances_dict["local"] = get_negative_distances_local(
            keys=universe_keys,
            key2idx=key2idx,
            features=universe_features,
            gt_keys=gt_keys,
            sample_size=sample_size_local,
            distance_metric=distance_metric,
        )

        # Obtain ground truth distances
        keys1, keys2, positive_distances = _get_positive_distances(
            key2idx=key2idx,
            features=universe_features,
            gt_graph=gt_graph,
            distance_metric=distance_metric,
        )

        for metric_flag in ["global", "local"]:
            negative_distances = negative_distances_dict[metric_flag]

            # Get thresholds
            threshold_idx = (
                0 if metric_flag == "global" else [keys2idx_local[k] for k in keys1]
            )

            # Prepend -infinity for lower percentile calculation
            distances_lower = np.hstack(
                [np.full((negative_distances.shape[0], 1), -np.inf), negative_distances]
            )
            lower_thresholds = np.nanpercentile(
                distances_lower, 100 - percentiles, method="lower", axis=-1
            ).T

            valid_pos_mask = ~np.isnan(positive_distances)
            n_valid_pos = valid_pos_mask.sum()

            for cutoff_type in ["one-sided", "two-sided"]:
                tps = positive_distances[..., np.newaxis] <= lower_thresholds[threshold_idx, :]

                if cutoff_type == "two-sided":
                    distances_upper = np.hstack(
                        [
                            negative_distances,
                            np.full((negative_distances.shape[0], 1), np.inf),
                        ]
                    )
                    upper_thresholds = np.nanpercentile(
                        distances_upper, percentiles, method="higher", axis=-1
                    ).T
                    tps |= (
                        positive_distances[..., np.newaxis]
                        >= upper_thresholds[threshold_idx, :]
                    )

                metric_key = f"{metric_flag}_{cutoff_type}"
                recalls[metric_key] = (
                    np.sum(tps, axis=0) / n_valid_pos
                    if n_valid_pos > 0
                    else np.zeros_like(percentiles, dtype=float)
                )

                # Weighted recall
                positive_weights = _calculate_weights(gt_graph, keys1, keys2)
                valid_weighted_tps = tps * positive_weights[..., np.newaxis]
                valid_weight_sum = np.sum(positive_weights[valid_pos_mask])
                weighted_recalls[metric_key] = (
                    np.sum(valid_weighted_tps, axis=0) / valid_weight_sum
                    if valid_weight_sum > 0
                    else np.zeros_like(percentiles, dtype=float)
                )

        # KS test — filter out NaN distances
        negative_distances = negative_distances_dict["global"].reshape(-1)
        positive_weights = _calculate_weights(gt_graph, keys1, keys2)

        pos_valid = ~np.isnan(positive_distances)
        neg_valid = ~np.isnan(negative_distances)
        pos_dist_clean = positive_distances[pos_valid]
        neg_dist_clean = negative_distances[neg_valid]
        pos_weights_clean = positive_weights[pos_valid]
        negative_weights = np.ones(neg_dist_clean.shape[0])

        ks_stat = _get_ks_stat(pos_dist_clean, neg_dist_clean)
        weighted_ks_stat = _get_weighted_ks_stat(
            pos_dist_clean,
            neg_dist_clean,
            pos_weights_clean,
            negative_weights,
        )

    # Build output DataFrame
    metrics_df = pd.DataFrame(
        {
            "percentile": percentiles,
            "global_recall_one-sided": recalls["global_one-sided"],
            "global_weighted_recall_one-sided": weighted_recalls["global_one-sided"],
            "local_recall_one-sided": recalls["local_one-sided"],
            "local_weighted_recall_one-sided": weighted_recalls["local_one-sided"],
            "global_recall_two-sided": recalls["global_two-sided"],
            "global_weighted_recall_two-sided": weighted_recalls["global_two-sided"],
            "local_recall_two-sided": recalls["local_two-sided"],
            "local_weighted_recall_two-sided": weighted_recalls["local_two-sided"],
            "ks_stat": ks_stat,
            "weighted_ks_stat": weighted_ks_stat,
            "num_nodes": num_nodes,
            "num_edges": num_edges,
            "num_universe": num_universe,
        }
    )

    return metrics_df


def multivariate_metric_all(
    df: pd.DataFrame,
    pert_field: str,
    data_dir: str,
    eval_datasets: List[str],
    columns: Optional[List[str]] = None,
    distance_metric: str = "cosine",
    sample_size_global: int = 10000,
    sample_size_local: int = 1000,
    test_universe: str = "ground_truth",
) -> Dict[str, pd.DataFrame]:
    """Calculate multivariate metrics for multiple ground truth datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with perturbation-level aggregated features.
    pert_field : str
        Column containing perturbation identifiers.
    data_dir : str
        Directory containing ground truth files.
    eval_datasets : List[str]
        List of ground truth file names.
    columns : List[str], optional
        Feature columns. If None, uses all non-Metadata columns.
    distance_metric : str
        Distance metric ("cosine" or "sqeuclidean").
    sample_size_global : int
        Sample size for global negative distances.
    sample_size_local : int
        Sample size for local negative distances.
    test_universe : str
        Universe to use ("ground_truth" or "all").

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary mapping dataset names to performance metrics DataFrames.
    """
    if columns is None:
        columns = [c for c in df.columns if not c.startswith("Metadata_")]

    # Ensure unique perturbations
    if (df[pert_field].value_counts() != 1).any():
        print(f"Warning: df contains duplicate entries for {pert_field}. Aggregating by median.")
        df = df.groupby(pert_field, observed=True)[columns].median().reset_index()

    keys = df[pert_field].values
    unique_keys = np.unique(keys)
    features = df[columns].values
    features = np.ascontiguousarray(features)

    data_dir = pathlib.Path(data_dir)
    results = {}

    for eval_dataset in eval_datasets:
        ground_truth_path = data_dir / eval_dataset

        if "opentarget" in eval_dataset:
            ground_truth_graph, gt_sets = _opentarget_to_mol_graph(ground_truth_path, unique_keys)
        else:
            ground_truth_graph, gt_sets = _enrichr_to_gene_graph(ground_truth_path, unique_keys)

        # Determine universe
        if test_universe == "ground_truth":
            universe = np.array(list(ground_truth_graph))
        else:
            universe = unique_keys

        # Calculate performance metrics
        performance_df = get_performance_metrics(
            keys,
            features,
            universe=universe,
            gt_graph=ground_truth_graph,
            sample_size_global=sample_size_global,
            sample_size_local=sample_size_local,
            distance_metric=distance_metric,
            percentiles=np.arange(75, 100, 1),
        )

        dataset_key = eval_dataset.replace(".pkl", "").replace(".csv", "").replace(".txt", "")
        results[dataset_key] = performance_df

        if gt_sets:
            per_term_df = compute_per_term_metrics(
                df, pert_field=pert_field, gt_sets=gt_sets,
                columns=columns, distance_metric=distance_metric,
            )
            results[f"{dataset_key}__per_term"] = per_term_df

    return results


# =============================================================================
# Per-Term Metrics (ported from jump_qc multivariate_ng.compute_metrics)
# =============================================================================


def compute_per_term_metrics(
    df: pd.DataFrame,
    pert_field: str,
    gt_sets: Dict[str, List[str]],
    columns: Optional[List[str]] = None,
    distance_metric: str = "cosine",
    max_fprs: Optional[List[Optional[float]]] = None,
) -> pd.DataFrame:
    """Compute per-term AUROC and AP metrics for ground truth annotation sets.

    For each ground truth term (e.g. MoA class), computes:
    - Global AUROC (full and partial): scored against globally-calibrated negatives
    - Global AP: scored against globally-calibrated negatives
    - Local AP per anchor: scored against locally-calibrated negatives per compound,
      then averaged across anchors in the aggregated output

    This replicates the evaluation strategy from jump_qc's
    ``multivariate_ng.compute_metrics`` without requiring ``xarray`` or ``sparse``.

    Parameters
    ----------
    df : pd.DataFrame
        Perturbation-level aggregated features (one row per unique perturbation).
    pert_field : str
        Column containing perturbation identifiers.
    gt_sets : Dict[str, List[str]]
        Ground truth sets mapping term names to lists of perturbation IDs.
    columns : List[str], optional
        Feature columns. If None, uses all non-Metadata columns.
    distance_metric : str
        Distance metric for ``scipy.spatial.distance.pdist`` (default: "cosine").
    max_fprs : List[Optional[float]], optional
        Max FPR values for partial AUROC. ``None`` entries yield full AUROC.
        Defaults to ``[None, 0.1]``.

    Returns
    -------
    pd.DataFrame
        Per-term metrics with columns: term, metric, value, locality,
        max_fpr, anchor, n_compounds.
    """
    from itertools import combinations
    from scipy.spatial.distance import pdist, squareform
    from sklearn.metrics import average_precision_score, roc_auc_score

    if max_fprs is None:
        max_fprs = [None, 0.1]

    if columns is None:
        columns = [c for c in df.columns if not c.startswith("Metadata_")]

    if (df[pert_field].value_counts() != 1).any():
        df = df.groupby(pert_field, observed=True)[columns].median().reset_index()

    universe = set(df[pert_field].values)
    filtered_sets = {
        k: sorted(set(v) & universe)
        for k, v in gt_sets.items()
    }
    filtered_sets = {k: v for k, v in filtered_sets.items() if len(v) >= 2}

    all_perts = sorted(set().union(*filtered_sets.values()))
    df_sub = df[df[pert_field].isin(all_perts)].set_index(pert_field)
    df_sub = df_sub.loc[all_perts]

    n = len(all_perts)
    pert_to_idx = {p: i for i, p in enumerate(all_perts)}

    sim = 1.0 - squareform(
        pdist(df_sub[columns].values, metric=distance_metric).astype(np.float32)
    )

    global_adj = np.zeros((n, n), dtype=bool)
    set_adj = {}
    for term, perts in filtered_sets.items():
        idx = [pert_to_idx[p] for p in perts]
        term_adj = np.zeros((n, n), dtype=bool)
        for i, j in combinations(idx, 2):
            term_adj[i, j] = term_adj[j, i] = True
            global_adj[i, j] = global_adj[j, i] = True
        set_adj[term] = term_adj

    triu_mask = np.triu(np.ones((n, n), dtype=bool), k=1)

    global_y_true = global_adj[triu_mask]
    global_y_score = sim[triu_mask]
    global_neg_scores = global_y_score[~global_y_true]

    results = []

    for term, perts in filtered_sets.items():
        term_pos_mask = set_adj[term] & triu_mask
        pos_scores = sim[term_pos_mask]
        if len(pos_scores) == 0:
            continue

        y_true_combined = np.concatenate([
            np.ones(len(pos_scores)), np.zeros(len(global_neg_scores))
        ])
        y_score_combined = np.concatenate([pos_scores, global_neg_scores])

        for max_fpr in max_fprs:
            try:
                auroc = roc_auc_score(
                    y_true_combined, y_score_combined, max_fpr=max_fpr
                )
            except ValueError:
                auroc = np.nan
            results.append({
                "term": term, "metric": "AUROC", "value": auroc,
                "locality": "global", "max_fpr": max_fpr,
                "anchor": None, "n_compounds": len(perts),
            })

        try:
            ap = average_precision_score(y_true_combined, y_score_combined)
        except ValueError:
            ap = np.nan
        results.append({
            "term": term, "metric": "AP", "value": ap,
            "locality": "global", "max_fpr": None,
            "anchor": None, "n_compounds": len(perts),
        })

    for anchor_idx, anchor in enumerate(all_perts):
        local_mask = np.ones(n, dtype=bool)
        local_mask[anchor_idx] = False
        local_y_true = global_adj[anchor_idx, local_mask]
        local_y_score = sim[anchor_idx, local_mask]
        local_neg = local_y_score[~local_y_true]

        for term, perts in filtered_sets.items():
            if anchor not in perts:
                continue
            term_pos = set_adj[term][anchor_idx, local_mask]
            pos_local = local_y_score[term_pos]
            if len(pos_local) == 0:
                continue

            y_true_local = np.concatenate([
                np.ones(len(pos_local)), np.zeros(len(local_neg))
            ])
            y_score_local = np.concatenate([pos_local, local_neg])

            try:
                ap_local = average_precision_score(y_true_local, y_score_local)
            except ValueError:
                ap_local = np.nan
            results.append({
                "term": term, "metric": "AP", "value": ap_local,
                "locality": "local", "max_fpr": None,
                "anchor": anchor, "n_compounds": len(perts),
            })

    return pd.DataFrame(results)


# =============================================================================
# Phenosimilarity Scoring (Anchor-Based)
# =============================================================================


def _plate_matched_null_sampler(
    plate_ids: np.ndarray,
    plate_vehicle_dict: Dict[Any, pd.DataFrame],
    feature_columns: List[str],
    n_samples: int = 10000,
    seed: int = 0,
) -> np.ndarray:
    """Sample plate-matched null compound profiles from vehicle controls.

    For each plate a compound appears on, randomly samples `n_samples` vehicle
    control wells (with replacement), then averages across plates to produce
    synthetic null compound-level profiles.

    Parameters
    ----------
    plate_ids : np.ndarray
        Plate identifiers for the compound of interest.
    plate_vehicle_dict : Dict[Any, pd.DataFrame]
        Mapping from plate ID to a DataFrame of vehicle control wells.
    feature_columns : List[str]
        Feature columns to sample.
    n_samples : int, default 10000
        Number of null profiles to generate.
    seed : int, default 0
        Random seed.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples, n_features)`` with null profiles.
    """
    np.random.seed(seed)
    null_arr = np.zeros((len(plate_ids), n_samples, len(feature_columns)))
    for i, plate_id in enumerate(plate_ids):
        plate_vehicle_df = plate_vehicle_dict[plate_id]
        random_idx = np.random.choice(len(plate_vehicle_df), size=n_samples, replace=True)
        null_arr[i] = plate_vehicle_df[feature_columns].values[random_idx]
    return null_arr.mean(0)


def compute_phenosimilarity_scores(
    well_df: pd.DataFrame,
    anchor_compounds: List[str],
    perturbation_column: str = "Metadata_Perturbation",
    annotation_column: str = "Metadata_Remarks",
    negative_control_annotation: str = "VehicleControl",
    plate_column: str = "Metadata_Plate",
    n_samples: int = 10000,
    random_seed: int = 0,
    aggregate: bool = True,
) -> pd.DataFrame:
    """Compute phenosimilarity scores with plate-matched empirical null p-values.

    For each non-anchor, non-vehicle compound, computes the mean cosine
    similarity to a set of anchor compounds and tests significance against a
    plate-matched empirical null distribution of vehicle control similarities.

    Parameters
    ----------
    well_df : pd.DataFrame
        Well-level perturbation embeddings. Expected to contain annotation
        columns, a plate column, and numeric feature columns.
    anchor_compounds : List[str]
        Compound identifiers to use as anchors. Similarity of every other
        compound is measured against these anchors.
    perturbation_column : str, default "Metadata_Perturbation"
        Column containing compound/perturbation identifiers used for grouping
        and matching against *anchor_compounds*.
    annotation_column : str, default "Metadata_Remarks"
        Column whose values identify negative controls (matched against
        *negative_control_annotation*).
    negative_control_annotation : str, default "VehicleControl"
        Value in *annotation_column* that marks negative control wells.
    plate_column : str, default "Metadata_Plate"
        Column identifying the plate each well belongs to (used for
        plate-matched null sampling).
    n_samples : int, default 10000
        Number of null profiles to draw for the empirical null distribution.
    random_seed : int, default 0
        Random seed for reproducibility.
    aggregate : bool, default True
        If True (default), aggregate well-level data to compound-level by
        taking the mean across replicates before computing similarities.
        If False, compute similarities directly on each well-level row.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per scored compound (or per well if
        ``aggregate=False``) and columns: *perturbation_column*,
        *annotation_column*, ``phenosimilarity_score``,
        ``empnullplatematched_p_value``.
    """
    groupby_columns = [perturbation_column, annotation_column]

    np.random.seed(random_seed)

    non_feature_cols = set(groupby_columns) | {plate_column}
    non_feature_cols |= {c for c in well_df.columns if c.startswith("Metadata_")}
    feature_columns = [c for c in well_df.columns if c not in non_feature_cols]

    if aggregate:
        compound_df = (
            well_df
            .groupby(groupby_columns, observed=True)[feature_columns]
            .mean()
            .reset_index()
        )
    else:
        compound_df = well_df[groupby_columns + feature_columns].copy()

    anchors_df = compound_df[compound_df[perturbation_column].isin(anchor_compounds)]
    compounds_to_score_df = compound_df

    anchor_features = anchors_df[feature_columns].values
    scores = compute_fewshot_similarities(
        anchor_features,
        compounds_to_score_df[feature_columns].values,
        metric="cosine",
        mode="mean",
    )

    score_lookup = dict(zip(compounds_to_score_df[perturbation_column].values, scores))

    plate_vehicle_dict = {
        plate_id: plate_df
        for plate_id, plate_df in well_df[well_df[annotation_column] == negative_control_annotation].groupby(plate_column)
    }

    results_dict: Dict[str, list] = {col: [] for col in groupby_columns}
    results_dict["phenosimilarity_score"] = []
    results_dict["empnullplatematched_p_value"] = []

    for row in tqdm(
        compounds_to_score_df[groupby_columns].itertuples(index=False),
        total=len(compounds_to_score_df),
        desc="Computing phenosimilarity p-values",
    ):
        row_dict = row._asdict()
        compound_name = row_dict[perturbation_column]
        control_value = row_dict[annotation_column]

        if control_value == negative_control_annotation:
            continue

        perturb_score = score_lookup[compound_name]

        perturb_plate_ids = well_df[well_df[perturbation_column] == compound_name][plate_column].unique()
        null_profiles = _plate_matched_null_sampler(
            perturb_plate_ids,
            plate_vehicle_dict,
            feature_columns,
            n_samples=n_samples,
            seed=random_seed,
        )
        null_scores = compute_fewshot_similarities(
            anchor_features,
            null_profiles,
            metric="cosine",
            mode="mean",
        )

        p_value = ((perturb_score < null_scores).sum() + 1) / (n_samples + 1)

        for col in groupby_columns:
            results_dict[col].append(row_dict[col])
        results_dict["phenosimilarity_score"].append(perturb_score)
        results_dict["empnullplatematched_p_value"].append(p_value)

    return pd.DataFrame(results_dict)
