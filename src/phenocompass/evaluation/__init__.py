"""Evaluation metrics and callbacks."""

from phenocompass.evaluation.enrichment import compute_enrichment_analysis
from phenocompass.evaluation.map_config import (
    CP_VALIDATION_CONTROLS,
    CPValidationConfig,
    DEFAULT_INCLUDED_SOURCES,
    DMSO_JCP_ID,
    EvalConfig,
    JUMPFields,
    NEGATIVE_CONTROLS,
    PreprocessingConfig,
    TVNConfig,
    WAY2019_FEATURE_BLACKLIST,
)
from phenocompass.evaluation.map_evaluation import (
    compute_per_term_metrics,
    compute_phenosimilarity_scores,
    get_negative_distances_local,
    get_performance_metrics,
    knn_vote_accuracy,
    multivariate_metric_all,
    reproducibility_ks,
    univariate_metric_all_random,
    univariate_self_consistency,
)
from phenocompass.evaluation.map_pipelines import (
    aggregate_by_perturbation,
    load_gprobe_ground_truth,
    run_all_evaluations,
    run_cp_validation_pipeline,
    run_cp_validation_sweep,
    run_preprocessing_pipeline,
)
from phenocompass.evaluation.map_preprocessing import (
    SpheringTransform,
    apply_global_sphering,
    apply_pca,
    apply_tvn,
    clip_features,
    cosine_similarity_numba,
    drop_low_var_columns,
    drop_nan,
    drop_way2019_blacklist,
    euclidean_distance_numba,
    get_feature_columns,
    get_metadata_columns,
    get_negcon_perturbation_ids,
    get_random_index_pairs,
    inf_to_nan,
    platewise_zscore,
    zscore_by_controls,
)
from phenocompass.evaluation.metrics import (
    LinearProbeEvaluator,
    LinearTransferOnlineEvaluator,
    RetrievalOnlineEvaluator,
    compute_fewshot_similarities,
    compute_pairwise_tanimoto_similarity,
    compute_paired_tanimoto_similarity,
    fixed_set_retrieval_acc,
    fixed_set_retrieval_acc_batch_match,
    get_rankings,
    process_all_rankings,
)

__all__ = [
    # Callbacks
    "RetrievalOnlineEvaluator",
    "LinearTransferOnlineEvaluator",
    "LinearProbeEvaluator",
    # Retrieval metrics
    "get_rankings",
    "process_all_rankings",
    "fixed_set_retrieval_acc",
    "fixed_set_retrieval_acc_batch_match",
    # Similarity functions
    "compute_fewshot_similarities",
    "compute_pairwise_tanimoto_similarity",
    "compute_paired_tanimoto_similarity",
    # Enrichment analysis
    "compute_enrichment_analysis",
    # Map evaluation config
    "JUMPFields",
    "NEGATIVE_CONTROLS",
    "DMSO_JCP_ID",
    "DEFAULT_INCLUDED_SOURCES",
    "PreprocessingConfig",
    "TVNConfig",
    "EvalConfig",
    "CP_VALIDATION_CONTROLS",
    "CPValidationConfig",
    "WAY2019_FEATURE_BLACKLIST",
    # Distance functions
    "cosine_similarity_numba",
    "euclidean_distance_numba",
    "get_random_index_pairs",
    # Data utilities
    "get_metadata_columns",
    "get_feature_columns",
    "get_negcon_perturbation_ids",
    # Preprocessing functions
    "inf_to_nan",
    "drop_nan",
    "drop_way2019_blacklist",
    "drop_low_var_columns",
    "clip_features",
    "platewise_zscore",
    "zscore_by_controls",
    "apply_pca",
    "apply_global_sphering",
    "apply_tvn",
    "SpheringTransform",
    # Evaluation functions
    "knn_vote_accuracy",
    "reproducibility_ks",
    "univariate_self_consistency",
    "compute_per_term_metrics",
    "get_negative_distances_local",
    "get_performance_metrics",
    "multivariate_metric_all",
    "univariate_metric_all_random",
    "compute_phenosimilarity_scores",
    # Pipeline runners
    "run_preprocessing_pipeline",
    "run_all_evaluations",
    "aggregate_by_perturbation",
    "load_gprobe_ground_truth",
    "run_cp_validation_pipeline",
    "run_cp_validation_sweep",
]
