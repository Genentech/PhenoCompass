#!/usr/bin/env python
"""LOO baseline evaluation worker.

Computes leave-one-out partial AUROC for a single (seed, model_type, moa)
combination and writes the result to a parquet file. Designed to be invoked
many times (288 combinations total: 8 model configs x 6 seeds x 6 MoAs)
from a driver such as ``run_loo_baselines_local.sh`` (local) or
``submit_loo_baselines.sbatch`` (SLURM array). The aggregated outputs are
then loaded by ``preprocess_loo_comparisons.py``.

Example:
    python scripts/loo_baseline_worker.py \\
        --mode regression \\
        --seed 0 \\
        --model-type linear_regression \\
        --moa pi3k_mtor \\
        --embeddings-path final_data/embeddings/structure/gneprop_embeddings.parquet \\
        --method-name gneprop \\
        --method-display-name "GNEprop Embeddings" \\
        --data-dir data/loo_comparisons_cache \\
        --output-dir data/loo_comparisons_cache/partials
"""
import argparse
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsRegressor
from tqdm import tqdm

import phenocompass

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(phenocompass.__file__), "..", "..")
)
sys.path.insert(0, os.path.join(BASE_DIR, "notebooks"))

from phenocompass.data.dataset import FastH5DatasetJUMP
from phenocompass.data.datamodule import MultiInput_DataModule
from phenocompass.data.featurization import (
    canonicalize_smiles,
    load_jump_canonical_map,
    mol_to_mf,
)
from phenocompass.evaluation.metrics import (
    compute_fewshot_similarities,
    compute_pairwise_tanimoto_similarity,
)
from torch.utils.data import DataLoader

# ── Paths derived from repo root ────────────────────────────────────────
FINAL_DATA_DIR = os.path.join(BASE_DIR, "final_data")
H5_PATH = os.path.join(FINAL_DATA_DIR, "jump_map", "jump_dino_tvn.h5")
INDEX_FILE = os.path.join(FINAL_DATA_DIR, "jump_map", "scaffold_cluster_splits.pkl")
GNEPROP_PATH = os.path.join(FINAL_DATA_DIR, "embeddings", "structure", "gneprop_embeddings.parquet")
ANCHOR_PATH = os.path.join(FINAL_DATA_DIR, "compound_annotations", "anchor_compounds.tsv")
EXCLUSIONS_PATH = os.path.join(FINAL_DATA_DIR, "compound_annotations", "loo_background_exclusions.tsv")
JUMP_CANON_CACHE_PATH = os.path.join(FINAL_DATA_DIR, "compound_annotations", "jump_canonicalized_smiles.parquet")

MOA_TO_CLUSTER_IDX = {
    "PI3K/mTOR": 3,
    "HSP90": 6,
    "JAK/ROCK": 7,
    "HDAC": 8,
    "MEK": 10,
    "CDK": 11,
}
MOA_CLI_TO_NAME = {
    "pi3k_mtor": "PI3K/mTOR",
    "hsp90": "HSP90",
    "jak_rock": "JAK/ROCK",
    "hdac": "HDAC",
    "mek": "MEK",
    "cdk": "CDK",
}
MOA_CLI_KEYS = list(MOA_CLI_TO_NAME.keys())

T_MAX_VALUES = [0.15]
MAX_FPR_VALUES = [0.1]
FEWSHOT_MODE = "mean"


# ── Helper functions (from notebook Section 3) ──────────────────────────

def curate_compounds_leave_one_out(candidate_compounds, holdout_compound, tanimoto_sim_df, T_max):
    candidates = np.array(sorted(list(set(candidate_compounds) - {holdout_compound})))
    T_cond = tanimoto_sim_df.loc[holdout_compound][candidates].values < T_max
    return candidates[T_cond], holdout_compound


def run_loo_for_cluster(embeddings_df, cluster_compounds, background_compounds,
                        tanimoto_df, T_max, metric, mode, max_fpr_values,
                        method_name, method_display_name, moa, cluster_idx,
                        modality=None, seed=None):
    results = defaultdict(list)
    bg_in_index = [s for s in background_compounds if s in embeddings_df.index]
    background_emb = embeddings_df.loc[bg_in_index]

    for target_compound in cluster_compounds:
        if target_compound not in embeddings_df.index:
            continue

        filtered_train, _ = curate_compounds_leave_one_out(
            cluster_compounds, target_compound, tanimoto_df, T_max
        )
        filtered_train = [s for s in filtered_train if s in embeddings_df.index]

        if len(filtered_train) == 0:
            continue

        train_emb = embeddings_df.loc[filtered_train]
        target_emb = embeddings_df.loc[[target_compound]]

        if metric == "tanimoto":
            score_tgt = compute_fewshot_similarities(
                list(train_emb.values), list(target_emb.values), metric="tanimoto", mode=mode
            ).squeeze()
            score_bg = compute_fewshot_similarities(
                list(train_emb.values), list(background_emb.values), metric="tanimoto", mode=mode
            )
        else:
            score_tgt = compute_fewshot_similarities(
                train_emb.values, target_emb.values, metric=metric, mode=mode
            ).squeeze()
            score_bg = compute_fewshot_similarities(
                train_emb.values, background_emb.values, metric=metric, mode=mode
            )

        for max_fpr in max_fpr_values:
            labels = [1] + [0] * len(score_bg)
            scores = np.array([score_tgt] + list(score_bg))
            p_auroc = roc_auc_score(labels, scores, max_fpr=max_fpr)

            results["cluster_idx"].append(cluster_idx)
            results["seed"].append(seed)
            results["T_max"].append(T_max)
            results["method"].append(method_name)
            results["modality"].append(modality)
            results["target"].append(target_compound)
            results["max_fpr"].append(max_fpr)
            results["partial_auroc"].append(p_auroc)
            results["moa"].append(moa)
            results["method_name"].append(method_display_name)

    return pd.DataFrame(results)


# ── Precomputed kNN classifier (matches moa_classifier.py) ─────────────

NEIGHBOR_FILE_PATTERNS = {
    "gneprop": "pairwise_gneprop_euclidean",
    "mf": "pairwise_mf_cosine_similarity",
}


def get_neighbor_indices(neighbor_indices_df, train_indices, test_indices, n_neighbors=10):
    neighbor_indices_arr = np.zeros((len(test_indices), n_neighbors), dtype=int)
    for i, test_index in enumerate(test_indices):
        neighbor_indices = [idx for idx in neighbor_indices_df.iloc[test_index].values
                            if idx in train_indices]
        neighbor_indices_arr[i] = neighbor_indices[-n_neighbors:]
    return neighbor_indices_arr


class kNNPredictorFromNeighborsTable:
    def __init__(self, neighbor_indices_df, n_neighbors=10):
        self.n_neighbors = n_neighbors
        self.neighbor_indices_df = neighbor_indices_df

    def fit(self, X_train_indices, y_train):
        self.X_train_indices = set(X_train_indices)
        self.y_train = pd.DataFrame(y_train, index=X_train_indices)

    def predict(self, X_indices):
        neighbor_indices_arr = get_neighbor_indices(
            self.neighbor_indices_df, self.X_train_indices,
            X_indices, n_neighbors=self.n_neighbors)
        y_pred = np.array([
            self.y_train.loc[neighbor_indices_row].values.squeeze()
            for neighbor_indices_row in neighbor_indices_arr
        ]).mean(1)
        return y_pred

    def predict_proba(self, X_indices):
        y_pred = self.predict(X_indices)
        return np.array([1 - y_pred, y_pred]).T


# ── Data loading ────────────────────────────────────────────────────────

def load_shared_data(args):
    print("Loading JUMP dataset...")
    dataset = FastH5DatasetJUMP(
        H5_PATH,
        avg_morph_feat=True,
        index_predetermined_file=INDEX_FILE,
        ohe_batch=False,
        struct_ipt="fixed",
        fixed_struct_data=GNEPROP_PATH,
    )
    print(f"  Dataset: {len(dataset)} compounds")

    print(f"Loading user embeddings from {args.embeddings_path}...")
    user_embeddings_df = pd.read_parquet(args.embeddings_path)
    print(f"  User embeddings: {user_embeddings_df.shape}")

    print("Loading anchor compounds and exclusions...")
    anchor_df = pd.read_csv(ANCHOR_PATH, sep="\t")
    background_exclusions_df = pd.read_csv(EXCLUSIONS_PATH, sep="\t")

    use_canon = getattr(args, "use_tautomer_canonicalization", False)
    if use_canon:
        print("  Tautomer canonicalization: ENABLED")
        smiles_to_canon = load_jump_canonical_map(JUMP_CANON_CACHE_PATH)
        canon_excluded = set(
            smiles_to_canon.get(s, canonicalize_smiles(s))
            for s in background_exclusions_df["SMILES"].values
        )
        jump_background_compounds = [
            s for s in dataset.ids
            if smiles_to_canon.get(s, s) not in canon_excluded
        ]
    else:
        all_excluded_smiles = set(background_exclusions_df["SMILES"].values)
        jump_background_compounds = [s for s in dataset.ids if s not in all_excluded_smiles]
    print(f"  Anchors: {len(anchor_df)}, Background: {len(jump_background_compounds)}")

    moa = MOA_CLI_TO_NAME[args.moa]
    print(f"Computing Tanimoto matrix for {moa}...")
    cluster_smiles = sorted(anchor_df[anchor_df["cluster_moa"] == moa]["canonical_SMILES"].values)

    if use_canon:
        from rdkit.Chem import AllChem
        fpgen = AllChem.GetMorganGenerator(radius=3, fpSize=2048)
        mf_featurizer = fpgen.GetCountFingerprint
        cluster_fps = [mol_to_mf(canonicalize_smiles(s), mf_featurizer) for s in cluster_smiles]
        valid_smiles = list(cluster_smiles)
    else:
        print("Loading Morgan fingerprints...")
        mf_cache_path = os.path.join(FINAL_DATA_DIR, "embeddings", "structure", "morgan_fingerprints.pkl")
        with open(mf_cache_path, "rb") as f:
            smiles_to_fingerprint = pickle.load(f)
        print(f"  Loaded: {len(smiles_to_fingerprint)}")
        cluster_fps = [smiles_to_fingerprint[s] for s in cluster_smiles if s in smiles_to_fingerprint]
        valid_smiles = [s for s in cluster_smiles if s in smiles_to_fingerprint]

    T_matrix = compute_pairwise_tanimoto_similarity(cluster_fps, cluster_fps)
    tanimoto_df = pd.DataFrame(T_matrix, index=valid_smiles, columns=valid_smiles)

    nn_df = None
    smiles_idx_dict = None
    if args.model_type == "knn_classifier":
        if not args.neighbor_dir:
            raise ValueError("--neighbor-dir is required for knn_classifier")
        smiles_idx_dict = {mol: i for i, mol in enumerate(dataset.ids)}
        pattern = args.neighbor_prefix or NEIGHBOR_FILE_PATTERNS.get(args.method_name)
        if pattern is None:
            raise ValueError(
                f"No neighbor file pattern for method '{args.method_name}'. "
                f"Provide --neighbor-prefix or use a known method name: "
                f"{list(NEIGHBOR_FILE_PATTERNS.keys())}"
            )
        import glob as _glob
        chunk_files = sorted(_glob.glob(
            os.path.join(args.neighbor_dir, f"{pattern}.*.indices.csv")
        ))
        if not chunk_files:
            raise FileNotFoundError(
                f"No neighbor index files matching '{pattern}.*.indices.csv' "
                f"in {args.neighbor_dir}"
            )
        print(f"Loading precomputed neighbor indices ({pattern}, {len(chunk_files)} chunks)...")
        nn_df = pd.concat([pd.read_csv(f, index_col=0) for f in chunk_files])
        print(f"  Neighbor index table: {nn_df.shape}")

    return dataset, user_embeddings_df, anchor_df, jump_background_compounds, tanimoto_df, nn_df, smiles_idx_dict


# ── Regression mode ─────────────────────────────────────────────────────

def run_regression(args, dataset, user_embeddings_df, anchor_df,
                   jump_background_compounds, tanimoto_df):
    moa = MOA_CLI_TO_NAME[args.moa]
    print(f"\n=== Regression: seed={args.seed}, model_type={args.model_type}, moa={moa} ===")

    morph_cache_path = os.path.join(args.data_dir, "morph_embeddings.parquet")
    if os.path.exists(morph_cache_path):
        morph_embeddings_df = pd.read_parquet(morph_cache_path)
        print(f"Loaded cached morphology embeddings: {morph_embeddings_df.shape}")
    else:
        print("Extracting morphology embeddings...")
        dataloader = DataLoader(dataset, batch_size=5096, pin_memory=False, drop_last=False, shuffle=False)
        morph_embeddings = np.concatenate([
            batch["inputs"]["morph"].numpy() for batch in tqdm(dataloader, desc="Extracting morphology")
        ])
        morph_embeddings_df = pd.DataFrame(morph_embeddings, index=dataset.ids)
        morph_embeddings_df.to_parquet(morph_cache_path)
        print(f"Extracted and cached: {morph_embeddings_df.shape}")

    mod = MultiInput_DataModule(
        dataset,
        split_sizes=(0.90, 0.05, 0.05),
        split_type="index_predetermined",
        holdout_notion="inchi",
        seed=args.seed,
        verbose=False,
    )
    mod.setup("fit")

    train_ids = mod.train_dataset.ids
    train_smiles_in_emb = [
        s for s in train_ids
        if s in user_embeddings_df.index and s in morph_embeddings_df.index
    ]

    train_X = user_embeddings_df.loc[train_smiles_in_emb].values
    train_y = morph_embeddings_df.loc[train_smiles_in_emb].values

    all_smiles = [s for s in dataset.ids if s in user_embeddings_df.index]
    all_X = user_embeddings_df.loc[all_smiles].values

    if args.model_type == "linear_regression":
        model = LinearRegression()
        display_name = f"{args.method_display_name} + Linear Regression"
    elif args.model_type == "knn_regression":
        knn_metric = args.knn_metric or ("cosine" if args.method_name == "mf" else "euclidean")
        model = KNeighborsRegressor(n_neighbors=10, metric=knn_metric)
        display_name = f"{args.method_display_name} + kNN Regression"
    else:
        raise ValueError(f"Unknown regression model type: {args.model_type}")

    method_key = f"{args.method_name}_{args.model_type}_{args.seed}"

    pred_cache = os.path.join(args.data_dir, f"pred_morph_{method_key}.parquet")
    if os.path.exists(pred_cache):
        pred_morph_df = pd.read_parquet(pred_cache)
        print(f"Loaded cached predictions: {pred_morph_df.shape}")
    else:
        print(f"Training {method_key}...")
        model.fit(train_X, train_y)
        pred_morph = model.predict(all_X)
        pred_morph_df = pd.DataFrame(pred_morph, index=all_smiles)
        pred_morph_df.to_parquet(pred_cache)
        print(f"Trained and cached: {pred_morph_df.shape}")

    cluster_smiles = sorted(
        anchor_df[anchor_df["cluster_moa"] == moa]["canonical_SMILES"].values
    )
    regression_parts = []
    for T_max in T_MAX_VALUES:
        df = run_loo_for_cluster(
            pred_morph_df, cluster_smiles, jump_background_compounds,
            tanimoto_df, T_max, "cosine", FEWSHOT_MODE,
            MAX_FPR_VALUES, method_key, display_name, moa,
            MOA_TO_CLUSTER_IDX[moa], seed=args.seed,
        )
        regression_parts.append(df)

    results = pd.concat(regression_parts, ignore_index=True)
    return results


# ── Classifier mode ─────────────────────────────────────────────────────

def run_classifier(args, dataset, user_embeddings_df, anchor_df,
                   jump_background_compounds, tanimoto_df, nn_df=None,
                   smiles_idx_dict=None):
    moa = MOA_CLI_TO_NAME[args.moa]
    print(f"\n=== Classifier: seed={args.seed}, model_type={args.model_type}, moa={moa} ===")

    mod = MultiInput_DataModule(
        dataset,
        split_sizes=(0.90, 0.05, 0.05),
        split_type="index_predetermined",
        holdout_notion="inchi",
        seed=args.seed,
        verbose=False,
    )
    mod.setup("fit")

    train_ids = [s for s in mod.train_dataset.ids if s in user_embeddings_df.index]
    bg_smiles = [s for s in jump_background_compounds if s in user_embeddings_df.index]
    bg_X = user_embeddings_df.loc[bg_smiles].values

    if args.model_type == "logistic_regression":
        display_name = f"{args.method_display_name} + Logistic Regression"
    elif args.model_type == "knn_classifier":
        display_name = f"{args.method_display_name} + kNN Classifier"
    else:
        raise ValueError(f"Unknown classifier model type: {args.model_type}")

    method_key = f"{args.method_name}_{args.model_type}_{args.seed}"

    cluster_idx = MOA_TO_CLUSTER_IDX[moa]
    cluster_smiles = sorted(
        anchor_df[anchor_df["cluster_moa"] == moa]["canonical_SMILES"].values
    )
    valid_cluster = [s for s in cluster_smiles if s in user_embeddings_df.index]
    print(f"  {moa}: {len(valid_cluster)} compounds")

    classifier_parts = []
    for T_max in T_MAX_VALUES:
        results = defaultdict(list)

        for target_compound in tqdm(valid_cluster, desc=f"{moa} T_max={T_max}"):
            filtered_train, _ = curate_compounds_leave_one_out(
                valid_cluster, target_compound,
                tanimoto_df, T_max,
            )
            filtered_train = [s for s in filtered_train if s in user_embeddings_df.index]

            if len(filtered_train) == 0:
                continue

            clf_train_smiles = list(train_ids) + list(filtered_train)
            clf_train_y = np.array(
                [0] * len(train_ids) + [1] * len(filtered_train)
            )

            if args.model_type == "logistic_regression":
                clf_train_X = user_embeddings_df.loc[clf_train_smiles].values
                clf = LogisticRegression(class_weight="balanced", max_iter=1000)
                clf.fit(clf_train_X, clf_train_y)

                target_X = user_embeddings_df.loc[[target_compound]].values
                target_prob = clf.predict_proba(target_X)[:, 1].squeeze()
                bg_probs = clf.predict_proba(bg_X)[:, 1]

            elif args.model_type == "knn_classifier":
                clf_train_indices = [smiles_idx_dict[s] for s in clf_train_smiles]
                test_indices = ([smiles_idx_dict[target_compound]]
                                + [smiles_idx_dict[s] for s in bg_smiles])
                clf = kNNPredictorFromNeighborsTable(nn_df, n_neighbors=100)
                clf.fit(clf_train_indices, clf_train_y)
                y_pred = clf.predict_proba(test_indices)[:, 1]
                target_prob = y_pred[0]
                bg_probs = y_pred[1:]

            for max_fpr in MAX_FPR_VALUES:
                labels = [1] + [0] * len(bg_probs)
                scores = [target_prob] + list(bg_probs)
                p_auroc = roc_auc_score(labels, scores, max_fpr=max_fpr)

                results["cluster_idx"].append(cluster_idx)
                results["seed"].append(args.seed)
                results["T_max"].append(T_max)
                results["method"].append(method_key)
                results["modality"].append(None)
                results["target"].append(target_compound)
                results["max_fpr"].append(max_fpr)
                results["partial_auroc"].append(p_auroc)
                results["moa"].append(moa)
                results["method_name"].append(display_name)

        classifier_parts.append(pd.DataFrame(results))

    results = pd.concat(classifier_parts, ignore_index=True)
    return results


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LOO baseline evaluation worker")
    parser.add_argument("--mode", required=True, choices=["regression", "classifier"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--model-type", required=True,
                        choices=["linear_regression", "knn_regression",
                                 "logistic_regression", "knn_classifier"])
    parser.add_argument("--moa", required=True, choices=MOA_CLI_KEYS)
    parser.add_argument("--embeddings-path", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--method-display-name", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--neighbor-dir", type=str, default=None,
                        help="Directory with precomputed neighbor index CSVs (required for knn_classifier)")
    parser.add_argument("--neighbor-prefix", type=str, default=None,
                        help="Filename prefix for neighbor CSVs (e.g., 'pairwise_gneprop_euclidean'). "
                             "Auto-detected from method-name if not provided.")
    parser.add_argument("--knn-metric", type=str, default=None,
                        help="Distance metric for kNN regressor (default: euclidean for gneprop, cosine for mf)")
    parser.add_argument("--use-tautomer-canonicalization", action="store_true",
                        help="Canonicalize SMILES before computing Morgan fingerprints "
                             "for Tanimoto similarity (embedding lookups stay on original SMILES)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    (dataset, user_embeddings_df, anchor_df, jump_background_compounds,
     tanimoto_df, nn_df, smiles_idx_dict) = load_shared_data(args)

    if args.mode == "regression":
        results = run_regression(
            args, dataset, user_embeddings_df, anchor_df,
            jump_background_compounds, tanimoto_df,
        )
    else:
        results = run_classifier(
            args, dataset, user_embeddings_df, anchor_df,
            jump_background_compounds, tanimoto_df, nn_df, smiles_idx_dict,
        )

    out_path = os.path.join(
        args.output_dir,
        f"{args.mode}_{args.method_name}_{args.model_type}_seed{args.seed}_{args.moa}.parquet",
    )
    results.to_parquet(out_path, index=False)
    print(f"\nSaved {len(results)} rows → {out_path}")


if __name__ == "__main__":
    main()
