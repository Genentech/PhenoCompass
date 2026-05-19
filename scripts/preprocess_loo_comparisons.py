#!/usr/bin/env python
"""Reproduce leave-one-out (LOO) evaluation results for Figure 3.

Generates the six per-MoA LOO result TSVs consumed by
``notebooks/Fig3_jump_simulated_vs.ipynb``:

    pi3k_mtor.loo_results.tsv
    hsp90.loo_results.tsv
    jak_rock.loo_results.tsv
    hdac.loo_results.tsv
    mapk.loo_results.tsv
    cdk.loo_results.tsv

Pipeline:

    1. Load anchor compounds, background exclusions, JUMP compound universe.
    2. Build Morgan fingerprints (with optional tautomer canonicalization).
    3. Compute per-cluster Tanimoto similarity matrices.
    4. Run direct-similarity LOO for GNEprop embeddings and Morgan fingerprints.
    5. Run PhenoCompass cross-modal ensemble LOO across
       6 seeds x 2 model types x 3 modalities (joint + struct + morph).
    6. Optionally load regression/classifier baseline partials produced by
       ``scripts/loo_baseline_worker.py``.
    7. Consolidate and write 6 per-MoA TSVs.

Baseline partials (Step 6) are produced by a separate worker script. To
generate them before running this script, run locally::

    bash scripts/run_loo_baselines_local.sh

or, on a SLURM cluster::

    sbatch scripts/submit_loo_baselines.sbatch

Usage::

    python scripts/preprocess_loo_comparisons.py \\
        [--output-dir final_data/loo_comparisons] \\
        [--cache-dir data/loo_comparisons_cache] \\
        [--partials-dir data/loo_comparisons_cache/partials] \\
        [--no-tautomer-canonicalization] \\
        [--skip-phenocompass] \\
        [--force-recompute]
"""
import argparse
import glob
import os
import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from phenocompass.data.dataset import FastH5DatasetJUMP
from phenocompass.data.featurization import (
    load_jump_canonical_map,
    mol_to_mf,
)
from phenocompass.evaluation.metrics import (
    compute_fewshot_similarities,
    compute_pairwise_tanimoto_similarity,
)


# ── Constants (match the manuscript Figure 3 evaluation) ─────────────────────

MOA_TO_CLUSTER_IDX = {
    "PI3K/mTOR": 3,
    "HSP90": 6,
    "JAK/ROCK": 7,
    "HDAC": 8,
    "MEK": 10,
    "CDK": 11,
}
MOA_DISPLAY_MAP = {
    "PI3K/mTOR": "pi3k_mtor",
    "HSP90": "hsp90",
    "JAK/ROCK": "jak_rock",
    "HDAC": "hdac",
    "MEK": "mapk",
    "CDK": "cdk",
}
MOA_LIST = list(MOA_DISPLAY_MAP.keys())

T_MAX_VALUES = [0.15]
MAX_FPR_VALUES = [0.01, 0.05, 0.1, 0.2, None]
MIN_TRAINING_COMPOUNDS = 3  # exclude held-out compounds with fewer than this many
                            # training neighbours after T_max filtering
FEWSHOT_MODE = "mean"

# Direct-similarity LOO method labels
METHOD_NAME = "gneprop"
METHOD_DISPLAY_NAME = "GNEprop Embeddings"
SIMILARITY_METRIC = "euclidean"
MF_METHOD_NAME = "fingerprint"
MF_DISPLAY_NAME = "Morgan Fingerprints"

# PhenoCompass co-embedding ensemble configuration
N_SEEDS = 6
MODEL_TYPES = ["combined_fp", "consistency_reg"]
MODALITIES = ["struct", "morph", "joint"]


# ── LOO helpers ──────────────────────────────────────────────────────────────


def curate_compounds_leave_one_out(candidate_compounds, holdout_compound, tanimoto_sim_df, T_max):
    """Remove holdout and filter candidates by Tanimoto < T_max."""
    candidates = np.array(sorted(list(set(candidate_compounds) - {holdout_compound})))
    T_cond = tanimoto_sim_df.loc[holdout_compound][candidates].values < T_max
    return candidates[T_cond], holdout_compound


def run_loo_for_cluster(embeddings_df, cluster_compounds, background_compounds,
                        tanimoto_df, T_max, metric, mode, max_fpr_values,
                        method_name, method_display_name, moa, cluster_idx,
                        modality=None, seed=None):
    """Run LOO evaluation for one cluster/T_max combination."""
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


def run_phenocompass_loo_for_cluster(coembeddings, anchor_modality, cluster_compounds,
                                     background_compounds, tanimoto_df, T_max,
                                     max_fpr_values, moa, cluster_idx, seed):
    """Run PhenoCompass ensemble LOO for one cluster/seed/modality combination.

    Cross-modal: anchors use ``anchor_modality``; target + background use ``struct``.
    Scores are averaged across both model types in ``MODEL_TYPES``.
    """
    results = defaultdict(list)
    bg_in_index = [s for s in background_compounds
                   if s in coembeddings[("combined_fp", seed, "struct")].index]

    for target_compound in cluster_compounds:
        if target_compound not in coembeddings[("combined_fp", seed, "struct")].index:
            continue

        filtered_train, _ = curate_compounds_leave_one_out(
            cluster_compounds, target_compound, tanimoto_df, T_max
        )
        filtered_train = [s for s in filtered_train
                          if s in coembeddings[("combined_fp", seed, anchor_modality)].index]

        if len(filtered_train) == 0:
            continue

        model_scores_tgt = []
        model_scores_bg = []
        for model_type in MODEL_TYPES:
            train_emb = coembeddings[(model_type, seed, anchor_modality)].loc[filtered_train].values
            target_emb = coembeddings[(model_type, seed, "struct")].loc[[target_compound]].values
            bg_emb = coembeddings[(model_type, seed, "struct")].loc[bg_in_index].values

            score_tgt = cosine_similarity(train_emb, target_emb).mean(axis=0).squeeze()
            score_bg = cosine_similarity(train_emb, bg_emb).mean(axis=0)

            model_scores_tgt.append(score_tgt)
            model_scores_bg.append(score_bg)

        ensemble_score_tgt = np.mean(model_scores_tgt)
        ensemble_score_bg = np.mean(model_scores_bg, axis=0)

        for max_fpr in max_fpr_values:
            labels = [1] + [0] * len(ensemble_score_bg)
            scores = np.array([ensemble_score_tgt] + list(ensemble_score_bg))
            p_auroc = roc_auc_score(labels, scores, max_fpr=max_fpr)

            results["cluster_idx"].append(cluster_idx)
            results["seed"].append(seed)
            results["T_max"].append(T_max)
            results["method"].append(f"ensemble_cross_model_{anchor_modality}_{seed}")
            results["modality"].append(anchor_modality)
            results["target"].append(target_compound)
            results["max_fpr"].append(max_fpr)
            results["partial_auroc"].append(p_auroc)
            results["moa"].append(moa)
            results["method_name"].append("PhenoCompass")

    return pd.DataFrame(results)


# ── Pipeline stages ───────────────────────────────────────────────────────────


def load_data(args, use_tautomer):
    """Load anchors, background, fingerprints, and per-cluster Tanimoto matrices.

    Returns a dict with keys:
        anchor_df, dataset_ids, jump_background_compounds,
        smiles_to_canon (or None), smiles_to_fingerprint (or None),
        cluster_tanimoto_dfs, user_embeddings_df, mf_embeddings_series
    """
    final_data_dir = args.final_data_dir

    # JUMP compound universe via FastH5 dataset
    print(f"Loading JUMP dataset from: {args.h5_path}")
    dataset = FastH5DatasetJUMP(
        args.h5_path,
        avg_morph_feat=True,
        index_predetermined_file=args.index_file,
        ohe_batch=False,
        struct_ipt="fixed",
        fixed_struct_data=args.gneprop_path,
    )
    dataset_ids = list(dataset.ids)
    print(f"  {len(dataset_ids)} compounds")

    # Anchor and exclusion lists
    anchor_df = pd.read_csv(
        os.path.join(final_data_dir, "compound_annotations", "anchor_compounds.tsv"),
        sep="\t",
    )
    background_exclusions_df = pd.read_csv(
        os.path.join(final_data_dir, "compound_annotations", "loo_background_exclusions.tsv"),
        sep="\t",
    )
    print(f"  Anchor compounds: {len(anchor_df)}")
    print(f"  Background exclusions: {len(background_exclusions_df)}")

    # Tautomer canonicalization map
    if use_tautomer:
        canon_path = os.path.join(
            final_data_dir, "compound_annotations", "jump_canonicalized_smiles.parquet"
        )
        smiles_to_canon = load_jump_canonical_map(canon_path)
        canon_excluded = set(
            smiles_to_canon.get(s, s) for s in background_exclusions_df["SMILES"].values
        )
        jump_background_compounds = [
            s for s in dataset_ids if smiles_to_canon.get(s, s) not in canon_excluded
        ]
    else:
        smiles_to_canon = None
        all_excluded_smiles = set(background_exclusions_df["SMILES"].values)
        jump_background_compounds = [s for s in dataset_ids if s not in all_excluded_smiles]
    print(f"  Background compounds: {len(jump_background_compounds)}")

    # Morgan fingerprints (for direct-similarity Morgan baseline + Tanimoto matrices)
    fpgen = AllChem.GetMorganGenerator(radius=3, fpSize=2048)
    mf_featurizer = fpgen.GetCountFingerprint

    if not use_tautomer:
        mf_cache_path = os.path.join(
            final_data_dir, "embeddings", "structure", "morgan_fingerprints.pkl"
        )
        with open(mf_cache_path, "rb") as f:
            smiles_to_fingerprint = pickle.load(f)
        print(f"  Loaded pre-computed fingerprints: {len(smiles_to_fingerprint)}")
    else:
        smiles_to_fingerprint = None
        print("  Tautomer canonicalization enabled — fingerprints computed on the fly")

    # Per-cluster Tanimoto matrices
    cluster_tanimoto_dfs = {}
    for moa in MOA_LIST:
        cluster_smiles = sorted(anchor_df[anchor_df["cluster_moa"] == moa]["canonical_SMILES"].values)
        if use_tautomer:
            cluster_fps = [mol_to_mf(smiles_to_canon.get(s, s), mf_featurizer) for s in cluster_smiles]
            valid_smiles = list(cluster_smiles)
        else:
            cluster_fps = [smiles_to_fingerprint[s] for s in cluster_smiles if s in smiles_to_fingerprint]
            valid_smiles = [s for s in cluster_smiles if s in smiles_to_fingerprint]
        T_matrix = compute_pairwise_tanimoto_similarity(cluster_fps, cluster_fps)
        cluster_tanimoto_dfs[moa] = pd.DataFrame(T_matrix, index=valid_smiles, columns=valid_smiles)
        print(f"  Tanimoto matrix [{moa}]: {len(valid_smiles)} compounds")

    # User (GNEprop) embeddings
    user_embeddings_df = pd.read_parquet(args.gneprop_path)
    print(f"  GNEprop embeddings: {user_embeddings_df.shape}")

    # Morgan fingerprint series (used in MF direct-similarity LOO)
    if use_tautomer:
        mf_embeddings_series = pd.Series({
            orig: mol_to_mf(smiles_to_canon.get(orig, orig), mf_featurizer)
            for orig in tqdm(dataset_ids, desc="Computing canonicalized Morgan FPs")
        })
    else:
        mf_embeddings_series = pd.Series(smiles_to_fingerprint)

    return {
        "anchor_df": anchor_df,
        "dataset_ids": dataset_ids,
        "jump_background_compounds": jump_background_compounds,
        "smiles_to_canon": smiles_to_canon,
        "smiles_to_fingerprint": smiles_to_fingerprint,
        "cluster_tanimoto_dfs": cluster_tanimoto_dfs,
        "user_embeddings_df": user_embeddings_df,
        "mf_embeddings_series": mf_embeddings_series,
    }


def run_direct_similarity_loo(args, data):
    """Run direct-similarity LOO for GNEprop and Morgan fingerprints."""
    cache_dir = args.cache_dir
    results = []

    # GNEprop direct similarity
    gneprop_cache = os.path.join(cache_dir, f"direct_similarity_{METHOD_NAME}.parquet")
    if os.path.exists(gneprop_cache) and not args.force_recompute:
        df = pd.read_parquet(gneprop_cache)
        print(f"Loaded cached GNEprop direct LOO: {len(df)} rows")
    else:
        parts = []
        for moa in tqdm(MOA_LIST, desc="GNEprop direct LOO"):
            cluster_smiles = sorted(
                data["anchor_df"][data["anchor_df"]["cluster_moa"] == moa]["canonical_SMILES"].values
            )
            for T_max in T_MAX_VALUES:
                parts.append(run_loo_for_cluster(
                    data["user_embeddings_df"], cluster_smiles, data["jump_background_compounds"],
                    data["cluster_tanimoto_dfs"][moa], T_max, SIMILARITY_METRIC, FEWSHOT_MODE,
                    MAX_FPR_VALUES, METHOD_NAME, METHOD_DISPLAY_NAME, moa,
                    MOA_TO_CLUSTER_IDX[moa],
                ))
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(gneprop_cache, index=False)
        print(f"GNEprop direct LOO complete: {len(df)} rows -> {gneprop_cache}")
    results.append(df)

    # Morgan fingerprint direct similarity
    mf_cache = os.path.join(cache_dir, f"direct_similarity_{MF_METHOD_NAME}.parquet")
    if os.path.exists(mf_cache) and not args.force_recompute:
        df = pd.read_parquet(mf_cache)
        print(f"Loaded cached MF direct LOO: {len(df)} rows")
    else:
        parts = []
        for moa in tqdm(MOA_LIST, desc="MF direct LOO"):
            cluster_smiles = sorted(
                data["anchor_df"][data["anchor_df"]["cluster_moa"] == moa]["canonical_SMILES"].values
            )
            for T_max in T_MAX_VALUES:
                parts.append(run_loo_for_cluster(
                    data["mf_embeddings_series"], cluster_smiles, data["jump_background_compounds"],
                    data["cluster_tanimoto_dfs"][moa], T_max, "tanimoto", FEWSHOT_MODE,
                    MAX_FPR_VALUES, MF_METHOD_NAME, MF_DISPLAY_NAME, moa,
                    MOA_TO_CLUSTER_IDX[moa],
                ))
        df = pd.concat(parts, ignore_index=True)
        df.to_parquet(mf_cache, index=False)
        print(f"MF direct LOO complete: {len(df)} rows -> {mf_cache}")
    results.append(df)

    return results


def load_coembeddings(coembedding_dir):
    """Load all 36 PhenoCompass co-embedding parquets into a dict keyed by (model_type, seed, modality)."""
    coembeddings = {}
    for model_type in MODEL_TYPES:
        for seed in range(N_SEEDS):
            for modality in MODALITIES:
                fname = f"{model_type}_seed{seed}_{modality}.parquet"
                df = pd.read_parquet(os.path.join(coembedding_dir, fname))
                coembeddings[(model_type, seed, modality)] = df
    print(f"Loaded {len(coembeddings)} co-embedding DataFrames")
    return coembeddings


def run_phenocompass_loo(args, data):
    """Run PhenoCompass cross-modal LOO (joint + all modalities)."""
    cache_dir = args.cache_dir
    coembedding_dir = os.path.join(args.final_data_dir, "embeddings", "coembedding")

    joint_cache = os.path.join(cache_dir, "direct_similarity_phenocompass_joint.parquet")
    all_cache = os.path.join(cache_dir, "direct_similarity_phenocompass_all_modalities.parquet")

    need_joint = args.force_recompute or not os.path.exists(joint_cache)
    need_all = args.force_recompute or not os.path.exists(all_cache)

    if need_joint or need_all:
        coembeddings = load_coembeddings(coembedding_dir)

    # Joint modality (main results)
    if need_joint:
        parts = []
        for moa in tqdm(MOA_LIST, desc="PhenoCompass LOO (joint)"):
            cluster_smiles = sorted(
                data["anchor_df"][data["anchor_df"]["cluster_moa"] == moa]["canonical_SMILES"].values
            )
            for T_max in T_MAX_VALUES:
                for seed in range(N_SEEDS):
                    parts.append(run_phenocompass_loo_for_cluster(
                        coembeddings, "joint", cluster_smiles,
                        data["jump_background_compounds"], data["cluster_tanimoto_dfs"][moa],
                        T_max, MAX_FPR_VALUES, moa, MOA_TO_CLUSTER_IDX[moa], seed,
                    ))
        pc_joint_results = pd.concat(parts, ignore_index=True)
        pc_joint_results.to_parquet(joint_cache, index=False)
        print(f"PhenoCompass joint LOO complete: {len(pc_joint_results)} rows -> {joint_cache}")
    else:
        pc_joint_results = pd.read_parquet(joint_cache)
        print(f"Loaded cached PhenoCompass joint LOO: {len(pc_joint_results)} rows")

    # All modalities (struct vs morph vs joint comparison)
    if need_all:
        parts = []
        for anchor_modality in tqdm(MODALITIES, desc="PhenoCompass LOO (all modalities)"):
            for moa in MOA_LIST:
                cluster_smiles = sorted(
                    data["anchor_df"][data["anchor_df"]["cluster_moa"] == moa]["canonical_SMILES"].values
                )
                for T_max in T_MAX_VALUES:
                    for seed in range(N_SEEDS):
                        parts.append(run_phenocompass_loo_for_cluster(
                            coembeddings, anchor_modality, cluster_smiles,
                            data["jump_background_compounds"], data["cluster_tanimoto_dfs"][moa],
                            T_max, MAX_FPR_VALUES, moa, MOA_TO_CLUSTER_IDX[moa], seed,
                        ))
        pc_all_results = pd.concat(parts, ignore_index=True)
        pc_all_results.to_parquet(all_cache, index=False)
        print(f"PhenoCompass all-modality LOO complete: {len(pc_all_results)} rows -> {all_cache}")
    else:
        pc_all_results = pd.read_parquet(all_cache)
        print(f"Loaded cached PhenoCompass all-modality LOO: {len(pc_all_results)} rows")

    # Joint goes in directly; struct + morph come from the all-modality run (joint already covered above)
    return [pc_joint_results, pc_all_results[pc_all_results["modality"] != "joint"]]


def load_worker_partials(partials_dir):
    """Load regression and classifier baseline partials produced by ``loo_baseline_worker.py``.

    Missing files are treated as 'baseline not available' (warned, not fatal).
    """
    if not os.path.isdir(partials_dir):
        print(f"Partials directory not found: {partials_dir}")
        print("  Run scripts/run_loo_baselines_local.sh to produce baseline partials.")
        return []

    results = []
    patterns = [
        ("regression", METHOD_NAME, "GNEprop regression"),
        ("regression", "mf", "MF regression"),
        ("classifier", METHOD_NAME, "GNEprop classifier"),
        ("classifier", "mf", "MF classifier"),
    ]
    for kind, prefix, label in patterns:
        files = sorted(glob.glob(os.path.join(partials_dir, f"{kind}_{prefix}_*.parquet")))
        if files:
            df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            results.append(df)
            print(f"Loaded {len(files)} {label} partials: {len(df)} rows")
        else:
            print(f"No {label} partials found at {partials_dir} (skipping).")

    return results


def write_per_moa_tsvs(combined_results, output_dir):
    """Write 6 per-MoA TSVs to ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    for moa, moa_key in MOA_DISPLAY_MAP.items():
        moa_df = combined_results[combined_results["moa"] == moa]
        if len(moa_df) == 0:
            print(f"  WARN: no rows for {moa}; skipping {moa_key}.loo_results.tsv")
            continue
        out_path = os.path.join(output_dir, f"{moa_key}.loo_results.tsv")
        moa_df.to_csv(out_path, sep="\t", index=False)
        print(f"  Saved: {out_path} ({len(moa_df)} rows)")


# ── CLI entry point ───────────────────────────────────────────────────────────


def parse_args():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_final_data = os.path.join(repo_root, "final_data")

    p = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--final-data-dir", default=default_final_data,
                   help="Path to consolidated input data directory.")
    p.add_argument("--output-dir", default=os.path.join(default_final_data, "loo_comparisons"),
                   help="Directory where the 6 per-MoA TSVs are written.")
    p.add_argument("--cache-dir", default=os.path.join(repo_root, "data", "loo_comparisons_cache"),
                   help="Directory for intermediate per-stage parquet caches.")
    p.add_argument("--partials-dir", default=None,
                   help="Directory containing baseline-worker partial outputs "
                        "(default: <cache-dir>/partials).")
    p.add_argument("--h5-path", default=None,
                   help="Path to JUMP FastH5 dataset (default: "
                        "<final-data-dir>/jump_map/jump_dino_tvn.h5).")
    p.add_argument("--index-file", default=None,
                   help="Path to scaffold-cluster split index pickle "
                        "(default: <final-data-dir>/jump_map/scaffold_cluster_splits.pkl).")
    p.add_argument("--gneprop-path", default=None,
                   help="Path to GNEprop embeddings parquet "
                        "(default: <final-data-dir>/embeddings/structure/gneprop_embeddings.parquet).")
    p.add_argument("--no-tautomer-canonicalization", action="store_true",
                   help="Disable tautomer canonicalization (use original SMILES).")
    p.add_argument("--skip-phenocompass", action="store_true",
                   help="Skip the PhenoCompass cross-modal LOO stage (slow, ~5-10 min).")
    p.add_argument("--force-recompute", action="store_true",
                   help="Ignore cached parquets and recompute every stage.")

    args = p.parse_args()

    # Resolve dependent defaults
    if args.h5_path is None:
        args.h5_path = os.path.join(args.final_data_dir, "jump_map", "jump_dino_tvn.h5")
    if args.index_file is None:
        args.index_file = os.path.join(args.final_data_dir, "jump_map", "scaffold_cluster_splits.pkl")
    if args.gneprop_path is None:
        args.gneprop_path = os.path.join(
            args.final_data_dir, "embeddings", "structure", "gneprop_embeddings.parquet"
        )
    if args.partials_dir is None:
        args.partials_dir = os.path.join(args.cache_dir, "partials")

    return args


def main():
    args = parse_args()
    use_tautomer = not args.no_tautomer_canonicalization

    print("=" * 70)
    print("LOO comparisons preprocessing")
    print("=" * 70)
    print(f"  use_tautomer_canonicalization = {use_tautomer}")
    print(f"  min_training_compounds        = {MIN_TRAINING_COMPOUNDS}")
    print(f"  output_dir                    = {args.output_dir}")
    print(f"  cache_dir                     = {args.cache_dir}")
    print(f"  partials_dir                  = {args.partials_dir}")
    print(f"  skip_phenocompass             = {args.skip_phenocompass}")
    print(f"  force_recompute               = {args.force_recompute}")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    print("\n[1/4] Loading data")
    data = load_data(args, use_tautomer)

    print("\n[2/4] Direct-similarity LOO")
    all_results = run_direct_similarity_loo(args, data)

    if not args.skip_phenocompass:
        print("\n[3/4] PhenoCompass cross-modal LOO")
        all_results.extend(run_phenocompass_loo(args, data))
    else:
        print("\n[3/4] PhenoCompass cross-modal LOO  -> SKIPPED")

    print("\n[3.5/4] Loading worker baseline partials")
    all_results.extend(load_worker_partials(args.partials_dir))

    combined = pd.concat(all_results, ignore_index=True)
    print(f"\nCombined results: {len(combined)} rows, "
          f"methods = {sorted(combined['method_name'].unique())}, "
          f"MoAs = {sorted(combined['moa'].unique())}")

    print(f"\n[4/4] Writing per-MoA TSVs to {args.output_dir}")
    write_per_moa_tsvs(combined, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
