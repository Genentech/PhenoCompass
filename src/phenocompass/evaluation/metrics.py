"""Evaluation metrics and callbacks."""

from collections import defaultdict
from functools import partial
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import sklearn
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import BulkTanimotoSimilarity, TanimotoSimilarity
from scipy.stats import pearsonr, rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    pairwise_distances,
    roc_auc_score,
)
from sklearn.metrics.pairwise import cosine_similarity, paired_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
from torch_geometric.loader import DataLoader
from tqdm import tqdm, trange

from phenocompass.data.dataset import StructPredictionDataset
from phenocompass.data.utils import split_data
from phenocompass.models.utils import move_batch_input_to_device


def train_test_k_folds_model(model, dataset, split_type="random", init_seed=0, n_folds=10):
    """K-fold cross-validation for model evaluation.

    Parameters
    ----------
    model : LightningModule
        Model to evaluate.
    dataset : Dataset
        Dataset to use.
    split_type : str, default "random"
        Split type.
    init_seed : int, default 0
        Initial random seed.
    n_folds : int, default 10
        Number of folds.

    Returns
    -------
    dict
        Dictionary with "auc" and "ap" mean scores.
    """
    all_test_ap = []
    all_test_auc = []
    for i in trange(n_folds):
        train_set, _, test_set = split_data(
            dataset, split_type=split_type, sizes=(0.8, 0.1, 0.1), seed=init_seed + i
        )

        train_loader = DataLoader(
            train_set, batch_size=100, num_workers=0, pin_memory=False, shuffle=False
        )
        test_loader = DataLoader(
            test_set, batch_size=100, num_workers=0, pin_memory=False, shuffle=False
        )

        reprs_train = model.compute_representation_dataloader(
            train_loader, disable_progress_bar=True
        )
        reprs_test = model.compute_representation_dataloader(
            test_loader, disable_progress_bar=True
        )

        clf = LogisticRegression(random_state=0, solver="lbfgs", max_iter=10000).fit(
            reprs_train, train_set.y
        )
        clf_preds = clf.predict(reprs_test)

        test_ap = sklearn.metrics.average_precision_score(test_set.y, clf_preds)
        test_auc = sklearn.metrics.roc_auc_score(test_set.y, clf_preds)
        all_test_ap.append(test_ap)
        all_test_auc.append(test_auc)

    return {"auc": np.mean(all_test_auc), "ap": np.mean(all_test_ap)}


def get_rankings(
    given_emb, target_emb, neg_size=99, metric="cosine", predefined_background=None, seed=0
):
    """Compute retrieval rankings.

    Parameters
    ----------
    given_emb : torch.Tensor
        Query embeddings.
    target_emb : torch.Tensor
        Target embeddings.
    neg_size : int, default 99
        Number of negative samples.
    metric : str, default "cosine"
        Distance metric.
    predefined_background : dict, optional
        Predefined background indices.
    seed : int, default 0
        Random seed.

    Returns
    -------
    np.ndarray
        Rankings for each query.
    """
    np.random.seed(seed)
    given_emb = torch.Tensor(given_emb)
    target_emb = torch.Tensor(target_emb)

    # distance matrix of embeddings
    if metric == "euclidean":
        dist = torch.cdist(given_emb, target_emb, p=2).cpu().numpy()
    elif metric == "cosine":
        dist = 1 - torch.mm(given_emb, target_emb.transpose(0, 1)).cpu().numpy()
    else:
        raise NotImplementedError

    # random sample negative samples
    select_idxs = sample_select_idxs(
        dist, neg_size, predefined_background=predefined_background
    )

    # get distance matrix of sampled samples
    sample_dist = np.zeros((dist.shape[0], neg_size + 1))
    for cur_mol_idx in range(dist.shape[0]):
        indices = select_idxs[cur_mol_idx]
        sample_dist[cur_mol_idx] = dist[cur_mol_idx][indices]

    all_rankings = rankdata(sample_dist, method="min", axis=1)[:, 0]
    return all_rankings


def process_all_rankings(all_rankings, key, r_range=[1, 5, 10]):
    """Process rankings into metrics.

    Parameters
    ----------
    all_rankings : np.ndarray
        Array of rankings.
    key : str
        Metric key prefix.
    r_range : list, default [1, 5, 10]
        Rank thresholds for Hits@K.

    Returns
    -------
    dict
        Dictionary with MRR and Hits@K metrics.
    """
    all_rankings = np.array(all_rankings)
    mrr = float(np.mean(1 / all_rankings))
    mr = float(np.mean(all_rankings))
    metric_dict = {f"{key}_mr": mr, f"{key}_mrr": mrr}
    h_dict = defaultdict(lambda: 0.0)
    for r in r_range:
        h = float(np.mean(all_rankings <= r))
        h_dict[f"{key}_h{r}"] = h
    metric_dict.update(h_dict)
    return metric_dict


def sample_select_idxs(sim_mat, neg_size, predefined_background=None, seed=0):
    """Sample indices for negative sampling."""
    np.random.seed(seed)
    select_idxs = []
    for cur_mol_idx in range(sim_mat.shape[0]):
        idxs = np.arange(sim_mat.shape[1]).tolist()
        idxs = idxs[:cur_mol_idx] + idxs[cur_mol_idx + 1 :]
        if predefined_background is None:
            indices = np.random.choice(idxs, neg_size + 1, replace=False)
        else:
            candidate_indices = predefined_background[cur_mol_idx]
            replace = len(candidate_indices) < neg_size + 1
            indices = np.random.choice(candidate_indices, neg_size + 1, replace=replace)
        indices[0] = cur_mol_idx
        select_idxs.append(indices)
    return select_idxs


def fixed_set_retrieval_acc(
    representations, dataset, seed, retrieval_pairs=None, metric="cosine", neg_size=99, r_range=[1, 5, 10]
):
    """Compute retrieval accuracy on fixed set.

    Parameters
    ----------
    representations : dict
        Dictionary of representations per modality.
    dataset : Dataset
        Dataset (unused, for compatibility).
    seed : int
        Random seed.
    retrieval_pairs : list, optional
        List of (source, target) modality pairs.
    metric : str, default "cosine"
        Distance metric.
    neg_size : int, default 99
        Number of negatives.
    r_range : list, default [1, 5, 10]
        Rank thresholds.

    Returns
    -------
    tuple
        (metrics_dict, retrieval_pairs)
    """
    import itertools

    if retrieval_pairs is None:
        mod_names = list(representations.keys())
        retrieval_pairs = list(itertools.product(mod_names, mod_names))
        for mod in mod_names:
            retrieval_pairs.remove((mod, mod))

    dict_rankings = defaultdict(list)
    for given, target in retrieval_pairs:
        key = f"{given}>{target}"
        given_emb = representations[given]
        target_emb = representations[target]
        rankings = get_rankings(
            given_emb, target_emb, neg_size=neg_size, metric=metric, seed=seed
        )
        dict_rankings[key].extend(rankings)

    all_metric_dict = defaultdict()
    for key, all_rankings in dict_rankings.items():
        metric_dict = process_all_rankings(all_rankings, key, r_range)
        all_metric_dict.update(metric_dict)

    return all_metric_dict, retrieval_pairs


def fixed_set_retrieval_acc_batch_match(
    representations,
    dataset,
    seed,
    retrieval_pairs=None,
    morph_meta_idx=0,
    metric="cosine",
    neg_size=99,
    r_range=[1, 5, 10],
    l1_thresh=0.5,
    use_knn=True,
):
    """Compute retrieval accuracy with batch matching.

    Parameters
    ----------
    representations : dict
        Dictionary of representations.
    dataset : Dataset
        Dataset with batch assignments.
    seed : int
        Random seed.
    retrieval_pairs : list, optional
        Retrieval pairs.
    morph_meta_idx : int, default 0
        Metadata index.
    metric : str, default "cosine"
        Distance metric.
    neg_size : int, default 99
        Number of negatives.
    r_range : list, default [1, 5, 10]
        Rank thresholds.
    l1_thresh : float, default 0.5
        L1 distance threshold.
    use_knn : bool, default True
        Whether to use KNN.

    Returns
    -------
    tuple
        (metrics_dict, retrieval_pairs)
    """
    import itertools

    predefined_background_dict = defaultdict(list)

    if len(dataset.batch_assignments.shape) > 1:
        # one hot encoding
        l1_dist = pairwise_distances(dataset.batch_assignments, metric="l1")
        l1_dist_norm = l1_dist / dataset.batch_assignments.sum(1)
        if use_knn:
            nn = NearestNeighbors(metric="precomputed")
            nn.fit(l1_dist_norm)
            neigh_idx = nn.kneighbors(
                l1_dist_norm, n_neighbors=neg_size + 1, return_distance=False
            )
            for i, neigh in enumerate(neigh_idx):
                predefined_background_dict[i] = neigh[neigh != i].tolist()
        else:
            for i, j in zip(*np.where(l1_dist_norm <= l1_thresh)):
                predefined_background_dict[i].append(j)
    else:
        df = pd.DataFrame(
            {"batch": dataset.batch_assignments, "indices": np.arange(len(dataset))}
        )
        df_groupby = df.groupby("batch")
        for _, sdf in df_groupby:
            indices = sdf["indices"].values
            x, y = np.meshgrid(indices, indices)
            for i, j in zip(x.reshape(-1), y.reshape(-1)):
                predefined_background_dict[i].append(j)

    if retrieval_pairs is None:
        mod_names = list(representations.keys())
        retrieval_pairs = list(itertools.product(mod_names, mod_names))
        for mod in mod_names:
            retrieval_pairs.remove((mod, mod))

    dict_rankings = defaultdict(list)
    for given, target in retrieval_pairs:
        key = f"{given}>{target}"
        given_emb = representations[given]
        target_emb = representations[target]
        rankings = get_rankings(
            given_emb,
            target_emb,
            neg_size=neg_size,
            metric=metric,
            predefined_background=predefined_background_dict,
            seed=seed,
        )
        dict_rankings[key].extend(rankings)

    all_metric_dict = defaultdict()
    for key, all_rankings in dict_rankings.items():
        metric_dict = process_all_rankings(all_rankings, key, r_range)
        all_metric_dict.update(metric_dict)

    return all_metric_dict, retrieval_pairs


class RetrievalOnlineEvaluator(Callback):
    """Online retrieval evaluation callback."""

    def __init__(
        self,
        model,
        retrieval_pairs=None,
        neg_size=99,
        seed=0,
        batch_match=True,
        batch_size=512,
        l1_thresh=0.5,
        use_knn=True,
    ):
        super().__init__()
        self.seed = seed
        self.retrieval_dataset = None
        self.neg_size = neg_size
        self.eval_model = model
        self.retrieval_pairs = retrieval_pairs
        self.set = False
        self.batch_match = batch_match
        self.batch_size = batch_size
        self.l1_thresh = l1_thresh
        self.use_knn = use_knn

    def set_dataset(self, dataset):
        self.retrieval_dataset = dataset

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.set_dataset(trainer.datamodule.val_dataloader().dataset)

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.eval_model.load_state_dict(pl_module.state_dict())
        self.eval_model.eval()
        self.eval_model.to(device=pl_module.device)

        with torch.no_grad():
            test_loader = DataLoader(
                self.retrieval_dataset,
                batch_size=self.batch_size,
                num_workers=0,
                pin_memory=False,
                shuffle=False,
            )
            representations = self.eval_model.compute_representation_dataloader(
                test_loader, mod_name=None, disable_progress_bar=False, return_mol=False
            )
            retrieval_results, pairs = fixed_set_retrieval_acc(
                representations,
                dataset=self.retrieval_dataset,
                retrieval_pairs=self.retrieval_pairs,
                seed=self.seed,
                neg_size=self.neg_size,
            )
            if self.batch_match:
                retrieval_results_batch_match, _ = fixed_set_retrieval_acc_batch_match(
                    representations,
                    dataset=self.retrieval_dataset,
                    retrieval_pairs=self.retrieval_pairs,
                    l1_thresh=self.l1_thresh,
                    seed=self.seed,
                    neg_size=self.neg_size,
                    use_knn=self.use_knn,
                )

        for given, target in pairs:
            key = f"{given}>{target}"
            pl_module.log(
                f"{key}/mrr", retrieval_results[f"{key}_mrr"], prog_bar=True, sync_dist=True
            )
            pl_module.log(
                f"{key}/h1", retrieval_results[f"{key}_h1"], prog_bar=True, sync_dist=True
            )
            pl_module.log(
                f"{key}/h5", retrieval_results[f"{key}_h5"], prog_bar=True, sync_dist=True
            )
            pl_module.log(
                f"{key}/h10", retrieval_results[f"{key}_h10"], prog_bar=True, sync_dist=True
            )
            if self.batch_match:
                pl_module.log(
                    f"{key}/mrr_batch_match",
                    retrieval_results_batch_match[f"{key}_mrr"],
                    prog_bar=True,
                    sync_dist=True,
                )
                pl_module.log(
                    f"{key}/h1_batch_match",
                    retrieval_results_batch_match[f"{key}_h1"],
                    prog_bar=True,
                    sync_dist=True,
                )


class LinearTransferOnlineEvaluator(Callback):
    """Online linear transfer evaluation callback."""

    def __init__(self, transfer_dataset, model, num_folds=10, seed=0):
        super().__init__()
        self.seed = seed
        self.num_folds = num_folds
        self.transfer_dataset = transfer_dataset
        self.transfer_model = model

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.transfer_model.load_state_dict(pl_module.state_dict())
        self.transfer_model.eval()
        self.transfer_model.to(device=pl_module.device)

        with torch.no_grad():
            retrieval_results = train_test_k_folds_model(
                self.transfer_model,
                self.transfer_dataset,
                init_seed=self.seed,
                n_folds=self.num_folds,
            )

        pl_module.log("mrr", retrieval_results.get("mrr", 0), prog_bar=True, sync_dist=True)
        pl_module.log("h1", retrieval_results.get("h1", 0), prog_bar=True, sync_dist=True)


class LinearProbeEvaluator(Callback):
    """Linear probe evaluation callback."""

    def __init__(self, dataset, model, num_folds=10, seed=0):
        super().__init__()
        self.seed = seed
        self.num_folds = num_folds
        self.LP_dataset = dataset
        self.LP_model = model

    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.LP_model.to(device=pl_module.device)

        # Run linear probe evaluation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        all_test_ap = []
        all_test_auc = []

        for i in range(self.num_folds):
            self.LP_model.backbone.load_state_dict(pl_module.state_dict())
            train_set, val_set, test_set = split_data(
                self.LP_dataset, split_type="random", sizes=(0.8, 0.0, 0.2), seed=self.seed + i
            )
            train_loader = DataLoader(
                train_set, batch_size=128, num_workers=0, pin_memory=False, shuffle=True
            )
            test_loader = DataLoader(
                test_set, batch_size=99, num_workers=0, pin_memory=False, shuffle=False
            )

            # Train
            self.LP_model.to(device)
            self.LP_model.train()
            for epoch in range(20):
                for batch in train_loader:
                    label = batch["labels"].to(device)
                    self.LP_model.optimizer.zero_grad()
                    probs, logits = self.LP_model._forward_with_sigmoid(
                        batch, device=device, return_mod="logits"
                    )
                    loss = self.LP_model.loss(logits, label)
                    loss.backward()
                    self.LP_model.optimizer.step()

            # Test
            self.LP_model.eval()
            with torch.no_grad():
                probs, labels = self.LP_model.predict_probs_dataloader(
                    test_loader, device=device, return_mod="label", mod_name=None, return_mol=False
                )
                lp_results, _, _ = compute_metrics(preds=probs.T, labels=labels.T)
                all_test_ap.append(lp_results["ap"])
                all_test_auc.append(lp_results["auc"])

        pl_module.log(
            "LinearProbe_auc", np.mean(all_test_auc), prog_bar=True, sync_dist=True
        )
        pl_module.log(
            "LinearProbe_ap", np.mean(all_test_ap), prog_bar=True, sync_dist=True
        )


def compute_metrics(preds, labels):
    """Compute AP and AUC metrics."""
    test_ap_list = []
    test_auc_list = []
    for idx in range(0, len(labels)):
        mask = ~np.isnan(labels[idx])
        s1 = set(mask)
        s2 = set(labels[idx][mask])
        if len(s1) == 1 or len(s2) == 1:
            test_ap_list.append(np.nan)
            test_auc_list.append(np.nan)
        else:
            y_true = labels[idx][mask]
            y_scores = preds[idx][mask]
            test_ap = sklearn.metrics.average_precision_score(y_true, y_scores)
            test_auc = sklearn.metrics.roc_auc_score(y_true, y_scores)
            test_ap_list.append(test_ap)
            test_auc_list.append(test_auc)

    return {"auc": np.nanmean(test_auc_list), "ap": np.nanmean(test_ap_list)}, test_auc_list, test_ap_list


def compute_pairwise_tanimoto_similarity(morgan_fpts_X, morgan_fpts_Y=None):
    """Compute pairwise Tanimoto similarity.

    Parameters
    ----------
    morgan_fpts_X : list
        List of Morgan fingerprints.
    morgan_fpts_Y : list, optional
        Second list of fingerprints.

    Returns
    -------
    np.ndarray
        Similarity matrix.
    """
    if morgan_fpts_Y is None:
        sim_arr = np.zeros((len(morgan_fpts_X), len(morgan_fpts_X)))
        for i in range(len(morgan_fpts_X)):
            sim_arr[i] = BulkTanimotoSimilarity(morgan_fpts_X[i], morgan_fpts_X)
        return sim_arr
    else:
        sim_arr = np.zeros((len(morgan_fpts_X), len(morgan_fpts_Y)))
        for i in range(len(morgan_fpts_X)):
            sim_arr[i] = BulkTanimotoSimilarity(morgan_fpts_X[i], morgan_fpts_Y)
        return sim_arr


def compute_paired_tanimoto_similarity(morgan_fpts_X, morgan_fpts_Y):
    """Compute paired Tanimoto similarity."""
    return np.array(
        [TanimotoSimilarity(morgan_fpts_X[i], morgan_fpts_Y[i]) for i in range(len(morgan_fpts_X))]
    )


def compute_fewshot_similarities(train_embeddings, test_embeddings, metric="cosine", mode="mean"):
    """Compute few-shot similarities.

    Parameters
    ----------
    train_embeddings : np.ndarray
        Training/anchor embeddings.
    test_embeddings : np.ndarray
        Test embeddings.
    metric : str, default "cosine"
        Similarity metric.
    mode : str, default "mean"
        Aggregation mode ("mean" or "max").

    Returns
    -------
    np.ndarray
        Similarity scores.
    """
    if metric == "cosine":
        all_scores = cosine_similarity(train_embeddings, test_embeddings)
    elif metric == "tanimoto":
        all_scores = compute_pairwise_tanimoto_similarity(train_embeddings, test_embeddings)
    elif metric == "euclidean":
        all_scores = -pairwise_distances(train_embeddings, test_embeddings)
    else:
        raise NotImplementedError

    if mode == "mean":
        return all_scores.mean(0)
    elif mode == "max":
        return all_scores.max(0)


def compute_struct_embeddings(model, mols, mol_notation="smile", batch_size=256):
    """Compute structure embeddings for molecules.

    Parameters
    ----------
    model : LightningModule
        Model with compute_representation_dataloader method.
    mols : list
        List of molecules.
    mol_notation : str, default "smile"
        Molecule notation type.
    batch_size : int, default 256
        Batch size.

    Returns
    -------
    pd.DataFrame
        DataFrame with embeddings indexed by molecule.
    """
    mol_df = pd.DataFrame({"mol": mols})
    dataset = StructPredictionDataset(mol_df, mol_col="mol", mol_notation=mol_notation)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    struct_representations, mols = model.compute_representation_dataloader(
        dataloader, mod_name="struct", disable_progress_bar=False, return_mol=True
    )
    X = struct_representations["struct"]

    return pd.DataFrame(X, index=mols)
