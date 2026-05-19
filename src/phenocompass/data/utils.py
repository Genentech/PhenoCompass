"""Data utilities for splitting and preprocessing."""

from math import isclose
from random import Random
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.utils.data
from deepchem.splits import (
    ButinaSplitter,
    FingerprintSplitter,
    RandomSplitter,
    ScaffoldSplitter,
)
from rdkit import Chem
from sklearn.neighbors import NearestNeighbors
from torch_geometric.loader import DataLoader


SPLITTERS = {
    "random": RandomSplitter,
    "scaffold": ScaffoldSplitter,
    "butina": ButinaSplitter,
    "fingerprint": FingerprintSplitter,
}


def random_split(dataset, seed, sizes):
    """Random train/val/test split.

    Parameters
    ----------
    dataset : Dataset
        Dataset to split.
    seed : int
        Random seed.
    sizes : tuple
        Tuple of (train_size, val_size, test_size) fractions.

    Returns
    -------
    tuple
        (train, val, test) CustomSubset objects.
    """
    from phenocompass.data.dataset import CustomSubset

    random = Random(seed)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    train_size = int(sizes[0] * len(dataset))
    train_val_size = int((sizes[0] + sizes[1]) * len(dataset))

    train = CustomSubset(dataset, indices[:train_size])
    val = CustomSubset(dataset, indices[train_size:train_val_size])
    test = CustomSubset(dataset, indices[train_val_size:])
    return train, val, test


def split_data(
    dataset,
    split_type="random",
    sizes=(0.8, 0.1, 0.1),
    seed=0,
    holdout=None,
    holdout_notion="smiles",
    holdout_to=None,
    verbose=False,
):
    """Split dataset into train/val/test sets.

    Parameters
    ----------
    dataset : Dataset
        Dataset to split.
    split_type : str, default "random"
        Split strategy ("random", "scaffold", "butina", "fingerprint", "index_predetermined").
    sizes : tuple, default (0.8, 0.1, 0.1)
        Fractions for train/val/test.
    seed : int, default 0
        Random seed.
    holdout : list or str, optional
        Compounds to hold out (list of SMILES/InChI or path to .npy file).
    holdout_notion : str, default "smiles"
        Notation type for holdout compounds ("smiles" or "inchi").
    holdout_to : str, optional
        Where to place holdout compounds ("train", "val", "test", or None to drop).
    verbose : bool, default False
        Whether to print debug information.

    Returns
    -------
    tuple
        (train, val, test) CustomSubset objects.
    """
    from phenocompass.data.dataset import CustomSubset

    assert isclose(np.array(sizes).sum(), 1, abs_tol=1e-03)

    if split_type == "index_predetermined":
        train_ix, val_ix, test_ix = index_predetermined_split(dataset, seed=seed)

    elif holdout is None:
        splitter = SPLITTERS[split_type]()
        train_ix, val_ix, test_ix = splitter.split(
            dataset, frac_train=sizes[0], frac_valid=sizes[1], frac_test=sizes[2], seed=seed
        )

    elif split_type == "scaffold" or split_type == "fingerprint":
        splitter = SPLITTERS[split_type]()

        if isinstance(holdout, List):
            smiles_to_holdout = holdout
        elif holdout.endswith(".npy"):
            smiles_to_holdout = np.load(holdout)

        if holdout_notion == "smiles":
            pass
        elif holdout_notion == "inchi":
            smiles_to_holdout = set(
                np.array(
                    [Chem.MolToSmiles(Chem.inchi.MolFromInchi(i)) for i in smiles_to_holdout]
                ).astype(str)
            )
        else:
            raise NotImplementedError

        dataset_ids_df = pd.DataFrame({"smiles": dataset.ids})
        id_idx_to_holdout = dataset_ids_df[
            dataset_ids_df["smiles"].isin(smiles_to_holdout)
        ].index.values
        if verbose:
            print(f"Number of compounds to hold out: {len(id_idx_to_holdout)}")

        if split_type == "scaffold":
            # remove hold out scaffold from scaffold
            scaffold_sets = splitter.generate_scaffolds(dataset)

            # shuffle scaffold sets
            np.random.seed(seed)
            np.random.shuffle(scaffold_sets)

            holdout_scaffold_sets = []
            for idx_val in id_idx_to_holdout:
                for scaffold_idx, scaffold_set in enumerate(scaffold_sets):
                    if idx_val in scaffold_set:
                        holdout_scaffold_sets.append(scaffold_sets.pop(scaffold_idx))
            if verbose:
                print(f"Total compounds to hold out: {recursive_len(holdout_scaffold_sets)}")

            # split
            split_mapper = {"train": [], "test": [], "val": []}
            if holdout_to is not None:
                assert holdout_to in ["train", "test", "val"]
                cur_dataset_len = len(dataset)
                if verbose:
                    print(f"Adding holdout set to subset {holdout_to}")
                for h in holdout_scaffold_sets:
                    split_mapper[holdout_to] += h
            else:
                cur_dataset_len = recursive_len(scaffold_sets)

            frac_train, frac_valid = sizes[0], sizes[1]
            train_cutoff = frac_train * cur_dataset_len
            valid_cutoff = (frac_train + frac_valid) * cur_dataset_len

            train_ix = split_mapper["train"]
            val_ix = split_mapper["val"]
            test_ix = split_mapper["test"]

            for scaffold_set in scaffold_sets:
                if len(train_ix) + len(scaffold_set) > train_cutoff:
                    if len(train_ix) + len(val_ix) + len(scaffold_set) > valid_cutoff:
                        test_ix += scaffold_set
                    else:
                        val_ix += scaffold_set
                else:
                    train_ix += scaffold_set

        elif split_type == "fingerprint":
            # hold out compounds + shuffle dataset
            retain_ix = [i for i in range(len(dataset)) if i not in id_idx_to_holdout]
            np.random.seed(seed)
            np.random.shuffle(retain_ix)

            dataset = CustomSubset(dataset, retain_ix)

            if verbose:
                print("Performing fingerprint splitting")
            train_ix, val_ix, test_ix = splitter.split(
                dataset, frac_train=sizes[0], frac_valid=sizes[1], frac_test=sizes[2], seed=seed
            )

    train, val, test = (
        CustomSubset(dataset, train_ix),
        CustomSubset(dataset, val_ix),
        CustomSubset(dataset, test_ix),
    )

    return train, val, test


def recursive_len(item):
    """Recursively compute length of nested lists."""
    if type(item) == list:
        return sum(recursive_len(subitem) for subitem in item)
    else:
        return 1


def filter_dataset_by_source(dataset, ix_list, split):
    """Filter dataset indices by source eligibility."""
    split_ix_set = set(dataset.f[f"ix_{split}_eligible"])
    retained_ix = [ix for ix in ix_list if ix in split_ix_set]
    return retained_ix


def split_data_source(dataset, split_type="random", sizes=(0.8, 0.1, 0.1), seed=0):
    """Split dataset by source eligibility.

    Parameters
    ----------
    dataset : Dataset
        Dataset to split.
    split_type : str, default "random"
        Split strategy.
    sizes : tuple, default (0.8, 0.1, 0.1)
        Fractions for train/val/test.
    seed : int, default 0
        Random seed.

    Returns
    -------
    tuple
        (train, val, test) CustomSubset objects.
    """
    from phenocompass.data.dataset import CustomSubset

    assert isclose(np.array(sizes).sum(), 1, abs_tol=1e-03)

    splitter = SPLITTERS[split_type]()
    train_ix, val_ix, test_ix = splitter.split(
        dataset, frac_train=sizes[0], frac_valid=sizes[1], frac_test=sizes[2], seed=seed
    )

    train_ix = filter_dataset_by_source(dataset, train_ix, "train")
    val_ix = filter_dataset_by_source(dataset, val_ix, "val")
    test_ix = filter_dataset_by_source(dataset, test_ix, "test")

    train = CustomSubset(dataset, train_ix, "source_train")
    val = CustomSubset(dataset, val_ix, "source_val")
    test = CustomSubset(dataset, test_ix, "source_test")

    return train, val, test


def index_predetermined_split(dataset, seed):
    """Use predetermined split indices from dataset.

    Parameters
    ----------
    dataset : Dataset
        Dataset with index_predetermined attribute.
    seed : int
        Seed to select split.

    Returns
    -------
    list
        List of [train_ix, val_ix, test_ix].
    """
    split_indices = [
        sorted(list(set(indices))) for indices in dataset.index_predetermined[seed]
    ]

    if len(split_indices) != 3:
        raise ValueError("Split indices must have three splits: train, validation, and test")

    return split_indices


def dataset_similarity_edge_index(dataset, neighbor_thresh, n_neighbors, metric="cosine", batch_size=5096):
    """Compute similarity-based edge index for neighbor sampling.

    Parameters
    ----------
    dataset : Dataset
        Dataset with morphology features.
    neighbor_thresh : float, optional
        Threshold for radius neighbors.
    n_neighbors : int
        Number of neighbors for KNN.
    metric : str, default "cosine"
        Distance metric.
    batch_size : int, default 5096
        Batch size for loading data.

    Returns
    -------
    np.ndarray
        Edge index array.
    """
    dataloader = DataLoader(
        dataset=dataset, batch_size=batch_size, pin_memory=False, drop_last=False, shuffle=False
    )

    img_representations = np.concatenate([batch["inputs"]["morph"] for batch in dataloader])

    nn = NearestNeighbors(metric=metric, n_jobs=16)
    nn.fit(img_representations)

    if metric == "cosine":
        if neighbor_thresh is not None:
            radius_graph = nn.radius_neighbors_graph(
                img_representations, radius=1 - neighbor_thresh
            )
            edge_index = np.array(np.nonzero(radius_graph))
            edge_index = edge_index[:, edge_index[0] != edge_index[1]]
            print(f"Number of edges: {edge_index.shape[1]}")
            return edge_index
        else:
            knn_graph = nn.kneighbors_graph(img_representations, n_neighbors=n_neighbors)
            return np.array(np.nonzero(knn_graph))


