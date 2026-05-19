"""PyTorch Lightning data modules."""

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader, NeighborLoader

from phenocompass.data.utils import (
    dataset_similarity_edge_index,
    split_data,
    split_data_source,
)


class MultiInput_DataModule(pl.LightningDataModule):
    """Lightning data module for multimodal datasets.

    Parameters
    ----------
    dataset : Dataset
        Dataset to use.
    batch_size : int, default 128
        Batch size.
    num_workers : int, default 8
        Number of data loading workers.
    pin_memory : bool, default False
        Whether to pin memory.
    split_sizes : tuple, default (0.8, 0.1, 0.1)
        Train/val/test split sizes.
    split_type : str, default "random"
        Split type.
    split_by_source : bool, default False
        Whether to split by source eligibility.
    holdout : list, optional
        Compounds to hold out.
    holdout_notion : str, default "inchi"
        Notation for holdout compounds.
    holdout_to : str, optional
        Where to place holdout compounds.
    shuffle_train : bool, default True
        Whether to shuffle training data.
    seed : int, default 0
        Random seed.
    batch_sampler : bool, default False
        Whether to use batch-based sampling.
    meta_idx : int, default 0
        Metadata column index for batch sampling.
    batch_sampler_mode : str, default "metadata"
        Batch sampler mode.
    neighbor_sampler : bool, default False
        Whether to use neighbor-based sampling.
    neighbor_thresh : float, optional
        Threshold for neighbor sampling.
    n_neighbors : int, default 10
        Number of neighbors for sampling.
    verbose : bool, default True
        Whether to print debug info.
    """

    def __init__(
        self,
        dataset,
        batch_size=128,
        num_workers=8,
        pin_memory=False,
        split_sizes=(0.8, 0.1, 0.1),
        split_type="random",
        split_by_source=False,
        holdout=None,
        holdout_notion="inchi",
        holdout_to=None,
        shuffle_train=True,
        seed=0,
        batch_sampler=False,
        meta_idx=0,
        batch_sampler_mode="metadata",
        neighbor_sampler=False,
        neighbor_thresh=None,
        n_neighbors=10,
        verbose=True,
    ):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.split_sizes = split_sizes
        self.split_type = split_type
        self.split_by_source = split_by_source
        self.seed = seed
        self.holdout = holdout
        self.holdout_notion = holdout_notion
        self.holdout_to = holdout_to
        self.shuffle_train = shuffle_train
        self.batch_sampler = batch_sampler
        self.meta_idx = meta_idx
        self.batch_sampler_mode = batch_sampler_mode
        self.neighbor_sampler = neighbor_sampler
        self.neighbor_thresh = neighbor_thresh
        self.n_neighbors = n_neighbors
        self.verbose = verbose

    def setup(self, stage: str):
        if self.verbose:
            print(f"Dataset size: {len(self.dataset)}")
            print(f"Train on samples from {self.split_type}.")

        if self.split_by_source:
            self.train_dataset, self.val_dataset, self.test_dataset = split_data_source(
                self.dataset,
                split_type=self.split_type,
                sizes=self.split_sizes,
                seed=self.seed,
            )
        else:
            if self.split_type == "index_predetermined":
                assert self.dataset.index_predetermined is not None

            self.train_dataset, self.val_dataset, self.test_dataset = split_data(
                self.dataset,
                split_type=self.split_type,
                sizes=self.split_sizes,
                seed=self.seed,
                holdout=self.holdout,
                holdout_notion=self.holdout_notion,
                holdout_to=self.holdout_to,
                verbose=self.verbose,
            )
        # No data augmentation for validation and test set
        self.val_dataset.morph_aug = None
        self.test_dataset.morph_aug = None

        if self.verbose:
            print(f"Train on {len(self.train_dataset)} samples.")
            print(f"Validate on {len(self.val_dataset)} samples.")
            print(f"Test on {len(self.test_dataset)} samples.")

    def train_dataloader(self):
        if self.batch_sampler:
            sampler = BatchSampler(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                mode=self.batch_sampler_mode,
                meta_idx=self.meta_idx,
                shuffle=self.shuffle_train,
            )
            data_loader = DataLoader(
                dataset=self.train_dataset,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=False,
            )
        elif self.neighbor_sampler:
            self.train_dataset.edge_index = dataset_similarity_edge_index(
                self.train_dataset, self.neighbor_thresh, self.n_neighbors
            )
            sampler = NeighborSampler(
                self.train_dataset,
                self.batch_size,
                n_neighbors=self.n_neighbors,
                shuffle=self.shuffle_train,
            )
            data_loader = DataLoader(
                dataset=self.train_dataset,
                batch_sampler=sampler,
                num_workers=self.num_workers,
                pin_memory=False,
            )
        else:
            data_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False,
                shuffle=self.shuffle_train,
            )
        return data_loader

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            shuffle=False,
        )


class BatchSampler(torch.utils.data.Sampler):
    """Sample well-level profiles from the same experimental batch."""

    def __init__(self, dataset, batch_size, meta_idx=0, mode="batch", shuffle=False):
        self.mode = mode
        self.shuffle = shuffle
        self.dataset = dataset
        self.morph_idx = dataset.morph_idx
        self.morph_meta = dataset.morph_meta
        self.ids = dataset.ids
        self.compound_ids = dataset.compound_ids
        metadata_arr = self.morph_meta

        self.meta_col = metadata_arr[:, meta_idx]
        df_meta = pd.DataFrame(
            {"meta": self.meta_col, "indices": np.arange(metadata_arr.shape[0])}
        )
        self.df_meta_groupby = df_meta.groupby("meta")

        self.batch_size = batch_size
        self.meta_ids = list(np.unique(self.meta_col))
        scale_factor = len(dataset.ids) // len(self.meta_ids)
        print(f"the number of compounds is {len(dataset.ids)}")
        print(f"the number of experimental batches is {len(self.meta_ids)}")
        print(f"{metadata_arr.shape=}")
        # to roughly keep the epoch sizes of the two modes equal
        self.meta_ids = self.meta_ids * scale_factor

    def __len__(self):
        if self.mode == "metadata":
            return len(self.meta_ids)
        elif self.mode == "sample":
            return len(self.compound_ids)
        else:
            raise ValueError

    def __iter__(self):
        if self.mode == "metadata":
            # sample the mini-batch based on the given metadata
            if self.shuffle:
                np.random.shuffle(self.meta_ids)
            for meta_value in self.meta_ids:
                indices = np.where(self.meta_col == meta_value)[0]
                if len(indices) > self.batch_size:
                    sampled_indices = np.random.choice(
                        indices, size=self.batch_size, replace=False
                    )
                else:
                    sampled_indices = np.random.choice(
                        indices, size=self.batch_size, replace=True
                    )
                yield sampled_indices

        elif self.mode == "sample":
            # sample the mini-batch based on the metadata of the given compound
            sample_indices = np.arange(len(self.compound_ids))

            if self.shuffle:
                np.random.shuffle(sample_indices)
            for idx in sample_indices:
                comp_indices = sample_indices[idx]
                mol_sidx, mol_eidx = self.morph_idx[comp_indices]
                profile_idx = np.random.choice(
                    np.arange(mol_sidx, mol_eidx), size=1, replace=False
                )[0]

                flag = True
                delta = 0
                while flag:
                    profile_idx = idx + delta
                    batch = self.meta_col[profile_idx]

                    indices = np.where(self.meta_col == batch)[0]
                    id_arr = [self.dataset.ids[tmp] for tmp in indices]
                    batch_arr = [self.meta_col[tmp] for tmp in indices]

                    target_id = self.dataset.ids[profile_idx]
                    df = pd.DataFrame(
                        {"indices": indices, "id": id_arr, "batch": batch_arr}
                    )
                    df = df[df["id"] != target_id]

                    df_sampled = df.groupby("id").sample(n=1)
                    flag = df_sampled.shape[0] <= self.batch_size - 1
                    delta += 1

                df_sampled = df_sampled.sample(n=self.batch_size - 1)

                indices = df_sampled["indices"].values
                sampled_indices = np.append(profile_idx, indices)
                yield sampled_indices

        else:
            raise ValueError


class NeighborSampler(torch.utils.data.Sampler):
    """Graph-based neighbor sampling."""

    def __init__(self, dataset, batch_size, n_neighbors, shuffle=True):
        self.shuffle = shuffle
        self.dataset = dataset
        self.ids = dataset.ids

        self.batch_size = batch_size
        self.n_neighbors = n_neighbors

        self.data = Data(
            x=torch.ones((len(self.ids), 1)),
            edge_index=torch.LongTensor(self.dataset.edge_index),
        )
        self.neighbor_loader = NeighborLoader(
            self.data,
            num_neighbors=[self.n_neighbors],
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def __len__(self):
        return len(self.ids)

    def __iter__(self):
        for batch in self.neighbor_loader:
            yield batch.n_id.data.numpy()
