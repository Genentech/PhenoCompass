"""SimCLR pretraining components."""

import math
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm


class Projection(nn.Module):
    """Projection head for SimCLR (3-layer MLP with batch norm)."""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class ProjectionLinear(nn.Module):
    """Linear projection head."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.model = nn.Linear(self.input_dim, self.output_dim, bias=False)

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)


class SimCLR(pl.LightningModule):
    """SimCLR self-supervised pretraining module for molecular graphs."""

    def __init__(self, **kwargs):
        super().__init__()

        # backcompatibility, add defaults
        self._add_default_args(kwargs)

        self.save_hyperparameters()

        self.encoder = self.init_model(
            self.hparams.model_hidden_size, self.hparams.model_depth, jk=self.hparams.jk
        )

        if self.hparams.jk == "none":
            output_model_dim = self.hparams.model_hidden_size
        elif self.hparams.jk == "cat":
            output_model_dim = self.hparams.model_hidden_size * (self.hparams.model_depth + 1)
        else:
            raise NotImplementedError

        if self.hparams.linear_projection:
            self.projection = ProjectionLinear(
                input_dim=output_model_dim, output_dim=self.hparams.project_output_dim
            )
        else:
            self.projection = Projection(
                input_dim=output_model_dim,
                hidden_dim=self.hparams.project_hidden_dim,
                output_dim=self.hparams.project_output_dim,
            )

        # compute iters per epoch
        global_batch_size = (
            self.hparams.num_nodes * self.hparams.gpus * self.hparams.batch_size
            if self.hparams.gpus > 0
            else self.hparams.batch_size
        )
        self.train_iters_per_epoch = self.hparams.num_samples // global_batch_size

        self.predict_with_projection = False

    def on_train_start(self):
        self.print(f"Training model: {self.hparams.experiment_name}")

    def init_model(self, hidden_size, depth, jk="cat"):
        from phenocompass.models.components import GNEpropGIN

        return GNEpropGIN(
            in_channels=133,
            edge_dim=12,
            hidden_channels=hidden_size,
            ffn_hidden_channels=None,
            num_layers=depth,
            out_channels=1,
            dropout=0.13,
            num_readout_layers=1,
            mol_features_size=0,
            aggr="mean",
            jk=jk,
        )

    def _add_default_args(self, kwargs):
        model_parser = self.add_model_specific_args(ArgumentParser())
        default_args = model_parser.parse_known_args()[0]._get_kwargs()
        for k, v in default_args:
            if k not in kwargs:
                kwargs[k] = v

    def forward(self, data):
        o = self.encoder.compute_representations(
            data.x, data.edge_index, data.edge_attr, data.batch
        )
        if self.predict_with_projection:
            o = self.projection(o)
        return o

    def shared_step(self, batch):
        data1, data2 = batch

        # get h representations
        h1 = self.encoder.compute_representations(
            data1.x, data1.edge_index, data1.edge_attr, data1.batch
        )
        h2 = self.encoder.compute_representations(
            data2.x, data2.edge_index, data2.edge_attr, data2.batch
        )

        # get z representations
        z1 = self.projection(h1)
        z2 = self.projection(h2)

        loss = self.nt_xent_loss(z1, z2)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log(
            "train_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, sync_dist=True, prog_bar=True, logger=True
        )
        return loss

    def exclude_from_wt_decay(self, named_params, weight_decay, skip_list=["bias", "bn"]):
        params = []
        excluded_params = []

        for name, param in named_params:
            if not param.requires_grad:
                continue
            elif any(layer_name in name for layer_name in skip_list):
                excluded_params.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": weight_decay},
            {"params": excluded_params, "weight_decay": 0.0},
        ]

    def configure_optimizers(self):
        if self.hparams.exclude_bn_bias:
            params = self.exclude_from_wt_decay(
                self.named_parameters(), weight_decay=self.hparams.weight_decay
            )
        else:
            params = self.parameters()

        if self.hparams.optim == "adam":
            optimizer = torch.optim.Adam(
                params, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
            )

        warmup_steps = self.train_iters_per_epoch * self.hparams.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.hparams.max_epochs

        scheduler = None

        return [optimizer], [scheduler]

    def nt_xent_loss(self, out_1, out_2, temperature=0.1, eps=1e-6):
        """NT-Xent loss for contrastive learning.

        Assumes out_1 and out_2 are normalized.
        out_1: [batch_size, dim]
        out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            out_1_dist = SyncFunction.apply(out_1)
            out_2_dist = SyncFunction.apply(out_2)
        else:
            out_1_dist = out_1
            out_2_dist = out_2

        # out: [2 * batch_size, dim]
        # out_dist: [2 * batch_size * world_size, dim]
        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        # cov and sim: [2 * batch_size, 2 * batch_size * world_size]
        # neg: [2 * batch_size]
        cov = torch.mm(out, out_dist.t().contiguous())
        sim = torch.exp(cov / temperature)
        neg = sim.sum(dim=-1)

        # from each row, subtract e^(1/temp) to remove similarity measure for x1.x1
        row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=eps)  # clamp for numerical stability

        # Positive similarity, pos becomes [2 * batch_size]
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        loss = -torch.log(pos / (neg + eps)).mean()

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", type=int, default=256)
        parser.add_argument(
            "--exclude_bn_bias", action="store_true", help="exclude bn/bias from weight decay"
        )
        parser.add_argument("--optim", default="adam", type=str, help="choose between adam/lars")
        parser.add_argument("--warmup_epochs", default=5, type=int, help="number of warmup epochs")
        parser.add_argument("--lr", type=float, default=1.0e-05)
        parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")
        parser.add_argument("--model_hidden_size", default=300, type=int)
        parser.add_argument("--project_hidden_dim", default=512, type=int)
        parser.add_argument("--project_output_dim", default=128, type=int)
        parser.add_argument("--model_depth", default=5, type=int)
        parser.add_argument(
            "--jk", default="cat", const="cat", nargs="?", choices=["cat", "none"]
        )
        parser.add_argument("--linear_projection", action="store_true")
        parser.add_argument("--train_resample_number", default=0, type=int)
        parser.add_argument(
            "--log_directory", type=str, default="gneprop_clr/logs", help="Where logs are saved"
        )
        parser.add_argument("--linear_evaluation_dataset", type=str, default=None)
        parser.add_argument("--shared_memory", action="store_true")
        return parser
