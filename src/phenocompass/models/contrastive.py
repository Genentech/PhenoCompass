"""PyTorch Lightning modules for multimodal contrastive learning."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.nn import ModuleDict

from phenocompass.models.components import CommonEncoder, JointEncoder, MultiTask_model
from phenocompass.models.loss import get_loss
from phenocompass.models.utils import move_batch_input_to_device
from phenocompass.training.utils import LARS
from tqdm import tqdm


class MultiModalContrastive_PL(pl.LightningModule):
    """Base class for multimodal contrastive learning.

    Parameters
    ----------
    loss_name : str
        Name of the loss function.
    encoders_mod : dict
        Dictionary of encoder modules for each modality.
    projectors_mod : dict
        Dictionary of projector modules for each modality.
    temperature : float
        Temperature parameter for contrastive loss.
    lr : float
        Learning rate.
    use_weighted_loss : bool
        Whether to use learnable weights for multi-task loss.
    share_weights : bool, default False
        Whether to share weights across modalities.
    optimizer : str, default "adam"
        Optimizer type ("adam" or "lars").
    """

    def __init__(
        self,
        loss_name,
        encoders_mod,
        projectors_mod,
        temperature,
        lr,
        use_weighted_loss,
        share_weights=False,
        optimizer="adam",
    ):
        super().__init__()

        self.temperature = temperature
        self.lr = lr
        self.loss = get_loss(name=loss_name)
        if isinstance(encoders_mod, DictConfig):
            encoders_mod = OmegaConf.to_object(encoders_mod)
        if isinstance(projectors_mod, DictConfig):
            projectors_mod = OmegaConf.to_object(projectors_mod)
        self.encoders = ModuleDict()
        self.encoders.update(encoders_mod)
        self.projectors = ModuleDict()
        self.projectors.update(projectors_mod)
        if use_weighted_loss is True:
            self.alpha = nn.ParameterDict(
                {
                    "learnable_w": nn.Parameter(torch.randn(len(encoders_mod))),
                }
            )
        else:
            self.alpha = None
        self.share_weights = share_weights
        self.optimizer = optimizer

    def configure_optimizers(self):
        params = self.parameters()
        if self.optimizer == "adam":
            return torch.optim.Adam(params, lr=self.lr)
        elif self.optimizer == "lars":
            return LARS(params, lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")

    def _mod_encode(self, x_mod, encoder_mod, proj, **kwargs):
        raise NotImplementedError

    def compute_representations(self, x_dict, mod_name=None):
        """Compute representations for input modalities.

        Parameters
        ----------
        x_dict : dict
            Dictionary mapping modality names to inputs.
        mod_name : str, optional
            If provided, only compute representation for this modality.

        Returns
        -------
        dict or torch.Tensor
            Representations for all modalities or single modality.
        """
        if mod_name is not None:
            if not self.share_weights:
                return self._mod_encode(
                    x_dict[mod_name], self.encoders[mod_name], self.projectors[mod_name]
                )
            else:
                joint_mod = "joint"
                encoder_representations = {}
                for mod_name, x in x_dict.items():
                    if mod_name != joint_mod:
                        _, encoder_representation = self._mod_encode(
                            x_dict[mod_name],
                            self.encoders[mod_name],
                            self.projectors[mod_name],
                            return_encoder_output=True,
                        )
                        encoder_representations[mod_name] = encoder_representation
                return self._joint_encode(encoder_representations, self.projectors[joint_mod])
        else:
            batch_representations = {}
            encoder_representations = {}
            if not self.share_weights:
                for mod_name, x in x_dict.items():
                    mod_representations = self._mod_encode(
                        x_dict[mod_name], self.encoders[mod_name], self.projectors[mod_name]
                    )
                    batch_representations[mod_name] = mod_representations
            else:
                joint_mod = "joint"
                assert joint_mod in x_dict.keys()
                for mod_name, x in x_dict.items():
                    if mod_name != joint_mod:
                        mod_representations, encoder_representation = self._mod_encode(
                            x_dict[mod_name],
                            self.encoders[mod_name],
                            self.projectors[mod_name],
                            return_encoder_output=True,
                        )
                        encoder_representations[mod_name] = encoder_representation
                        batch_representations[mod_name] = mod_representations
                batch_representations[joint_mod] = self._joint_encode(
                    encoder_representations, self.projectors[joint_mod]
                )

            return batch_representations

    def forward(self, batch, mod_name=None):
        x_dict = batch["inputs"]
        return self.compute_representations(x_dict, mod_name=mod_name)

    def compute_representation_dataloader(
        self,
        dataloader,
        device=None,
        mod_name=None,
        disable_progress_bar=False,
        return_mol=False,
    ):
        """Compute representations for all samples in a dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            Input dataloader.
        device : torch.device, optional
            Device for computation.
        mod_name : str, optional
            Modality name to compute.
        disable_progress_bar : bool, default False
            Whether to disable progress bar.
        return_mol : bool, default False
            Whether to return molecule identifiers.

        Returns
        -------
        dict
            Dictionary mapping modality names to representation arrays.
        """
        outs = []
        mols = []

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for batch in tqdm(dataloader, position=0, leave=True, disable=disable_progress_bar):
                x_dict = batch["inputs"]
                x_dict = move_batch_input_to_device(x_dict, device=device)
                out = self.compute_representations(x_dict, mod_name=mod_name)
                if isinstance(out, List):
                    outs_list = tuple([_.cpu().data.numpy() for _ in out])
                    outs.append(outs_list)
                elif isinstance(out, dict):
                    outs.append({k: v.cpu().data.numpy() for k, v in out.items()})
                else:
                    outs.append(out.cpu().data.numpy())

                if return_mol:
                    mols.extend([*batch["inputs"]["struct"].mols])

        final_outs = dict()
        if isinstance(outs[0], Tuple):
            num_mods = len(outs[0])
        elif isinstance(outs[0], dict):
            num_mods = len(outs[0])
        else:
            num_mods = 1

        if num_mods == 1:
            final_outs[mod_name] = np.vstack(outs)
        else:
            if isinstance(outs[0], Tuple):
                for idx in range(num_mods):
                    final_outs[mod_name] = np.vstack(list(zip(*outs))[idx])
            elif isinstance(outs[0], dict):
                for mod_name in outs[0].keys():
                    final_outs[mod_name] = np.vstack([out[mod_name] for out in outs])

        if return_mol:
            return final_outs, mols
        else:
            return final_outs

    def _step(self, batch, batch_idx, step_name, dataloader_idx=None):
        batch_size = batch["labels"].shape[0]
        batch_representations = self.forward(batch)

        if self.alpha is not None:
            learnable_w = self.alpha["learnable_w"].softmax(-1)
        else:
            learnable_w = None

        loss = self.loss(
            batch_representations, self.temperature, batch_size, learnable_w=learnable_w
        )

        return loss

    def training_step(self, train_batch, batch_idx):
        loss = self._step(batch=train_batch, batch_idx=batch_idx, step_name="train")

        batch_size = train_batch["labels"].shape[0]

        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, val_batch, batch_idx, dataloader_idx=None):
        loss = self._step(
            batch=val_batch,
            batch_idx=batch_idx,
            step_name="val",
            dataloader_idx=dataloader_idx,
        )

        batch_size = val_batch["labels"].shape[0]

        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        return loss


class TripleContrastive_PL(MultiModalContrastive_PL):
    """Triple contrastive learning module (GMC with PUMA dataset).

    Parameters
    ----------
    loss_name : str
        Name of the loss function.
    encoders_mod : dict
        Dictionary of encoder modules.
    encoder_joint : JointEncoder
        Joint encoder module.
    projectors_mod : dict
        Dictionary of projector modules.
    common_encoder : CommonEncoder
        Common encoder for all modalities.
    temperature : float
        Temperature parameter.
    lr : float
        Learning rate.
    share_weights : bool, default False
        Whether to share weights.
    optimizer : str, default "adam"
        Optimizer type.
    use_weighted_loss : bool, default False
        Whether to use weighted loss.
    """

    def __init__(
        self,
        loss_name,
        encoders_mod,
        encoder_joint,
        projectors_mod,
        common_encoder,
        temperature,
        lr,
        share_weights=False,
        optimizer="adam",
        use_weighted_loss=False,
        **kwargs,
    ):
        super().__init__(
            loss_name,
            encoders_mod,
            projectors_mod,
            temperature,
            lr,
            use_weighted_loss,
            share_weights,
            optimizer,
        )
        self.save_hyperparameters()
        if not share_weights:
            self.encoder_joint = encoder_joint
            self.encoders.update({"joint": self.encoder_joint})
        self.common_encoder = common_encoder

    def _mod_encode(self, x_mod, encoders_mod, proj, return_encoder_output=False):
        emb_mod_ = encoders_mod(x_mod)
        emb_mod = proj(emb_mod_)
        mod_representations = self.common_encoder(emb_mod)

        if return_encoder_output:
            return mod_representations, emb_mod_
        else:
            return mod_representations

    def _joint_encode(self, encoder_representation, proj):
        rep_list = []
        with torch.no_grad():
            for key, value in encoder_representation.items():
                rep_list.append(value)
            reps = torch.concat(rep_list, dim=-1)
        emb_mod = proj(reps)
        mod_representations = self.common_encoder(emb_mod)
        return mod_representations


class DualContrastive_PL(MultiModalContrastive_PL):
    """Dual contrastive learning module (MOCOP with JUMP dataset)."""

    def __init__(
        self,
        loss_name,
        encoders_mod,
        projectors_mod,
        temperature,
        lr,
        use_weighted_loss=False,
        **kwargs,
    ):
        super().__init__(
            loss_name, encoders_mod, projectors_mod, temperature, lr, use_weighted_loss
        )
        self.save_hyperparameters()

    def _mod_encode(self, x_mod, encoders_mod, proj):
        emb_mod_ = encoders_mod(x_mod)
        emb_mod = proj(emb_mod_)
        return emb_mod


class MultiTask_PL(pl.LightningModule):
    """Multi-task learning module with frozen backbone."""

    def __init__(
        self,
        loss_name,
        num_tasks,
        lr,
        backbone=None,
        ckp=None,
        backbone_name=None,
        mod_name="struct",
    ):
        super().__init__()
        self.mod_name = mod_name
        if backbone is None:
            if backbone_name == "gmc":
                backbone = TripleContrastive_PL.load_from_checkpoint(ckp)
        self.model = MultiTask_model(
            backbone=backbone,
            loss_name=loss_name,
            num_tasks=num_tasks,
            mod_name=self.mod_name,
            lr=lr,
        )
        self.save_hyperparameters()

    def configure_optimizers(self):
        return self.model.optimizer

    def _step(self, batch, batch_idx, step_name, dataloader_idx=None):
        label = batch["labels"]
        probs, logits = self.model._forward_with_sigmoid(
            batch, mod_name=self.mod_name, return_mod="logits"
        )
        loss = self.model.loss(logits, label)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch=batch, batch_idx=batch_idx, step_name="train")

        batch_size = batch["labels"].shape[0]

        self.log(
            "train/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(
            batch=batch,
            batch_idx=batch_idx,
            step_name="val",
        )

        batch_size = batch["labels"].shape[0]

        self.log(
            "val/loss",
            loss,
            sync_dist=True,
            prog_bar=True,
            logger=True,
            batch_size=batch_size,
        )

        return loss


class TripleContrastive_PL_Neighbors(MultiModalContrastive_PL):
    """Triple contrastive learning with soft nearest neighbors."""

    def __init__(
        self,
        loss_name,
        encoders_mod,
        encoder_joint,
        projectors_mod,
        common_encoder,
        temperature,
        lr,
        neighbor_thresh=None,
        use_weighted_loss=False,
        **kwargs,
    ):
        super().__init__(
            loss_name, encoders_mod, projectors_mod, temperature, lr, use_weighted_loss
        )
        self.save_hyperparameters()
        self.encoder_joint = encoder_joint
        self.encoders.update({"joint": self.encoder_joint})
        self.common_encoder = common_encoder
        self.neighbor_thresh = neighbor_thresh

    def _mod_encode(self, x_mod, encoders_mod, proj):
        emb_mod_ = encoders_mod(x_mod)
        emb_mod = proj(emb_mod_)
        mod_representations = self.common_encoder(emb_mod)
        return mod_representations

    def _compute_pairwise_similarity(self, batch):
        feat = batch["inputs"]["morph"]
        cossim = F.cosine_similarity(feat.unsqueeze(0), feat.unsqueeze(1), dim=2)

        if self.neighbor_thresh is None:
            return cossim
        else:
            return (cossim > self.neighbor_thresh).float()

    def _step(self, batch, batch_idx, step_name, dataloader_idx=None):
        batch_size = batch["labels"].shape[0]
        batch_representations = self.forward(batch)
        pairwise_similarity = self._compute_pairwise_similarity(batch)

        if self.alpha is not None:
            learnable_w = self.alpha["learnable_w"].softmax(-1)
        else:
            learnable_w = None

        loss = self.loss(
            batch_representations,
            pairwise_similarity,
            self.temperature,
            batch_size,
            learnable_w=learnable_w,
        )

        return loss


class TripleContrastive_PL_Batch(MultiModalContrastive_PL):
    """Triple contrastive learning compatible with CCL (conditional contrastive learning)."""

    def __init__(
        self,
        loss_name,
        encoders_mod,
        encoder_joint,
        projectors_mod,
        common_encoder,
        temperature,
        lr,
        use_weighted_loss=False,
        **kwargs,
    ):
        super().__init__(
            loss_name, encoders_mod, projectors_mod, temperature, lr, use_weighted_loss
        )
        self.save_hyperparameters()
        self.encoder_joint = encoder_joint
        self.encoders.update({"joint": self.encoder_joint})
        self.common_encoder = common_encoder

    def _mod_encode(self, x_mod, encoders_mod, proj):
        if not isinstance(x_mod, dict):
            x_mod, _ = x_mod
        emb_mod_ = encoders_mod(x_mod)
        emb_mod = proj(emb_mod_)
        mod_representations = self.common_encoder(emb_mod)
        return mod_representations

    def _step(self, batch, batch_idx, step_name, dataloader_idx=None):
        batch_size = batch["labels"].shape[0]
        batch_representations = self.forward(batch)
        batch_meta = batch["inputs"]["morph"][1]

        if self.alpha is not None:
            learnable_w = self.alpha["learnable_w"].softmax(-1)
        else:
            learnable_w = None

        loss = self.loss(
            batch_representations,
            batch_meta,
            self.temperature,
            batch_size,
            learnable_w=learnable_w,
        )

        return loss

    def compute_representations(self, x_dict, mod_name=None):
        if mod_name is not None:
            return self._mod_encode(
                x_dict[mod_name], self.encoders[mod_name], self.projectors[mod_name]
            )

        else:
            batch_representations = {}
            for mod_name, x in x_dict.items():
                mod_representations = self._mod_encode(
                    x_dict[mod_name], self.encoders[mod_name], self.projectors[mod_name]
                )
                batch_representations[mod_name] = mod_representations
            return batch_representations

    def compute_representation_dataloader(
        self,
        dataloader,
        device=None,
        mod_name=None,
        disable_progress_bar=False,
        return_mol=False,
    ):
        outs = []
        mols = []

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            for batch in tqdm(dataloader, position=0, leave=True, disable=disable_progress_bar):
                x_dict = batch["inputs"]
                x_dict = move_batch_input_to_device(x_dict, device=device)
                out = self.compute_representations(x_dict, mod_name=mod_name)
                if isinstance(out, List):
                    outs_list = tuple([_.cpu().data.numpy() for _ in out])
                    outs.append(outs_list)
                elif isinstance(out, dict):
                    outs.append({k: v.cpu().data.numpy() for k, v in out.items()})
                else:
                    outs.append(out.cpu().data.numpy())

                if return_mol:
                    mols.extend([*batch["inputs"]["struct"].mols])

        final_outs = dict()
        if isinstance(outs[0], Tuple):
            num_mods = len(outs[0])
        elif isinstance(outs[0], dict):
            num_mods = len(outs[0])
        else:
            num_mods = 1

        if num_mods == 1:
            final_outs[mod_name] = np.vstack(outs)
        else:
            if isinstance(outs[0], Tuple):
                for idx in range(num_mods):
                    final_outs[mod_name] = np.vstack(list(zip(*outs))[idx])
            elif isinstance(outs[0], dict):
                for mod_name in outs[0].keys():
                    final_outs[mod_name] = np.vstack([out[mod_name] for out in outs])

        if return_mol:
            return final_outs, mols
        else:
            return final_outs
