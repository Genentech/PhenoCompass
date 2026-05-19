"""Training utilities."""

import logging
from typing import Any, Callable, List, Optional, Sequence

import hydra
import torch
import torch.nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Callback
from pytorch_lightning.utilities import rank_zero_only
from torch.optim.optimizer import Optimizer, required

log = logging.getLogger()
log.setLevel(logging.INFO)


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling (LARS) optimizer.

    Extends SGD with LARS scaling for large batch training.

    See: Large batch training of Convolutional Networks
    https://arxiv.org/pdf/1708.03888.pdf

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize.
    lr : float
        Learning rate.
    momentum : float, default 0
        Momentum factor.
    weight_decay : float, default 0
        Weight decay (L2 penalty).
    dampening : float, default 0
        Dampening for momentum.
    nesterov : bool, default False
        Enable Nesterov momentum.
    trust_coefficient : float, default 0.001
        Trust coefficient for LARS.
    eps : float, default 1e-8
        Epsilon for numerical stability.
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.eps = eps
        self.trust_coefficient = trust_coefficient

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Parameters
        ----------
        closure : callable, optional
            A closure that reevaluates the model and returns the loss.

        Returns
        -------
        loss
            Loss value if closure provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if weight_decay != 0 and p_norm != 0 and g_norm != 0:
                    lars_lr = p_norm / (g_norm + p_norm * weight_decay + self.eps)
                    lars_lr *= self.trust_coefficient

                    d_p = d_p.add(p, alpha=weight_decay)
                    d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = d_p.add(buf, alpha=momentum) if nesterov else buf

                p.add_(d_p, alpha=-group["lr"])

        return loss


def instantiate_callbacks(cfg: DictConfig) -> List[Callback]:
    """Instantiate callbacks from config.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with callbacks section.

    Returns
    -------
    list
        List of instantiated Callback objects.
    """
    from phenocompass.models.components import MultiTask_model

    callbacks_cfg = cfg.get("callbacks")
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")

            if cb_conf._target_ == "phenocompass.evaluation.metrics.LinearProbeEvaluator":
                backbone = hydra.utils.instantiate(cfg.model)
                lp_model = MultiTask_model(
                    num_tasks=1310, loss_name="mtl_bceloss", backbone=backbone
                )
                cb_instance = hydra.utils.instantiate(cb_conf, model=lp_model)
            else:
                cb_instance = hydra.utils.instantiate(cb_conf)
            callbacks.append(cb_instance)

    return callbacks


def instantiate_model(cfg: DictConfig):
    """Instantiate Lightning model from config.

    Parameters
    ----------
    cfg : DictConfig
        Configuration with model section.

    Returns
    -------
    LightningModule
        Instantiated model.
    """
    from phenocompass.models.components import JointEncoder

    if cfg.model._target_ in [
        "phenocompass.models.contrastive.TripleContrastive_PL",
        "phenocompass.models.contrastive.TripleContrastive_PL_Neighbors",
    ]:
        # Init encoder_joint
        logging.info("Instantiating torch.nn.module JointEncoder")
        encoder_joint = JointEncoder(
            encoders_mod=hydra.utils.instantiate(cfg.model.encoders_mod)
        )

        # Init lightning model
        logging.info(f"Instantiating lightning model <{cfg.model._target_}>")
        model = hydra.utils.instantiate(
            cfg.model,
            encoder_joint=encoder_joint,
        )

    elif cfg.model._target_ == "phenocompass.models.contrastive.DualContrastive_PL":
        # Init lightning model
        logging.info(f"Instantiating lightning model <{cfg.model._target_}>")
        model = hydra.utils.instantiate(cfg.model)

    else:
        # Generic instantiation
        logging.info(f"Instantiating lightning model <{cfg.model._target_}>")
        model = hydra.utils.instantiate(cfg.model)

    # use pretrained model and freeze
    if cfg.model.get("struct_pretrain") and cfg.model.struct_pretrain is not None:
        print("Using pretrain path")
        model.encoders.struct.load_pretrained(pretrain_path=cfg.model.struct_pretrain)
        model.encoders.joint.encoders.struct.load_pretrained(
            pretrain_path=cfg.model.struct_pretrain
        )

    return model


@rank_zero_only
def log_hyperparameters(object_dict: dict) -> None:
    """Log hyperparameters to lightning loggers.

    Also saves number of model parameters.

    Parameters
    ----------
    object_dict : dict
        Dictionary with "cfg", "model", and "trainer" keys.
    """
    hparams = {}

    cfg = object_dict["cfg"]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        log.warning("Logger not found! Skipping hyperparameter logging...")
        return

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params/non_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    dict_model = OmegaConf.to_container(cfg.model)
    if dict_model.get("encoders_mod") is not None:
        for ix, module_mod in enumerate(dict_model["encoders_mod"]):
            hparams[f"model/encoders_mod/{ix}"] = module_mod
    if dict_model.get("projectors_mod") is not None:
        for ix, module_mod in enumerate(dict_model["projectors_mod"]):
            hparams[f"model/projectors_mod/{ix}"] = module_mod

    # save others
    hparams["datamodule"] = OmegaConf.to_container(cfg.datamodule)
    hparams["trainer"] = OmegaConf.to_container(cfg.trainer)
    hparams["ckp_path"] = trainer.checkpoint_callback.best_model_path

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)
