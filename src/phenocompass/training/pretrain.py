"""Pretraining entry point for multimodal contrastive learning."""

import logging
import warnings
from typing import List

import hydra
import numpy as np
import omegaconf
from omegaconf import OmegaConf
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)

from phenocompass.evaluation.metrics import RetrievalOnlineEvaluator
from phenocompass.training import utils

logging.getLogger().setLevel(logging.INFO)
warnings.filterwarnings(
    "ignore", category=UserWarning, message="TypedStorage is deprecated"
)

OmegaConf.register_new_resolver("sum", lambda input_list: np.sum(input_list))


@hydra.main(config_path="../../configs", config_name="pretrain_scaffold", version_base=None)
def train(cfg):
    """Main training function.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration.
    """
    if "wandb" in cfg:
        import wandb

        config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            group=cfg.group_name,
            config=config,
        )

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        logging.info(f"Seed everything with <{cfg.seed}>")
        seed_everything(cfg.seed, workers=True)

    # Init lightning datamodule
    logging.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    # Init lightning module
    model: LightningModule = utils.instantiate_model(cfg)

    # Init callbacks
    logging.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg)

    # Init evaluator
    logging.info(f"Instantiating evaluation callback... <{cfg.evaluation._target_}>")
    evaluator: RetrievalOnlineEvaluator = hydra.utils.instantiate(cfg.evaluation, model=model)

    # Add evaluator to callbacks
    callbacks = [evaluator] + callbacks

    # Init lightning trainer
    logging.info(f"Instantiating trainer... <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks)

    trainer.fit(model, datamodule, ckpt_path=cfg.get("ckpt_path"))

    # Send some parameters from config to all lightning loggers
    log_dict = {
        "cfg": cfg,
        "model": model,
        "trainer": trainer,
    }
    logging.info("Logging hyperparameters!")
    utils.log_hyperparameters(log_dict)


if __name__ == "__main__":
    train()
