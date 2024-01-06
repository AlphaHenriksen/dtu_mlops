# TODO: S4 M14 12. Lightning CLI implementation
# https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html
# TODO: S4 M14 11. Lightning profiler
# https://lightning.ai/docs/pytorch/latest/tuning/profiler.html

# TODO: Check out:
#     torchmetrics
#     https://lightning.ai/docs/torchmetrics/stable/
#     lightning flash
#     https://lightning-flash.readthedocs.io/en/latest/
#     lightning bolts
#     https://lightning-bolts.readthedocs.io/en/latest/

import torch
import os
import sys
import matplotlib.pyplot as plt
import hydra
import random
import numpy as np
import logging
import logging.config
import wandb
import yaml
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from rich.logging import RichHandler
from pytorch_lightning import Trainer

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models.model_lightning import MyAwesomeModel
from data.make_dataset import mnist

# log = logging.getLogger('development')
log = logging.getLogger(__name__)
# log.handlers = [RichHandler(markup=True)]  # set rich handler

model_path = hydra.utils.to_absolute_path('models')
visualization_path = hydra.utils.to_absolute_path('reports/figures')

@hydra.main(config_path="../config", config_name="train_config.yaml")
def train(config):
    """
    Using the corrupted MNIST dataset, train a cnn model.

    Parameters:
        lr (float): learning rate for the optimizer.

    Returns:
        None.
    """

    # Kind of backwards way of getting the hyperparameters fed into wandb
    with open(hydra.utils.to_absolute_path("corruptmnist/config/train_config.yaml"), "r") as file:
        wandb_config = yaml.safe_load(file)


    # Set the seeds
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    # Just to test that the different levels of logging works
    log.debug("Used for debugging your code.")
    log.info("Informative messages from your code.")
    log.warning("Everything works but there is something to be aware of.")
    log.error("There's been a mistake with the process.")
    log.critical("There is something terribly wrong and process may terminate.\n")
    
    log.info("Training day and night")
    log.info(config.learning_rate)

    model = MyAwesomeModel(config)
    
    train_set, validation_set, test_set = mnist(config.train_batch_size, config.test_batch_size, 5, 1)

    # Training
    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="validation_accuracy", mode="max")
    trainer = Trainer(
        accelerator="cpu",
        check_val_every_n_epoch=1,
        default_root_dir=os.getcwd(),
        max_epochs=10,
        limit_train_batches=0.20,
        callbacks=[checkpoint_callback],
        logger=pl.loggers.WandbLogger(project="corruptmnist_lightning", config=wandb_config),
        precision="32",
        )
    trainer.fit(model, train_set, validation_set)
    trainer.test(model, test_set, ckpt_path="best")


if __name__ == "__main__":
    train()
    
    # The file should be run from the home directory of the project (the first corruptmnist folder)
    
    # Run multirun using hydra and yaml
    # python corruptmnist/models/train_model.py learning_rate=0.001,0.0001 optimizer.name=sgd,adam,rmsprop --multirun
    
    # Wandb
    # Run sweeps with wandb
    # wandb login --relogin
    # wandb sweep --project <project_name> corruptmnist/config/sweep.yaml
    # wandb agent <sweep-ID>
    