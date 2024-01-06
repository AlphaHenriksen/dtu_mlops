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
import pytest
from dotwiz import DotWiz
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from rich.logging import RichHandler
from pytorch_lightning import Trainer
from types import SimpleNamespace

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


class TestModel:

    @pytest.fixture(autouse=True) 
    def _setup(self): 
        config = {"train_batch_size": 32,
                "test_batch_size": 10000,
                "learning_rate": 0.001,
                "epochs": 3,
                "seed": 123,
                "dataset": "mnist", 
                "optimizer": {"name": "adam"},
                "model": {
                    "hidden_dim": 32,
                    "latent_dim": 64,
                    "output_dim": 10,
                    "x_dim": 784,
                    "kernel_size": 3,
                    "padding": 1,
                    "dropout": 0.2
                    }
                }
        config = DotWiz(config)
        self.config = config
        self.model = MyAwesomeModel(self.config)
    
    
    def test_optimizer(self):
        
        self.config.optimizer.name = "badam"
        with pytest.raises(ValueError):
            model = MyAwesomeModel(self.config)
        self.config.optimizer.name = "adam"


    def test_dimensions(self):
        assert self.model.layer1.in_channels == 1, "Incorrect input channels in layer1"

        # Test output channels of the first convolutional layer
        assert self.model.layer1.out_channels == self.model.hidden_dim, "Incorrect output channels in layer1"

        # Test output channels of the second convolutional layer
        assert self.model.layer2.out_channels == self.model.latent_dim, "Incorrect output channels in layer2"

        # Verify the number of input features to the linear layer
        expected_input_features = self.model.latent_dim * self.model.x_dim
        actual_input_features = self.model.layer3.in_features
        assert actual_input_features == expected_input_features, "Incorrect number of input features in layer3"
        # self.config.model.kernel_size = 5
        # with pytest.raises(ValueError):
        #     model = MyAwesomeModel(self.config)
        # self.config.model.kernel_size = 3

        # self.config.model.padding = 4
        # with pytest.raises(ValueError):
        #     model = MyAwesomeModel(self.config)
        # self.config.model.padding = 1

        # self.config.model.padding = 0
        # with pytest.raises(ValueError):
        #     model = MyAwesomeModel(self.config)
        # self.config.model.padding = 1
        
        # model = MyAwesomeModel(self.config)
        
        # x_dim_1d = np.sqrt(self.config.model.x_dim).astype(int)
        
        # print(np.shape(model(torch.empty((x_dim_1d, x_dim_1d, 1, 1), dtype=torch.int64))))
        # assert np.shape(model(torch.empty((x_dim_1d, x_dim_1d, 1, 1), dtype=torch.int64))) == (self.config.model.output_dim, self.config.model.output_dim)