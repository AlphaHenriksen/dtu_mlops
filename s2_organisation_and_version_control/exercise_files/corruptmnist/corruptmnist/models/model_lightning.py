from torch import nn
from pytorch_lightning import LightningModule
import torch


class MyAwesomeModel(LightningModule):
    """My awesome model."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.x_dim = config.model.x_dim
        self.hidden_dim = config.model.hidden_dim
        self.latent_dim = config.model.latent_dim
        self.output_dim = config.model.output_dim
        self.kernel_size = config.model.kernel_size
        self.padding = config.model.padding
        self.dropout = config.model.dropout
        
        self.layer1 = nn.Conv2d(1, self.hidden_dim, self.kernel_size, padding=self.padding)
        self.layer2 = nn.Conv2d(self.hidden_dim, self.latent_dim, self.kernel_size, padding=self.padding)
        self.layer3 = nn.Linear(self.latent_dim * self.x_dim, self.output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=self.dropout)
        self.flatten = nn.Flatten()
        
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.layer3(x)
        return x

    def training_step(self, batch):
        data, target = batch
        outputs = self(data)
        loss = self.criterion(outputs, target)
        return loss

    def configure_optimizers(self):
        if self.config.optimizer.name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer.name == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer.name == 'rmsprop':
            return torch.optim.RMSprop(self.parameters(), lr=self.config.learning_rate)
        else:
            raise ValueError(f"Optimizer {self.config.optimizer.name} not supported. Please choose one of [adam, sgd, rmsprop].")
