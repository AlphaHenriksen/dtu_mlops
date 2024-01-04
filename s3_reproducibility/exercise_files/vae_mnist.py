"""Adapted from https://github.com/Jackson-Kang/Pytorch-VAE-tutorial/blob/master/01_Variational_AutoEncoder.ipynb.

A simple implementation of Gaussian MLP Encoder and Decoder trained on MNIST
"""
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from model import Decoder, Encoder, Model
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.utils import save_image
import random
import numpy as np
import hydra
import logging


log = logging.getLogger(__name__)


@hydra.main(config_name="config.yaml")
def train(config):
    # Model Hyperparameters
    torch.manual_seed(config.hyperparameters.seed)
    random.seed(config.hyperparameters.seed)
    np.random.seed(config.hyperparameters.seed)
    cuda = False
    DEVICE = torch.device("cuda" if cuda else "cpu")

    dataset_path = "~/datasets"


    # Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.hyperparameters.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.hyperparameters.batch_size, shuffle=False)

    encoder = Encoder(input_dim=config.hyperparameters.x_dim, 
                      hidden_dim=config.hyperparameters.hidden_dim, 
                      latent_dim=config.hyperparameters.latent_dim)

    decoder = Decoder(latent_dim=config.hyperparameters.latent_dim, 
                      hidden_dim=config.hyperparameters.hidden_dim, 
                      output_dim=config.hyperparameters.x_dim)

    model = Model(encoder=encoder, decoder=decoder).to(DEVICE)


    # This is probably a hyperparameter
    optimizer = Adam(model.parameters(), lr=config.hyperparameters.lr)


    def loss_function(x, x_hat, mean, log_var):
        """Elbo loss function."""
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
        kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + kld


    log.info("Start training VAE...")
    model.train()
    for epoch in range(config.hyperparameters.epochs):
        overall_loss = 0
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                log.info(batch_idx)
            x = x.view(config.hyperparameters.batch_size, config.hyperparameters.x_dim)
            x = x.to(DEVICE)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
        log.info(f"Epoch {epoch+1} complete!,  Average Loss: {overall_loss / (batch_idx*config.hyperparameters.batch_size)}")
    log.info("Finish!!")

    # save weights
    torch.save(model, f"{os.getcwd()}/trained_model.pt")

    # Generate reconstructions
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, _) in enumerate(test_loader):
            if batch_idx % 100 == 0:
                log.info(batch_idx)
            x = x.view(config.hyperparameters.batch_size, config.hyperparameters.x_dim)
            x = x.to(DEVICE)
            x_hat, _, _ = model(x)
            break

    save_image(x.view(config.hyperparameters.batch_size, 1, config.hyperparameters.image_size, config.hyperparameters.image_size), 
               "orig_data.png")
    save_image(x_hat.view(config.hyperparameters.batch_size, 1, config.hyperparameters.image_size, config.hyperparameters.image_size), 
               "reconstructions.png")

    # Generate samples
    with torch.no_grad():
        noise = torch.randn(config.hyperparameters.batch_size, config.hyperparameters.latent_dim).to(DEVICE)
        generated_images = decoder(noise)

    save_image(generated_images.view(config.hyperparameters.batch_size, 
                                     1, 
                                     config.hyperparameters.image_size, 
                                     config.hyperparameters.image_size), 
               "generated_sample.png")

if __name__ == "__main__":
    train()