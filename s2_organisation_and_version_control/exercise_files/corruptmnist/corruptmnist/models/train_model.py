import torch
import os
import sys
import matplotlib.pyplot as plt
import hydra
import random
import numpy as np
import logging
import logging.config
from rich.logging import RichHandler

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models.model import MyAwesomeModel
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

    model = MyAwesomeModel(
        config.model.x_dim,
        config.model.hidden_dim,
        config.model.latent_dim,
        config.model.output_dim,
        config.model.kernel_size,
        config.model.padding,
        config.model.dropout
        )
    train_set, _ = mnist(config.train_batch_size, config.test_batch_size)

    if config.optimizer.name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer.name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.learning_rate)
    elif config.optimizer.name == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)
    else:
        raise ValueError(f"Optimizer {config.optimizer.name} not supported. Please choose one of [adam, sgd, rmsprop].")

    criterion = torch.nn.CrossEntropyLoss()

    accuracies = []
    losses = []

    # Training loop
    for epoch in range(config.epochs):
        running_loss = 0
        correct = 0
        total = 0

        for images, targets in train_set:
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)

            # Backward pass
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Get metrics
        accuracy = 100 * correct / total
        loss = running_loss / len(train_set)
        accuracies.append(accuracy)
        losses.append(loss)

        log.info(f"Epoch {epoch+1} of 10")
        log.info(f"Training loss: {loss}")
        log.info(f"Accuracy: {accuracy}\n")

    # Do plotting
    plt.plot(losses, label="loss")
    plt.plot(accuracies, label="accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    # plt.savefig(f"{visualization_path}/viz.png")
    plt.savefig(f"viz.png")
    plt.show()

    log.info("Saving final model.")
    # torch.save(model.state_dict(), f"{model_path}/checkpoint.pth")
    torch.save(model.state_dict(), f"checkpoint.pth")


if __name__ == "__main__":
    train()
    
    # The file should be run from the home directory of the project (the first corruptmnist folder)
    
    # Run multirun
    # python corruptmnist/models/train_model.py learning_rate=0.001,0.0001 optimizer.name=sgd,adam,rmsprop --multirun