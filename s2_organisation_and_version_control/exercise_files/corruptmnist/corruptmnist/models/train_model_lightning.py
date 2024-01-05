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
from pytorch_lightning import LightningModule
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

    with wandb.init(project="corruptmnist", config=wandb_config):
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
        
        train_set, test_set = mnist(config.train_batch_size, config.test_batch_size)

        # Training
        trainer = Trainer(
            accelerator="cpu",
            check_val_every_n_epoch=1,
            default_root_dir=os.getcwd(),
            max_epochs=10,
            limit_train_batches=0.20
            )
        trainer.fit(model, train_set)
        # trainer.test(test_set)

        # # Training loop
        # for epoch in range(config.epochs):
        #     running_loss = 0
        #     correct = 0
        #     total = 0

        #     for images, targets in train_set:
        #         # Forward pass
        #         optimizer.zero_grad()
        #         outputs = model(images)

        #         # Backward pass
        #         loss = criterion(outputs, targets)
        #         loss.backward()
        #         optimizer.step()

        #         running_loss += loss.item()

        #         # Accuracy calculation
        #         _, predicted = torch.max(outputs.data, 1)
        #         total += targets.size(0)
        #         correct += (predicted == targets).sum().item()

        #     # Get metrics
        #     accuracy = 100 * correct / total
        #     loss = running_loss / len(train_set)
        #     accuracies.append(accuracy)
        #     losses.append(loss)

        #     log.info(f"Epoch {epoch+1} of {config.epochs}")
        #     log.info(f"Training loss: {loss}")
        #     log.info(f"Accuracy: {accuracy}\n")
        #     wandb.log({"Loss": loss})
        #     wandb.log({"Accuracy": accuracy})

        # # Do plotting
        # plt.plot(losses, label="loss")
        # plt.plot(accuracies, label="accuracy")
        # plt.xlabel("Epoch")
        # plt.ylabel("Score")
        # # plt.savefig(f"{visualization_path}/viz.png")
        # plt.savefig(f"viz.png")
        # fig = plt.gcf()
        # # wandb.log({"acc_curve": wandb.Image(fig)})
        # wandb.log({"acc_plot": fig})
        # wandb.finish()
        # # plt.show()

        # log.info("Saving final model.")
        # # torch.save(model.state_dict(), f"{model_path}/checkpoint.pth")
        # torch.save(model.state_dict(), f"checkpoint.pth")


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
    