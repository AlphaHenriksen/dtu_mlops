import numpy as np
import click
import torch
from models.model import MyAwesomeModel

from data.make_dataset import mnist


model_path = "models/"
visualization_path = "reports/figures/"


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.argument("model_checkpoint")
@click.argument("data_path")
def predict(model_checkpoint, data_path):
    """Run prediction for a given model and dataloader.

    Args:
        model_checkpoint: model to use for prediction
        data_path: path to pickled numpy array of images

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    # model = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Convert data to torch tensor
    images = np.load(data_path, allow_pickle=True)
    images = torch.from_numpy(images)

    with torch.no_grad():
        outputs = model(images)
    
    print(outputs.shape)

    return outputs


cli.add_command(predict)


if __name__ == "__main__":
    cli()
    
    # To run:
    # python corruptmnist/predict_model.py predict models/checkpoint.pth data/processed/example_images.npy
