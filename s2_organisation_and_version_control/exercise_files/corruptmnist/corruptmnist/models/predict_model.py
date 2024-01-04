import numpy as np
import torch
import os
import sys
import random
import hydra
import logging

log = logging.getLogger(__name__)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models.model import MyAwesomeModel
from data.make_dataset import mnist


@hydra.main(config_path="../config", config_name="predict_config.yaml")
def predict(config):
    """Run prediction for a given model and dataloader.

    Args:
        model_checkpoint: model to use for prediction
        data_path: path to pickled numpy array of images

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)
    
    
    model_checkpoint = hydra.utils.to_absolute_path(config.model_checkpoint)
    data_path = hydra.utils.to_absolute_path(config.data_path)
    
    log.info("Evaluating like my life dependends on it")
    log.info(model_checkpoint)

    # TODO: Implement evaluation logic here
    # model = torch.load(model_checkpoint)
    model = MyAwesomeModel(
        config.model.x_dim,
        config.model.hidden_dim,
        config.model.latent_dim,
        config.model.output_dim,
        config.model.kernel_size,
        config.model.padding,
        config.model.dropout
    )
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Convert data to torch tensor
    images = np.load(data_path, allow_pickle=True)
    images = torch.from_numpy(images)

    with torch.no_grad():
        outputs = model(images)
    
    log.info(outputs.shape)

    return outputs


if __name__ == "__main__":
    predict()
