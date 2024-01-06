import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import numpy as np
import hydra
import matplotlib.pyplot as plt
import os
import sys
import pytest

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from data import make_dataset

raw_path = hydra.utils.to_absolute_path('data/raw')
processed_path = hydra.utils.to_absolute_path('data/processed')


def test_custom_dataset():
    images_file = f"{raw_path}/train_images_0.pt"
    target_file = f"{raw_path}/train_target_0.pt"

    dataset = make_dataset.CustomDataset(images_file, target_file)

    image, target = dataset[0]
    assert isinstance(image, torch.Tensor), f"image was expected to be a torch tensor, not {type(image)}."
    assert isinstance(target, torch.Tensor), f"target was expected to be a torch tensor, not {type(target)}."


def test_mnist_dataloaders():
    
    train_loader, validation_loader, test_loader = make_dataset.mnist(32, 10000, 5, 1)
    assert isinstance(train_loader, DataLoader), f"train_loader was expected to be a DataLoader, not {type(train_loader)}."
    assert isinstance(test_loader, DataLoader), f"test_loader was expected to be a DataLoader, not {type(test_loader)}."
    assert isinstance(validation_loader, DataLoader), f"validation_loader was expected to be a DataLoader, not {type(validation_loader)}."
    
    _, validation_loader, _ = make_dataset.mnist(32, 10000, 6, 0)
    assert validation_loader is None, f"validation_loader should be empty when no data is defined for validation."
    

def test_mnist_type_handing():
    with pytest.raises(ValueError):
        make_dataset.mnist(0, 10000, 5, 1)
    with pytest.raises(ValueError):
        make_dataset.mnist(10000, 0, 5, 1)

    with pytest.raises(TypeError):
        make_dataset.mnist("32", 10000, 5, 1)
    with pytest.raises(TypeError):
        make_dataset.mnist(10000, "32", 5, 1)
    
    with pytest.raises(ValueError):
        make_dataset.mnist(0, 10000, 0, 1)
    with pytest.raises(ValueError):
        make_dataset.mnist(10000, 0, 5, -1)

    with pytest.raises(TypeError):
        make_dataset.mnist("32", 10000, "5", 1)
    with pytest.raises(TypeError):
        make_dataset.mnist(10000, "32", 5, "1")