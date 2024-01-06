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


@pytest.mark.skipif(not os.path.exists(raw_path), reason="Data files not found")
def test_custom_dataset():
    images_file = f"{raw_path}/train_images_0.pt"
    target_file = f"{raw_path}/train_target_0.pt"

    dataset = make_dataset.CustomDataset(images_file, target_file)

    image, target = dataset[0]
    assert isinstance(image, torch.Tensor), f"image was expected to be a torch tensor, not {type(image)}."
    assert isinstance(target, torch.Tensor), f"target was expected to be a torch tensor, not {type(target)}."


@pytest.mark.skipif(not os.path.exists(processed_path), reason="Data files not found")
def test_mnist_dataloaders():
    
    train_loader, validation_loader, test_loader = make_dataset.mnist(32, 10000, 5, 1)
    assert isinstance(train_loader, DataLoader), f"train_loader was expected to be a DataLoader, not {type(train_loader)}."
    assert isinstance(test_loader, DataLoader), f"test_loader was expected to be a DataLoader, not {type(test_loader)}."
    assert isinstance(validation_loader, DataLoader), f"validation_loader was expected to be a DataLoader, not {type(validation_loader)}."
    
    _, validation_loader, _ = make_dataset.mnist(32, 10000, 6, 0)
    assert validation_loader is None, f"validation_loader should be empty when no data is defined for validation."
    

@pytest.mark.skipif(not os.path.exists(processed_path), reason="Data files not found")
@pytest.mark.parametrize(
    "train_batch_size, test_batch_size, num_train_files, num_validation_files", 
    [(0, 10000, 5, 1),
     (32, 0, 5, 1)]
    )
def test_mnist_batch_value_handing(train_batch_size, test_batch_size, num_train_files, num_validation_files):
    with pytest.raises(ValueError):
        make_dataset.mnist(train_batch_size, test_batch_size, num_train_files, num_validation_files)


@pytest.mark.skipif(not os.path.exists(processed_path), reason="Data files not found")
@pytest.mark.parametrize(
    "train_batch_size, test_batch_size, num_train_files, num_validation_files", 
    [("32", 10000, 5, 1),
     (32, "10000", 5, 1)]
    )
def test_mnist_batch_type_handing(train_batch_size, test_batch_size, num_train_files, num_validation_files):
    with pytest.raises(TypeError):
        make_dataset.mnist(train_batch_size, test_batch_size, num_train_files, num_validation_files)


@pytest.mark.skipif(not os.path.exists(processed_path), reason="Data files not found")
@pytest.mark.parametrize(
    "train_batch_size, test_batch_size, num_train_files, num_validation_files", 
    [(32, 10000, 0, 1),
     (32, 10000, 5, -1)]
    )
def test_mnist_numfiles_value_handing(train_batch_size, test_batch_size, num_train_files, num_validation_files):
    with pytest.raises(ValueError):
        make_dataset.mnist(train_batch_size, test_batch_size, num_train_files, num_validation_files)


@pytest.mark.skipif(not os.path.exists(processed_path), reason="Data files not found")
@pytest.mark.parametrize(
    "train_batch_size, test_batch_size, num_train_files, num_validation_files", 
    [(32, 10000, "5", 1),
     (32, 10000, 5, "1")]
    )
def test_mnist_numfiles_type_handing(train_batch_size, test_batch_size, num_train_files, num_validation_files):
    with pytest.raises(TypeError):
        make_dataset.mnist(train_batch_size, test_batch_size, num_train_files, num_validation_files)