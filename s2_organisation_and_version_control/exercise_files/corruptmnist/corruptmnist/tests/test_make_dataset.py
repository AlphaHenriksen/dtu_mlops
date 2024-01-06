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


class TestMnist:

    @pytest.mark.parametrize("train_batch_size, test_batch_size, num_train_files, num_validation_files, out", [
    (0, 100000, 5, 1),
    (100000, 0, 5, 1),
    ])
    def test_train_batch_size(self, train_batch_size, test_batch_size, num_train_files, num_validation_files, out):
            with pytest.raises(TypeError):
                make_dataset.mnist(train_batch_size, test_batch_size, num_train_files, num_validation_files)

    def test_output(self, train_batch_size, test_batch_size, num_train_files, num_validation_files, out):
        train, val, test = make_dataset.mnist(train_batch_size, test_batch_size, num_train_files, num_validation_files)
            
        test_imgs, test_labels = next(iter(test))
        assert len(test_imgs) == 5000, f"There should be 5000 test images, not {len(test_imgs)}."

        if num_validation_files == 0:
            train_imgs, train_labels = next(iter(train))
            
            assert val is None, "The validation data should be None, since none of the files are picked for validation."
            
            assert len(train_imgs) == 30000, f"When all files are picked for training, there should be 30000 train images, not {len(train_imgs)}."

    # train_loader, validation_loader, test_loader = make_dataset.mnist(32, 10000, 5, 0)

    # train_images, train_labels = next(iter(train_loader))
    # # torch.save(train_images, f"{processed_path}/train_images.pt")
    # # torch.save(train_labels, f"{processed_path}/train_labels.pt")

    # print(validation_loader)
    # validation_images, validation_labels = next(iter(validation_loader))
    # # Show the image using plt and cmap="gray"
    # plt.imshow(validation_images[0].squeeze(), cmap="gray")
    # plt.show()

    # test_images, test_labels = next(iter(test_loader))