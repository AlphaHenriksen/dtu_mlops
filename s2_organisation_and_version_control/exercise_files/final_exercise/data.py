import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, images_file, target_file):
        self.images = torch.load(images_file)
        self.targets = torch.load(target_file)
        self.transform = None  # You can add transformations if needed

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]

        if self.transform:
            image = self.transform(image)

        return image, target


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    # Create a list of train datasets for each i in range(6)
    path = "C:/Users/ander/OneDrive/Skrivebord/02476 Machine Learning Operations/dtu_mlops/data/corruptmnist/"
    train_datasets = [CustomDataset(f'{path}train_images_{i}.pt', f'{path}train_target_{i}.pt') for i in range(6)]

    # Concatenate the train datasets into one
    concatenated_train_dataset = ConcatDataset(train_datasets)

    # Create a single train loader
    train_loader = DataLoader(concatenated_train_dataset, batch_size=32, shuffle=True)

    # Create the test dataset and loader
    test_dataset = CustomDataset(f'{path}test_images.pt', f'{path}test_target.pt')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = mnist()
    
    # Example of using next(iter(train_loader))
    train_iter = iter(train_loader)
    batch_images, batch_targets = next(train_iter)
    print(f"Train Loader: Images shape: {batch_images.shape}, Targets shape: {batch_targets.shape}")

    # Example of using next(iter(test_loader))
    test_iter = iter(test_loader)
    test_images, test_targets = next(test_iter)
    print(f"Test Loader: Images shape: {test_images.shape}, Targets shape: {test_targets.shape}")
    
    
    # Loading and showing an image
    i = 0
    while True:
        img = batch_images[i]
        label = batch_targets[i]
        print(label)
        plt.imshow(img.resize_(1, 28, 28).numpy().squeeze(), cmap='gray')
        plt.show()
        i += 1