import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import numpy as np


raw_path = "data/raw/"
processed_path = "data/processed/"


class CustomDataset(Dataset):
    def __init__(self, images_file, target_file, transform=None):
        self.images = torch.load(images_file)
        self.targets = torch.load(target_file)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        target = self.targets[idx]

        image = image.unsqueeze(0)

        if self.transform:
            image = self.transform(image)

        return image, target


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    # Create a list of train datasets for each i in range(6)
    # Define a transform to normalize the data
    # transform = transforms.Compose([transforms.ToTensor(),
    #                           transforms.Normalize(0, 1),
    #                           ])
    transform = transforms.Compose([transforms.Normalize((0,), (1,))])

    train_datasets = [
        CustomDataset(f"{raw_path}train_images_{i}.pt", f"{raw_path}train_target_{i}.pt", transform=transform)
        for i in range(6)
    ]

    # Concatenate the train datasets into one
    concatenated_train_dataset = ConcatDataset(train_datasets)

    # Create a single train loader
    train_loader = DataLoader(concatenated_train_dataset, batch_size=32, shuffle=True)

    # Create the test dataset and loader
    test_dataset = CustomDataset(f"{raw_path}test_images.pt", f"{raw_path}test_target.pt", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=10000, shuffle=False)

    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, test_loader = mnist()

    train_images, train_labels = next(iter(train_loader))
    # torch.save(train_images, f"{processed_path}train_images.pt")
    # torch.save(train_labels, f"{processed_path}train_labels.pt")

    test_images, test_labels = next(iter(test_loader))
    # # Save as torch tensors
    # torch.save(test_images, f"{processed_path}test_images.pt")
    # torch.save(test_labels, f"{processed_path}test_labels.pt")
    
    # # Save the first 10 images from test_images to example_images.npy
    # np.save(f"{processed_path}example_images.npy", test_images[:10])
