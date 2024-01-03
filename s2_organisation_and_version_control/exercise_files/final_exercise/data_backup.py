import torch
import os


def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    def __init__(self, path="data/corruptmnist"):
        self.path = path
        self.trainImages = []
        self.trainTargets = []
        for file in sorted(os.listdir(self.path)):
            fullPath = os.path.join(path, file)
            if "train_images" in file:
                self.trainImages.append(torch.load(fullPath))

            elif "train_target" in file:
                self.trainTargets.append(torch.load(fullPath))

        self.testImages = torch.load(os.path.join(path, "test_images.pt"))
        self.testTargets = torch.load(os.path.join(path, "test_target.pt"))

        self.trainImages = torch.cat(self.trainImages)
        self.trainTargets = torch.cat(self.trainTargets)

    def __len__(self):
        return len(self.trainTargets)

    def __getitem__(self, idx):
        return self.trainImages[idx], self.trainTargets[idx]