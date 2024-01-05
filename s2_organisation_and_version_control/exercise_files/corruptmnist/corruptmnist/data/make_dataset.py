import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import numpy as np
import hydra
import matplotlib.pyplot as plt


raw_path = hydra.utils.to_absolute_path('data/raw')
processed_path = hydra.utils.to_absolute_path('data/processed')


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


def mnist(train_batch_size, test_batch_size):
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    # Create a list of train datasets for each i in range(6)
    # Define a transform to normalize the data
    # transform = transforms.Compose([transforms.ToTensor(),
    #                           transforms.Normalize(0, 1),
    #                           ])
    transform = transforms.Compose([transforms.Normalize((0,), (1,))])

    total_files = 6
    train_files = 5
    validation_files = total_files - train_files
    
    train_datasets = [
        CustomDataset(f"{raw_path}/train_images_{i}.pt", f"{raw_path}/train_target_{i}.pt", transform=transform)
        for i in range(train_files)
    ]
    # Concatenate the train datasets into one
    concatenated_train_dataset = ConcatDataset(train_datasets)
    # Create a single train loader
    train_loader = DataLoader(concatenated_train_dataset, batch_size=train_batch_size, shuffle=True)  # , num_workers=8

    validation_loader = None
    if validation_files > 0:
        validation_datasets = [
            CustomDataset(f"{raw_path}/train_images_{i}.pt", f"{raw_path}/train_target_{i}.pt", transform=transform)
            for i in range(train_files, total_files)
        ]
        # Concatenate the train datasets into one
        if validation_files > 1:
            concatenated_validation_dataset = ConcatDataset(validation_datasets)
            # Create a single train loader
            validation_loader = DataLoader(concatenated_validation_dataset, batch_size=train_batch_size, shuffle=True)  # , num_workers=8
        else:
            validation_loader = DataLoader(validation_datasets[0], batch_size=train_batch_size, shuffle=True)  # , num_workers=8

    # Create the test dataset and loader
    test_dataset = CustomDataset(f"{raw_path}/test_images.pt", f"{raw_path}/test_target.pt", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)  # ,  num_workers=8

    return train_loader, validation_loader, test_loader


if __name__ == "__main__":
    train_loader, validation_loader, test_loader = mnist(32, 10000)

    train_images, train_labels = next(iter(train_loader))
    # torch.save(train_images, f"{processed_path}/train_images.pt")
    # torch.save(train_labels, f"{processed_path}/train_labels.pt")

    print(validation_loader)
    validation_images, validation_labels = next(iter(validation_loader))
    # Show the image using plt and cmap="gray"
    plt.imshow(validation_images[0].squeeze(), cmap="gray")
    plt.show()
    
    test_images, test_labels = next(iter(test_loader))
    # # Save as torch tensors
    # torch.save(test_images, f"{processed_path}/test_images.pt")
    # torch.save(test_labels, f"{processed_path}/test_labels.pt")
    
    # # Save the first 10 images from test_images to example_images.npy
    # np.save(f"{processed_path}/example_images.npy", test_images[:10])
