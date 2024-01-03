import torch
import os
import sys
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models.model import MyAwesomeModel
from data.make_dataset import mnist


model_path = "models/"
visualization_path = "reports/figures/"


def train(lr):
    """
    Using the corrupted MNIST dataset, train a cnn model.

    Parameters:
        lr (float): learning rate for the optimizer.

    Returns:
        None.
    """
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel()
    train_set, _ = mnist()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    accuracies = []
    losses = []

    # Training loop
    for epoch in range(2):
        running_loss = 0
        correct = 0
        total = 0

        for images, targets in train_set:
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)

            # Backward pass
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Accuracy calculation
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        # Get metrics
        accuracy = 100 * correct / total
        loss = running_loss / len(train_set)
        accuracies.append(accuracy)
        losses.append(loss)

        print(f"Epoch {epoch+1} of 10")
        print(f"Training loss: {loss}")
        print(f"Accuracy: {accuracy}\n")

    # Do plotting
    plt.plot(losses, label="loss")
    plt.plot(accuracies, label="accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.savefig(f"{visualization_path}viz.png")
    plt.show()

    print("Saving final model.")
    torch.save(model.state_dict(), f"{model_path}checkpoint.pth")


if __name__ == "__main__":
    train(1e-3)