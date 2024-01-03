import click
import torch
from models.model import MyAwesomeModel
from tqdm import tqdm
import matplotlib.pyplot as plt

from data.make_dataset import mnist


model_path = "models/"
visualization_path = "reports/figures/"


def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    accuracies = []
    losses = []

    for epoch in range(2):
        # Get accuracy
        running_loss = 0
        correct = 0
        total = 0
        
        for images, targets in train_set:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()    
            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        # Accuracy
        accuracy = 100 * correct / total
        loss = running_loss/len(train_set)
        accuracies.append(accuracy)
        losses.append(loss)
        
        print(f"Epoch {epoch+1} of 10")
        print(f"Training loss: {loss}")
        print(f"Accuracy: {accuracy}\n")
        
    plt.plot(losses, label="loss")
    plt.plot(accuracies, label="accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.savefig(f'{visualization_path}viz.png')
    plt.show()

    print("Saving final model.")
    torch.save(model.state_dict(), f'{model_path}checkpoint.pth')


if __name__ == "__main__":
    train(1e-3)
