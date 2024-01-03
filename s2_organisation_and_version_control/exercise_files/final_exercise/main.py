import click
import torch
from model import MyAwesomeModel
from tqdm import tqdm

from data import mnist


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set, _ = mnist()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(10):
        # Get accuracy
        running_loss = 0
        correct = 0
        total = 0
        
        for images, targets in train_set:
            images = images[:, None, ...]
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
        print(f"Epoch {epoch+1} of 10")
        print(f"Training loss: {running_loss/len(train_set)}")
        print(f"Accuracy: {accuracy}\n")
        
    
    print("Saving final model.")
    torch.save(model.state_dict(), 'checkpoint.pth')


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    """Evaluate a trained model."""
    print("Evaluating like my life dependends on it")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    # model = torch.load(model_checkpoint)
    model = MyAwesomeModel()
    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    _, test_set = mnist()
    
    with torch.no_grad():
        running_loss = 0
        correct = 0
        total = 0
        
        for images, targets in test_set:
            images = images[:, None, ...]
            outputs = model(images)
            loss = criterion(outputs, targets)
        
            running_loss += loss.item()    
            # Accuracy
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        # Accuracy
        accuracy = 100 * correct / total
        print(f"Training loss: {running_loss/len(test_set)}")
        print(f"Accuracy: {accuracy}\n")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
