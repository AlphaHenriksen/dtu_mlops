import torch
import torch
from models.model import MyAwesomeModel
import matplotlib.pyplot as plt

from data.make_dataset import mnist


model_path = "models/"
visualization_path = "reports/figures/"


def evaluate(model_checkpoint, test_set):
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
    
    with torch.no_grad():
        running_loss = 0
        correct = 0
        total = 0
        
        for images, targets in test_set:
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
        
    return outputs


def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    
    outputs = evaluate(model, dataloader)
    
    return outputs


if __name__ == "__main__":
    
    _, test_set = mnist()
    predict(f'{model_path}checkpoint.pth', test_set)