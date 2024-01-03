import numpy as np
from torch import nn
import click
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from models.model import MyAwesomeModel
from data.make_dataset import mnist


model_path = "models/"
visualization_path = "reports/figures/"


@click.group()
def cli():
    """Command line interface."""
    pass


@click.command()
@click.argument("model_checkpoint")
def visualize(model_checkpoint):
    """Show a representation of the feature space of the model right before the final layer.
    
    Args:
        model_checkpoint: model to use for prediction

    Returns
        None.
    """

    # Loads a pre-trained network
    model = MyAwesomeModel()
    train_set, _ = mnist()

    state_dict = torch.load(model_checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Extracts some intermediate representation of the data (your training set) from your cnn. This could be the features 
    # just before the final classification layer
    # Extract the features from the layer before the output layer
    feature_extractor = nn.Sequential(*list(model.children())[0:2])  # Remove the last layer
    print(feature_extractor)
    all_features = []
    all_labels = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(train_set):
            print(f"Running iteration {i}.")
            features = feature_extractor(images)
            all_features.append(features.view(features.size(0), -1).numpy())
            all_labels.append(labels.numpy())
            if i > 10:
                break
            
    
    # Visualize features in a 2D space using t-SNE to do the dimensionality reduction.
    # Concatenate and flatten the features
    all_features = np.concatenate(all_features, axis=0)

    # Use t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, random_state=42)
    tsne_representation = tsne.fit_transform(all_features)

    # Visualize the t-SNE representation
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1], c=all_labels, cmap='viridis', alpha=0.5)
    plt.title('t-SNE Representation of the Layer Before Output')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.colorbar()
    plt.savefig(f'{visualization_path}tsne.png')  # Save the visualization to a file in the reports/figures/ folder.
    plt.show()


cli.add_command(visualize)


if __name__ == "__main__":
    cli()
    
    # To run:
    # Go to dir:
    # C:\Users\ander\OneDrive\Skrivebord\02476 Machine Learning Operations\dtu_mlops\s2_organisation_and_version_control\exercise_files\corruptmnist
    # Run:
    # python corruptmnist/visualizations/visualize.py visualize models/checkpoint.pth
