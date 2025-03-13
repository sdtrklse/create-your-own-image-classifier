import argparse
from utils import train_loop, load_data, save_checkpoint

import torch
from torch import nn, optim
from torchvision import models


def train_model(
    data_dir: str,
    arch: str = "vgg16",
    hidden_units: int = 512,
    learning_rate: float = 0.001,
    epochs: int = 5,
    gpu: bool = False
) -> None:
    """
    Train a neural network model on a dataset.

    Parameters:
    data_dir (str): Directory containing the dataset.
    arch (str): Model architecture to use (default is 'vgg16').
    learning_rate (float): Learning rate for the optimizer (default is 0.001).
    hidden_units (int): Number of hidden units in the classifier (default is 512).
    epochs (int): Number of epochs to train for (default is 5).
    gpu (bool): Whether to use GPU for training (default is False).

    Returns:
    None
    """
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    # Load data and model
    train_loader, val_loader, class_to_idx = load_data(data_dir)
    model = getattr(models, arch)(weights="VGG16_Weights.DEFAULT")

    # Freeze model parameters to avoid backpropagation
    for param in model.parameters():
        param.requires_grad = False
    
    # Define a new classifier for the model
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, hidden_units),
        nn.ReLU(),
        nn.Linear(hidden_units, 102),
        nn.LogSoftmax(dim=1)
    )
    
    # Move the model to the selected device
    model.to(device)
    
    # Define the loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(
        model.classifier.parameters(),
        lr=learning_rate
    )
    
    # Train the model using the training loop
    train_loop(train_loader, val_loader, model, criterion, optimizer, epochs, device)
    
    # Save the model checkpoint
    save_checkpoint(model, arch, hidden_units, optimizer, class_to_idx, epochs)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network model on a dataset")
    parser.add_argument("data_dir", help="Dataset directory")
    parser.add_argument("--arch", default="vgg16", help="Model architecture")
    parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")
    
    args = parser.parse_args()
    train_model(args.data_dir, args.arch, args.hidden_units, args.learning_rate, args.epochs, args.gpu)
