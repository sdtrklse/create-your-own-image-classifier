import os
from typing import Dict, Tuple

import torch
from torchvision import datasets, transforms, models
from PIL import Image

os.environ["KERAS_BACKEND"] = "torch"
import keras


def load_data(
    data_dir: str,
    batch_size: int = 64
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, int]]:
    """
    Load data from a directory and apply transformations for training and validation.

    Parameters:
    data_dir (str): Directory containing "train" and "valid" subdirectories with image data.
    batch_size (int): The batch size for the data loaders.

    Returns:
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, int]]: 
    A tuple containing the training data loader, validation data loader, 
    and a dictionary mapping class names to indices.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "valid")
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(val_dir, transform=val_transforms)
    
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)
    
    return train_loader, val_loader, train_data.class_to_idx


def train_loop(
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 5,
    device: torch.device = torch.device("cpu")
) -> None:
    """
    Train a PyTorch model with a training loop.

    Parameters:
    train_loader (torch.utils.data.DataLoader): The training data loader.
    val_loader (torch.utils.data.DataLoader): The validation data loader.
    model (torch.nn.Module): The PyTorch model to train.
    criterion (torch.nn.Module): The loss function to use.
    optimizer (torch.optim.Optimizer): The optimizer to use.
    epochs (int): The number of epochs to train for (default is 5).
    device (torch.device): The device to run the training on (default is CPU).

    Returns:
    None
    """
    for epoch in range(epochs):
        train_loss = 0.0
        train_total, train_correct = 0, 0
        
        print(f"Epoch {epoch+1}/{epochs}")
        
        n_batches = len(train_loader)
        pbar = keras.utils.Progbar(target=n_batches)

        # Train step
        model.train()
        for batch, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Compute prediction and loss
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item() * images.size(0)
            
            # Compute accuracy
            _, preds = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()
            train_accuracy = 100 * (train_correct / train_total)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update the pbar with each batch
            pbar.update(batch, values=[("loss", loss.item()), ("acc", train_accuracy)])
            

        # Validation step
        val_loss = 0.0
        val_correct, val_total = 0, 0
        
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
        
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                
                # Compute accuracy
                _, preds = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

            val_loss /= val_total
            val_accuracy = 100 * (val_correct / val_total)

            # Final update belongs to the validation data
            pbar.update(n_batches, values=[("val_loss", val_loss), ("val_acc", val_accuracy)])


def save_checkpoint(
    model: torch.nn.Module,
    arch: str,
    hidden_units: int,
    optimizer: torch.optim.Optimizer,
    class_to_idx: Dict[str, int],
    epochs: int
) -> None:
    """
    Save a checkpoint of a PyTorch neural network model.

    Parameters:
    model (torch.nn.Module): The neural network model to save.
    arch (str): The model architecture.
    hidden_units (int): The number of hidden units in the classifier.
    optimizer (torch.optim.Optimizer): The optimizer to save.
    class_to_idx (Dict[str, int]): A dictionary mapping class names to indices.
    epochs (int): The number of epochs the model has been trained for.

    Returns:
    None
    """
    checkpoint = {
        "arch": arch,
        "hidden_units": hidden_units,
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "class_to_idx": class_to_idx,
        "epochs": epochs
    }

    torch.save(checkpoint, "checkpoint.pth")


def load_checkpoint(filepath: str) -> torch.nn.Module:
    """
    Load a model checkpoint from a file.

    Parameters:
    filepath (str): The path to the checkpoint file.

    Returns:
    torch.nn.Module: The loaded model.
    """
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"), weights_only=True)
    model = getattr(models, checkpoint["arch"])(weights="VGG16_Weights.DEFAULT")

    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[0].in_features, checkpoint["hidden_units"]),
        torch.nn.ReLU(),
        torch.nn.Linear(checkpoint["hidden_units"], 102),
        torch.nn.LogSoftmax(dim=1)
    )

    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    return model


def process_image(image_path: str) -> torch.Tensor:
    """
    Process an image by resizing and cropping it to the specified size, 
    then normalizing the pixel values.

    Parameters:
    image_path (str): The path to the image file.

    Returns:
    torch.Tensor: The processed image as a tensor.
    """
    image = Image.open(image_path)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return transform(image)
