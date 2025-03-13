import json
import argparse
from typing import Tuple
from utils import load_checkpoint, process_image

import torch


def predict(
    image_path: str,
    checkpoint: str,
    topk: int = 5,
    category_names: str = None,
    gpu: bool = False
) -> Tuple[list, list]:
    """
    Predict the class (or classes) of an image using a trained deep learning model.

    Parameters:
    image_path (str): Path to the image file.
    checkpoint (str): Path to the model checkpoint file.
    topk (int): Number of top predictions to return. Default is 5.
    category_names (str): Path to a JSON file mapping category labels to names. Default is None.
    gpu (bool): If True, use GPU for inference. Default is False.

    Returns:
    Tuple[list, list]: A tuple containing a list of probabilities and a list of class labels or names.
    """
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    # Load the model from the checkpoint and set it to evaluation mode
    model = load_checkpoint(checkpoint)
    model.to(device)
    model.eval()
    
    # Process the image into the correct format and move to specified device
    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)
    
    # Disable gradient calculation for inference
    with torch.no_grad():
        output = model(image)
        # Get the top k probabilities and indices
        probs, indices = torch.exp(output).topk(topk)
    
    # Convert probabilities and indices to numpy arrays and squeeze to remove single-dimensional entries
    probs = probs.cpu().numpy().squeeze()
    indices = indices.cpu().numpy().squeeze()
    
    # Map indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]
    
    # If category names are provided, map the class labels to names
    if category_names:
        with open(category_names, "r") as file:
            cat_to_name = json.load(file)
        classes = [cat_to_name.get(c, c) for c in classes]
    
    return probs, classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained model")
    parser.add_argument("image_path", help="Path to image")
    parser.add_argument("checkpoint", help="Model checkpoint file")
    parser.add_argument("--top_k", type=int, default=5, help="Return top k predictions")
    parser.add_argument("--category_names", help="Path to JSON mapping file")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")
    
    args = parser.parse_args()
    
    probs, classes = predict(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)
    
    print("Predictions:")
    for i in range(len(classes)):
        print(f"{probs[i]*100:05.2f}%: {classes[i]}")
