import json
import argparse
from typing import List, Tuple

import torch

from utils import load_checkpoint, process_image


def predict(
    image_path: str,
    checkpoint: str,
    topk: int = 5,
    category_names: str = None,
    gpu: bool = False
) -> Tuple[List[float], List[str]]:
    """
    Predict the class (or classes) of an image using a trained deep learning model.
    
    Args:
        image_path (str): Path to the image file.
        checkpoint (str): Path to the model checkpoint file.
        topk (int, optional): Number of top predictions to return. Default is 5.
        category_names (str, optional): Path to a JSON file mapping category labels to names. Default is None.
        gpu (bool, optional): Use GPU for inference. Default is False.
    
    Returns:
        Tuple[List[float], List[str]]: A tuple containing a list of probabilities and a list of class labels.
    """
    # Set the device to GPU if available and requested
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    # Load the model from the checkpoint and set it to evaluation mode
    model = load_checkpoint(checkpoint)
    model.to(device)
    model.eval()

    # Preprocess the image
    image = process_image(image_path).to(device)

    # Make predictions with the model
    with torch.no_grad():
        output = model(image)
    
    # Compute the probabilities of each class
    probs = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get the top k probabilities and their indices
    top_probs, top_indices = torch.topk(probs, topk)

    # Convert the probabilities and indices to numpy arrays
    top_probs = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()
    
    # Convert the indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in top_indices]

    # If category names are provided, map the class labels to names
    if category_names:
        with open(category_names, "r") as file:
            cat_to_name = json.load(file)
        classes = [cat_to_name.get(c, c) for c in classes]
    
    return top_probs.tolist(), classes


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
