import os
import torch
import numpy as np
import pandas as pd
import re
from PIL import Image
from model import ResNet_Model
import config

# Set computation device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# For natural filename sorting
def numerical_sort_key(s):
    """
    Generate a sort key that allows for natural sorting of strings with numbers.

    Args:
        s (str): Input string, typically a filename.

    Returns:
        list: A list containing integers and lowercase strings for natural sorting.
    """
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]


# Image preprocessing
def preprocess_image(img_path):
    """
    Preprocesses a grayscale image for model inference.

    Steps:
        - Opens the image and converts to grayscale
        - Normalizes pixel values to [0, 1]
        - Converts to a PyTorch tensor and adds batch and channel dimensions
        - Moves tensor to the appropriate device (CPU or GPU)

    Args:
        img_path (str): Path to the image file.

    Returns:
        torch.Tensor: Preprocessed image tensor of shape (1, 1, H, W).
    """
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    img = np.array(img, dtype=np.float32)  # Convert to float32 array
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # Normalize to [0,1]
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dims


# Main inference function
def run_inference():
    """
    Runs inference using a trained ResNet model on a directory of images.

    Workflow:
        - Loads the trained model and its normalization statistics
        - Iterates over subfolders containing images
        - Preprocesses and predicts the target property for each image
        - Saves the results as Excel sheets (one per subfolder)

    Output:
        Saves an Excel file named 'generator_property_prediction_col{target}.xlsx'
        where `{target}` is the index of the target property to predict.
    """
    model = ResNet_Model().to(device)  # Initialize model and move to device

    # Load trained model checkpoint
    model_path = os.path.join(config.model_dir, f"model{config.target_col}.pth")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    mean = checkpoint['mean']
    std = checkpoint['std']

    print(f"Running inference for target_col = {config.target_col}")
    model.eval()  # Set model to evaluation mode

    predictions = []
    excel_data = {}
    # Loop through subfolders in the inference image directory
    for folder_name in sorted(os.listdir(config.inference_image_dir), key=numerical_sort_key):
        subfolder = os.path.join(config.inference_image_dir, folder_name)
        if not os.path.isdir(subfolder):
            continue

        folder_preds = []

        # Loop through images in the subfolder
        for fname in sorted(os.listdir(subfolder), key=numerical_sort_key):
            if not fname.endswith(".png"):
                continue 
            img_path = os.path.join(subfolder, fname)
            img_tensor = preprocess_image(img_path)

            with torch.no_grad():
                pred = model(img_tensor)  # Run model prediction
                pred = pred * std + mean  # Denormalize prediction

            predictions.append((folder_name, fname, pred.item()))
            folder_preds.append((fname, pred.item()))

        # Convert predictions to DataFrame and sort naturally by filename
        df = pd.DataFrame(folder_preds, columns=["filename", "predicted_property"])
        df = df.sort_values(by="filename", key=lambda col: col.map(numerical_sort_key))
        excel_data[folder_name] = df

    # Save all results to a multi-sheet Excel file
    target = config.target_col
    inference_excel = f"generator_property_prediction_col{target}.xlsx"
    with pd.ExcelWriter(inference_excel) as writer:
        for sheet_name, df in excel_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Inference complete. Excel saved to: {inference_excel}")


if __name__ == "__main__":
    run_inference()