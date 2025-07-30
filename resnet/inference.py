import os
import torch
import numpy as np
import pandas as pd
import re
from PIL import Image
from model import ResNet_Model
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# For natural filename sorting
def numerical_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

# Image preprocessing
def preprocess_image(img_path):
    img = Image.open(img_path).convert('L')
    img = np.array(img, dtype=np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# Main inference function
def run_inference():
    model = ResNet_Model().to(device)
    model_path = os.path.join(config.model_dir, f"model{config.target_col}.pth")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    mean = checkpoint['mean']
    std = checkpoint['std']

    print(f"Running inference for target_col = {config.target_col}")
    model.eval()

    predictions = []
    excel_data = {}

    for folder_name in sorted(os.listdir(config.inference_image_dir), key=numerical_sort_key):
        subfolder = os.path.join(config.inference_image_dir, folder_name)
        if not os.path.isdir(subfolder):
            continue

        folder_preds = []
        for fname in sorted(os.listdir(subfolder), key=numerical_sort_key):
            if not fname.endswith(".png"):
                continue

            img_path = os.path.join(subfolder, fname)
            img_tensor = preprocess_image(img_path)

            with torch.no_grad():
                pred = model(img_tensor)
                pred = pred * std + mean

            predictions.append((folder_name, fname, pred.item()))
            folder_preds.append((fname, pred.item()))

        df = pd.DataFrame(folder_preds, columns=["filename", "predicted_property"])
        df = df.sort_values(by="filename", key=lambda col: col.map(numerical_sort_key))
        excel_data[folder_name] = df

    # Save results to Excel
    target = config.target_col
    inference_excel = f"generator_property_prediction_col{target}.xlsx"
    with pd.ExcelWriter(inference_excel) as writer:
        for sheet_name, df in excel_data.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Inference complete. Excel saved to: {inference_excel}")

if __name__ == "__main__":
    run_inference()