import os
import torch
import numpy as np
from PIL import Image
import config

def save_cropped_images():
    """
    Loads grayscale images, normalizes them, and saves non-overlapping or overlapping crops
    into folders named by label values (e.g., 500, 550, ...).

    Crop size and stride are defined in the config. Each cropped patch is saved as a PNG image.
    """
    save_root = config.crop_images
    os.makedirs(save_root, exist_ok=True)

    for idx, file in enumerate(config.image_files):
        label_value = 500 + idx * 50
        img_path = os.path.join(config.image_dir, file)
        img = Image.open(img_path).convert('L')
        img = np.array(img, dtype=np.float32)

        # Normalize to [0, 255] range
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255

        out_folder = os.path.join(save_root, str(label_value))
        os.makedirs(out_folder, exist_ok=True)

        count = 0  # Crop index
        # Slide window across image using defined stride
        for i in range(0, img.shape[1], config.stride):
            for j in range(0, img.shape[0], config.stride):
                # Ensure crop stays within image boundaries
                if i + config.crop_size <= img.shape[1] and j + config.crop_size <= img.shape[0]:
                    crop = img[j:j+config.crop_size, i:i+config.crop_size]
                    out_path = os.path.join(out_folder, f"{count}.png")
                    Image.fromarray(crop.astype(np.uint8)).save(out_path)
                    count += 1

    print(f"Cropped images saved to {save_root}/(label)/")


def load_cropped_images():
    """
    Loads all cropped image patches from subdirectories named with integer labels.

    Returns:
        torch.Tensor: A 4D tensor of shape (N, 1, H, W) where N is number of crops.
    """
    root = config.crop_images 
    cropped_images = []

    for label_name in sorted(os.listdir(root), key=lambda x: int(x) if x.isdigit() else 999999):
        folder = os.path.join(root, label_name)
        if not os.path.isdir(folder) or not label_name.isdigit():
            continue

        file_list = sorted([f for f in os.listdir(folder) if f.endswith('.png')],
                           key=lambda x: int(os.path.splitext(x)[0]))  # Sort numerically
        for fname in file_list:
            img_path = os.path.join(folder, fname)
            img = Image.open(img_path).convert('L')
            img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
            cropped_images.append(img)

    cropped_images = np.array(cropped_images, dtype=np.float32)
    return torch.tensor(cropped_images).unsqueeze(1)


def load_and_augment_labels():
    """
    Loads label values from a CSV file and aligns them with the number of cropped images.

    For each label group (e.g., folder 500, 550, etc.), generates random integers
    between min/max of its associated label group to simulate variability.

    Returns:
        tuple:
            - normalized (torch.Tensor): Normalized labels
            - mean (torch.Tensor): Mean of training labels
            - std (torch.Tensor): Std of training labels
    """
    label_file = config.label_file        
    img_root = config.crop_images

    # Identify folders with numeric names and count their images
    folder_list = sorted([f for f in os.listdir(img_root) if f.isdigit()], key=int)
    folder_img_counts = {
        int(f): len([x for x in os.listdir(os.path.join(img_root, f)) if x.endswith('.png')])
        for f in folder_list
    }

    # Read label CSV file
    with open(label_file, 'r') as file:
        lines = file.readlines()
    labels = np.array([list(map(int, line.strip().split(','))) for line in lines])

    label_tensor = []
    for i, folder_label in enumerate(folder_img_counts.keys()):
        # Group label values by group_size
        start_idx = i * config.group_size    
        end_idx = (i + 1) * config.group_size
        group = labels[start_idx:end_idx, config.target_col]

        num_img = folder_img_counts[folder_label]
        # Generate one random label per image from min-max of group
        for _ in range(num_img):
            label_tensor.extend(np.random.randint(np.min(group), np.max(group), size=1))

    label_tensor = torch.tensor(label_tensor, dtype=torch.float32).reshape(-1, 1)

    # Normalize using training split statistics
    train_split = int(0.8 * len(label_tensor))
    mean = label_tensor[:train_split].mean()
    std = label_tensor[:train_split].std()
    normalized = (label_tensor - mean) / std

    return normalized, mean, std


def run_prepare_data(target):
    """
    Wrapper function to execute the full data preparation pipeline.

    Args:
        target (int): Index of the column to use from the label CSV.
    """
    config.target_col = target
    save_cropped_images()
    images = load_cropped_images()
    labels, train_mean, train_std = load_and_augment_labels()

    # Print diagnostic info
    print(f"Cropped Images Shape: {images.shape}")       
    print(f"Augmented Labels Shape: {labels.shape}")      
    print(f"Mean: {train_mean.item():.2f}, Std: {train_std.item():.2f}")


if __name__ == "__main__":
    run_prepare_data(0)  # Run data prep for target column 0
