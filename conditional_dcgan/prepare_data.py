import os
from PIL import Image
import numpy as np
import torch
import config
from sklearn.model_selection import train_test_split

def save_cropped_images(image_dir, crop_size, stride, image_files, crop_images):
    """
    Crop full-resolution microstructure images using a sliding window 
    and save the patches for training.

    Args:
        image_dir (str): Path to the directory containing original images.
        crop_size (int): Size of square crop patches.
        stride (int): Step size for sliding window.
        image_files (list): List of image filenames (e.g., .tif).
        crop_images (str): Directory where cropped patches will be saved.
    """
    save_root = crop_images
    os.makedirs(save_root, exist_ok=True)

    for idx, file in enumerate(image_files):
        label_value = 500 + idx * 50  # Assign a label (e.g., 500, 550, ...) based on index
        img_path = os.path.join(image_dir, file)
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img = np.array(img, dtype=np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255  # Normalize to 0–255

        out_folder = os.path.join(save_root, str(label_value))
        os.makedirs(out_folder, exist_ok=True)

        count = 0
        # Crop image using sliding window
        for i in range(0, img.shape[1], stride):
            for j in range(0, img.shape[0], stride):
                if i + crop_size <= img.shape[1] and j + crop_size <= img.shape[0]:
                    crop = img[j:j+crop_size, i:i+crop_size]
                    out_path = os.path.join(out_folder, f"{count}.png")
                    Image.fromarray(crop.astype(np.uint8)).save(out_path)
                    count += 1
    print(f"Cropped images saved to {save_root}/(label)/")


def load_images(crop_images):
    """
    Load cropped grayscale images and generate corresponding integer labels.

    Args:
        crop_images (str): Directory containing cropped images.

    Returns:
        images (Tensor): Tensor of shape [N, 1, H, W]
        labels (Tensor): Tensor of shape [N], containing integer labels.
    """
    root = crop_images
    images = []
    labels = []

    folder_names = sorted([f for f in os.listdir(root) if f.isdigit()], key=int)
    label_map = {int(name): idx for idx, name in enumerate(folder_names)}  # e.g., 500 → 0, 550 → 1

    for folder_name in folder_names:
        label_value = int(folder_name)
        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for fname in sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0])):
            img_path = os.path.join(folder_path, fname)
            img = Image.open(img_path).convert('L')
            img = np.array(img, dtype=np.float32) / 255.0  # Normalize to 0–1
            images.append(img)
            labels.append(label_map[label_value])  # Map actual label to class index

    images = np.array(images, dtype=np.float32)
    images = torch.tensor(images).unsqueeze(1)  # Add channel dimension → [N, 1, H, W]
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def save_crop_image_fid(image_dir, crop_size, stride2, image_files, crop_images_fid):
    """
    Generate cropped image patches for FID evaluation using a different stride.

    Args:
        image_dir (str): Directory of original images.
        crop_size (int): Size of each crop patch.
        stride2 (int): Stride for FID patch cropping.
        image_files (list): List of original image filenames.
        crop_images_fid (str): Directory to save FID crop patches.
    """
    save_root = crop_images_fid
    os.makedirs(save_root, exist_ok=True)

    for idx, file in enumerate(image_files):
        label_value = 500 + idx * 50
        img_path = os.path.join(image_dir, file)
        img = Image.open(img_path).convert('L')
        img = np.array(img, dtype=np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255

        out_folder = os.path.join(save_root, str(label_value))
        os.makedirs(out_folder, exist_ok=True)

        count = 0
        for i in range(0, img.shape[1], stride2):
            for j in range(0, img.shape[0], stride2):
                if i + crop_size <= img.shape[1] and j + crop_size <= img.shape[0]:
                    crop = img[j:j+crop_size, i:i+crop_size]
                    out_path = os.path.join(out_folder, f"{count}.png")
                    Image.fromarray(crop.astype(np.uint8)).save(out_path)
                    count += 1
    print(f"Cropped images saved to {save_root}/(label)/")


def prepare_main(image_dir, crop_size, stride, stride2, image_files, crop_images, crop_images_fid):
    """
    Run the complete preprocessing pipeline:
    1. Crop training patches from original images.
    2. Load cropped images and assign labels.
    3. Generate FID evaluation patches with different stride.

    Args:
        image_dir (str): Path to original image directory.
        crop_size (int): Crop patch size.
        stride (int): Stride for training patches.
        stride2 (int): Stride for FID patches.
        image_files (list): List of input image filenames.
        crop_images (str): Directory to store cropped training patches.
        crop_images_fid (str): Directory to store FID patches.
    """
    save_cropped_images(image_dir, crop_size, stride, image_files, crop_images)
    images, labels = load_images(crop_images)
    save_crop_image_fid(image_dir, crop_size, stride2, image_files, crop_images_fid)


if __name__ == "__main__":
    prepare_main()
