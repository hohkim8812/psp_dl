import os
from PIL import Image
import numpy as np
import torch
import config
from sklearn.model_selection import train_test_split


def save_cropped_images(image_dir, crop_size, stride, image_files, crop_images):
    """
    Crop grayscale images into fixed-size patches and save them by label.

    Args:
        image_dir (str): Path to directory containing original images.
        crop_size (int): Size of square crop (e.g., 128).
        stride (int): Sliding window stride for cropping.
        image_files (list): List of filenames to process.
        crop_images (str): Root directory where cropped images will be saved.
    """
    save_root = crop_images
    os.makedirs(save_root, exist_ok=True)

    for idx, file in enumerate(image_files):
        label_value = 500 + idx * 50  # Assign synthetic label from image index
        img_path = os.path.join(image_dir, file)

        # Load image and normalize to [0, 255]
        img = Image.open(img_path).convert('L')
        img = np.array(img, dtype=np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255

        out_folder = os.path.join(save_root, str(label_value))
        os.makedirs(out_folder, exist_ok=True)

        count = 0
        # Slide over image using defined stride
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
    Load cropped grayscale image patches and assign numeric labels.

    Args:
        crop_images (str): Root directory containing label-named subfolders.

    Returns:
        tuple:
            - images (Tensor): 4D tensor of shape (N, 1, H, W)
            - labels (Tensor): 1D tensor of integer class labels
    """
    root = crop_images
    images = []
    labels = []

    # Folder names must be integers (e.g., 500, 550, ...)
    folder_names = sorted([f for f in os.listdir(root) if f.isdigit()], key=int)
    label_map = {int(name): idx for idx, name in enumerate(folder_names)}  # Map to index labels

    for folder_name in folder_names:
        label_value = int(folder_name)
        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        # Load and normalize each cropped image
        for fname in sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0])):
            img_path = os.path.join(folder_path, fname)
            img = Image.open(img_path).convert('L')
            img = np.array(img, dtype=np.float32) / 255.0
            images.append(img)
            labels.append(label_map[label_value])

    # Convert to PyTorch tensors
    images = np.array(images, dtype=np.float32)
    images = torch.tensor(images).unsqueeze(1)  # Add channel dim (N, 1, H, W)
    labels = torch.tensor(labels, dtype=torch.long)
    return images, labels


def save_crop_image_fid(image_dir, crop_size, stride2, image_files, crop_images_fid):
    """
    Crop grayscale images with a coarser stride (e.g., for FID evaluation) and save.

    Args:
        image_dir (str): Path to original high-res images.
        crop_size (int): Size of crop patch.
        stride2 (int): Coarse stride (typically > stride) for less overlap.
        image_files (list): List of image filenames.
        crop_images_fid (str): Directory to save FID-compatible crops.
    """
    save_root = crop_images_fid
    os.makedirs(save_root, exist_ok=True)

    for idx, file in enumerate(image_files):
        label_value = 500 + idx * 50  # Synthetic label
        img_path = os.path.join(image_dir, file)

        # Load and normalize image
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
    Main pipeline to prepare datasets for training and FID evaluation.

    Steps:
        1. Crop dense training images.
        2. Load and label cropped data.
        3. Crop coarsely for FID reference data.

    Args:
        image_dir (str): Path to input images.
        crop_size (int): Size of each crop patch.
        stride (int): Stride for training data cropping.
        stride2 (int): Stride for FID data cropping.
        image_files (list): List of filenames to use.
        crop_images (str): Output folder for training crops.
        crop_images_fid (str): Output folder for FID crops.
    """
    save_cropped_images(image_dir, crop_size, stride, image_files, crop_images)
    images, labels = load_images(crop_images)
    save_crop_image_fid(image_dir, crop_size, stride2, image_files, crop_images_fid)


if __name__ == "__main__":
    prepare_main()
