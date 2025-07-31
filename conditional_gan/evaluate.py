import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import config
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg


def set_seed(seed=None):
    """
    Set random seed for NumPy and PyTorch (CPU only).

    Args:
        seed (int, optional): Seed value to use for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)


def resize_and_save_images(src_dir, tgt_dir, size, n_per_label):
    """
    Resize and save a fixed number of PNG images per class into a target directory.

    Args:
        src_dir (str): Source directory containing class-labeled folders of PNGs.
        tgt_dir (str): Target directory to save resized images.
        size (tuple): (width, height) for resizing.
        n_per_label (int): Number of images to randomly sample per class.
    """
    os.makedirs(tgt_dir, exist_ok=True)

    for folder in sorted(os.listdir(src_dir), key=lambda x: int(x)):
        src_folder = os.path.join(src_dir, folder)
        tgt_folder = os.path.join(tgt_dir, folder)
        os.makedirs(tgt_folder, exist_ok=True)

        # List and randomly sample images
        img_files = sorted([f for f in os.listdir(src_folder) if f.endswith('.png')])
        selected = np.random.choice(img_files, min(n_per_label, len(img_files)), replace=False)

        for img_file in selected:
            img_path = os.path.join(src_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                # Resize using high-quality interpolation
                img = cv2.resize(img, size, interpolation=cv2.INTER_LANCZOS4)
                save_path = os.path.join(tgt_folder, img_file)
                cv2.imwrite(save_path, img)


def load_images(path, n_per_label):
    """
    Load a fixed number of resized PNG images from each class folder.

    Args:
        path (str): Path to root directory with class subfolders.
        n_per_label (int): Number of images to load per class.

    Returns:
        np.ndarray: Array of shape (N, 3, H, W) with pixel values in [0, 1].
    """
    images = []
    for folder_name in sorted(os.listdir(path), key=lambda x: int(x)):
        folder_path = os.path.join(path, folder_name)
        img_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        selected = np.random.choice(img_files, min(n_per_label, len(img_files)), replace=False)

        for img_file in selected:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is not None:
                img = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
                img = img.transpose(2, 0, 1)  # Convert HWC to CHW
                images.append(img)

    images = np.array(images)
    return images


class PartialInceptionNetwork(nn.Module):
    """
    Partial InceptionV3 network that extracts features from the Mixed_7c layer.

    The output features are pooled to (1, 1) and flattened to form a feature vector.
    Used for computing FID.
    """
    def __init__(self):
        super().__init__()
        # Load pretrained InceptionV3 weights and hook into Mixed_7c
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=True)
        self.inception.Mixed_7c.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self._features = output

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        self.inception(x)
        feats = self._features
        feats = nn.functional.adaptive_avg_pool2d(feats, (1,1)).view(x.shape[0], -1)
        return feats


def get_activations(images, batch_size=32):
    """
    Extract Inception features for a batch of images.

    Args:
        images (np.ndarray): Array of shape (N, 3, H, W).
        batch_size (int): Number of images per batch.

    Returns:
        np.ndarray: Inception feature vectors of shape (N, D).
    """
    model = PartialInceptionNetwork().eval()
    if torch.cuda.is_available():
        model.cuda()

    n = images.shape[0]
    feats = []

    for i in range(0, n, batch_size):
        batch = torch.from_numpy(images[i:i+batch_size]).float()
        if torch.cuda.is_available():
            batch = batch.cuda()
        with torch.no_grad():
            feat = model(batch).cpu().numpy()
        feats.append(feat)

    return np.concatenate(feats, axis=0)


def calculate_activation_statistics(images, batch_size=32):
    """
    Compute mean and covariance of Inception features.

    Args:
        images (np.ndarray): Preprocessed image array.
        batch_size (int): Batch size for feature extraction.

    Returns:
        tuple: (mu, sigma)
            - mu (np.ndarray): Mean vector of features.
            - sigma (np.ndarray): Covariance matrix of features.
    """
    act = get_activations(images, batch_size)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Compute the Frechet Distance between two Gaussians (FID metric).

    Args:
        mu1 (np.ndarray): Mean of real image features.
        sigma1 (np.ndarray): Covariance of real image features.
        mu2 (np.ndarray): Mean of generated image features.
        sigma2 (np.ndarray): Covariance of generated image features.
        eps (float): Small constant to improve numerical stability.

    Returns:
        float: FID score.
    """
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Handle numerical instability
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # If result has small imaginary component, discard it
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # Return FID formula
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)


def calculate_fid(images1, images2, batch_size=32):
    """
    Calculate FID score between two image datasets.

    Args:
        images1 (np.ndarray): Real images.
        images2 (np.ndarray): Generated images.
        batch_size (int): Batch size for feature extraction.

    Returns:
        float: FID score.
    """
    mu1, sigma1 = calculate_activation_statistics(images1, batch_size)
    mu2, sigma2 = calculate_activation_statistics(images2, batch_size)
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)


def main(
    real_src_dir=config.real_src_dir,
    real_dir=config.real_dir,
    gen_src_dir=config.gen_src_dir,
    gen_dir=config.gen_dir,
    img_size=config.img_size,
    batch_size=config.batch_size,
    n_per_label=config.n_per_label,
    seed=config.seed
):
    """
    Main function to compute FID between real and generated image sets.

    Workflow:
        1. Resize images from original folders into FID-compatible resolution.
        2. Load a fixed number of images per label.
        3. Compute FID score and print the result.
    """
    set_seed(seed)

    print("Resizing images...")
    resize_and_save_images(real_src_dir, real_dir, img_size, n_per_label)
    resize_and_save_images(gen_src_dir, gen_dir, img_size, n_per_label)

    print("Loading resized images...")
    real_imgs = load_images(real_dir, n_per_label)
    gen_imgs = load_images(gen_dir, n_per_label)

    print("Shape check (should be [N, 3, 299, 299]):", real_imgs.shape)

    fid = calculate_fid(real_imgs, gen_imgs, batch_size=batch_size)
    print(f"\n FID Score: {fid:.2f}")


if __name__ == "__main__":
    main()
