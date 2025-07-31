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
    Set random seed for reproducibility.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

def resize_and_save_grayscale_images(src_dir, tgt_dir, size, n_per_label):
    """
    Resize grayscale images and save to target directory.

    Args:
        src_dir (str): Source image directory grouped by label.
        tgt_dir (str): Output directory for resized images.
        size (tuple): Target size (width, height).
        n_per_label (int): Number of images to sample per label.
    """
    os.makedirs(tgt_dir, exist_ok=True)
    for folder in sorted(os.listdir(src_dir), key=lambda x: int(x)):
        src_folder = os.path.join(src_dir, folder)
        tgt_folder = os.path.join(tgt_dir, folder)
        os.makedirs(tgt_folder, exist_ok=True)

        img_files = sorted([f for f in os.listdir(src_folder) if f.endswith('.png')])
        selected = np.random.choice(img_files, min(n_per_label, len(img_files)), replace=False)

        for img_file in selected:
            img_path = os.path.join(src_folder, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, size)
                save_path = os.path.join(tgt_folder, img_file)
                cv2.imwrite(save_path, img)

def load_images(path, n_per_label):
    """
    Load grayscale images from directory and normalize them to [-1, 1].

    Args:
        path (str): Root directory containing labeled subfolders.
        n_per_label (int): Number of images to load per label.

    Returns:
        np.ndarray: Array of shape [N, 1, H, W]
    """
    images = []
    for folder_name in sorted(os.listdir(path), key=lambda x: int(x)):
        folder_path = os.path.join(path, folder_name)
        img_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        selected = np.random.choice(img_files, min(n_per_label, len(img_files)), replace=False)

        for img_file in selected:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Normalize to [-1, 1] for FID compatibility
                img = img.astype(np.float32) / 255.0 * 2.0 - 1.0
                images.append(img)

    images = np.array(images)[:, np.newaxis, :, :]  # Shape: [N, 1, H, W]
    return images

class PartialInceptionNetwork(nn.Module):
    """
    InceptionV3-based network to extract intermediate feature activations
    for FID computation.
    """
    def __init__(self):
        super().__init__()
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=True)
        self.inception.Mixed_7c.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self._features = output

    def forward(self, x):
        # Convert grayscale to 3-channel RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        self.inception(x)  # Only need to forward once; features captured in hook
        feats = self._features
        feats = nn.functional.adaptive_avg_pool2d(feats, (1, 1)).view(x.shape[0], -1)
        return feats

def get_activations(images, batch_size=32):
    """
    Extract activations from PartialInceptionNetwork.

    Args:
        images (np.ndarray): Normalized image tensor [N, C, H, W]
        batch_size (int): Batch size for inference

    Returns:
        np.ndarray: Activation vectors [N, D]
    """
    model = PartialInceptionNetwork().eval()
    if torch.cuda.is_available():
        model.cuda()

    n = images.shape[0]
    feats = []

    for i in range(0, n, batch_size):
        batch = torch.from_numpy(images[i:i + batch_size]).float()
        if torch.cuda.is_available():
            batch = batch.cuda()
        with torch.no_grad():
            feat = model(batch).cpu().numpy()
        feats.append(feat)

    return np.concatenate(feats, axis=0)

def calculate_activation_statistics(images, batch_size=32):
    """
    Calculate mean and covariance of inception features.

    Args:
        images (np.ndarray): Input image tensor [N, C, H, W]

    Returns:
        tuple: (mean vector, covariance matrix)
    """
    act = get_activations(images, batch_size)
    mu, sigma = np.mean(act, axis=0), np.cov(act, rowvar=False)
    return mu, sigma

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """
    Calculate Frechet Distance (FID) between two distributions.

    Args:
        mu1, mu2 (np.ndarray): Mean vectors
        sigma1, sigma2 (np.ndarray): Covariance matrices
        eps (float): Small offset for numerical stability

    Returns:
        float: FID score
    """
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)

    # Handle non-finite or complex matrix case
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

def calculate_fid(images1, images2, batch_size=32):
    """
    Full FID computation between two image sets.

    Args:
        images1 (np.ndarray): Real or reference images
        images2 (np.ndarray): Generated images

    Returns:
        float: FID score
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
    Main function for calculating FID between real and generated grayscale images.
    """
    set_seed(seed)

    print("Resizing images...")
    resize_and_save_grayscale_images(real_src_dir, real_dir, img_size, n_per_label)
    resize_and_save_grayscale_images(gen_src_dir, gen_dir, img_size, n_per_label)

    print("Loading resized images...")
    real_imgs = load_images(real_dir, n_per_label)
    gen_imgs = load_images(gen_dir, n_per_label)

    print("Shape check (should be [N, 1, 299, 299]):", real_imgs.shape)

    fid = calculate_fid(real_imgs, gen_imgs, batch_size=batch_size)
    print(f"\n FID Score: {fid:.2f}")

if __name__ == "__main__":
    main()
