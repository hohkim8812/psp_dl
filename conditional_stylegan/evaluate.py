import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from torchvision.models import inception_v3, Inception_V3_Weights
from tqdm import tqdm
from scipy import linalg
import argparse

# Set random seed for reproducibility
def set_seed(seed=None):
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

# Resize grayscale images and save them into corresponding label folders
def resize_and_save_grayscale_images(src_dir, tgt_dir, size, n_per_label):
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

# Load grayscale images into a NumPy array and normalize
def load_images(path, n_per_label):
    images = []
    for folder_name in sorted(os.listdir(path), key=lambda x: int(x)):
        folder_path = os.path.join(path, folder_name)
        img_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')])
        selected = np.random.choice(img_files, min(n_per_label, len(img_files)), replace=False)
        for img_file in selected:
            img_path = os.path.join(folder_path, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = img.astype(np.float32) / 255.0 
                images.append(img)
    images = np.array(images)[:, np.newaxis, :, :]  # Add channel dimension [N, 1, H, W]
    return images

# Custom module to extract features from InceptionV3's Mixed_7c layer
class PartialInceptionNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=True)
        self.inception.Mixed_7c.register_forward_hook(self._hook)
    def _hook(self, module, input, output):
        self._features = output
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # Convert grayscale to 3-channel
        self.inception(x)
        feats = self._features
        feats = nn.functional.adaptive_avg_pool2d(feats, (1,1)).view(x.shape[0], -1)
        return feats

# Extract activation features from Inception model
def get_activations(images, batch_size=32):
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

# Compute mean and covariance of feature activations
def calculate_activation_statistics(images, batch_size=32):
    act = get_activations(images, batch_size)
    mu, sigma = np.mean(act, axis=0), np.cov(act, rowvar=False)
    return mu, sigma

# Compute Frechet Inception Distance (FID) between two distributions
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)

# Wrapper to compute FID from two image sets
def calculate_fid(images1, images2, batch_size=32):
    mu1, sigma1 = calculate_activation_statistics(images1, batch_size)
    mu2, sigma2 = calculate_activation_statistics(images2, batch_size)
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

# Analyze phase fraction using binarization and compute white/black pixel ratios
def phase_fraction_analysis(image_dir, output_root, n_per_label=108, threshold=140):
    import pandas as pd
    print(f"Creating output root folder: {output_root}")
    os.makedirs(output_root, exist_ok=True)
    folder_names = sorted([f for f in os.listdir(image_dir) if os.path.isdir(os.path.join(image_dir, f))])
    print(f"Found folders for analysis: {folder_names}")

    for folder_name in folder_names:
        bin_path = os.path.join(output_root, folder_name)
        print(f"Creating subfolder for label: {bin_path}")
        os.makedirs(bin_path, exist_ok=True)

    excel_output = os.path.join(output_root, "phase_fraction.xlsx")
    txt_output = os.path.join(output_root, "Total_fraction.txt")
    fraction_data = {}
    with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
        for folder_name in folder_names:
            input_path = os.path.join(image_dir, folder_name)
            bin_path = os.path.join(output_root, folder_name)
            os.makedirs(bin_path, exist_ok=True)
            white_fractions, black_fractions, image_data = [], [], []
            file_list = sorted([f for f in os.listdir(input_path) if f.endswith('.png')],
                               key=lambda x: int(os.path.splitext(x)[0]))
            selected = file_list[:n_per_label]
            for file_name in tqdm(selected, desc=f"Processing {folder_name}"):
                try:
                    img = cv2.imread(os.path.join(input_path, file_name), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    min_val, max_val = np.min(img), np.max(img)
                    if max_val != min_val:
                        img_norm = ((img - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
                    else:
                        img_norm = img.astype(np.uint8)
                    _, binary_img = cv2.threshold(img_norm, threshold, 255, cv2.THRESH_BINARY)
                    cv2.imwrite(os.path.join(bin_path, file_name), binary_img)
                    total = binary_img.size
                    white = np.sum(binary_img == 255)
                    black = np.sum(binary_img == 0)
                    white_frac = (white / total) * 100
                    black_frac = (black / total) * 100
                    white_fractions.append(white_frac)
                    black_fractions.append(black_frac)
                    image_data.append({
                        "Image Name": file_name,
                        "White Fraction (%)": white_frac,
                        "Black Fraction (%)": black_frac
                    })
                except Exception as e:
                    print(f"Error processing {file_name} in {folder_name}: {e}")
            fraction_data[folder_name] = {
                "white_fraction_avg": np.mean(white_fractions) if white_fractions else 0,
                "black_fraction_avg": np.mean(black_fractions) if black_fractions else 0
            }
            df = pd.DataFrame(image_data)
            df.to_excel(writer, sheet_name=folder_name, index=False)
    with open(txt_output, "w") as f:
        for folder_name, frac in fraction_data.items():
            f.write(f"{folder_name}:\n")
            f.write(f"  White Fraction Avg: {frac['white_fraction_avg']:.2f}%\n")
            f.write(f"  Black Fraction Avg: {frac['black_fraction_avg']:.2f}%\n")
    print(f"ferrite fraction: {image_dir} → {output_root}")
    print(f"excel : {excel_output} / TXT: {txt_output}")

# Main script for evaluation pipeline
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_src_dir', type=str, default='images_fid')
    parser.add_argument('--real_dir', type=str, default='images_fid_resized')
    parser.add_argument('--gen_src_dir_base', type=str, default='gen_images')     
    parser.add_argument('--gen_dir_base', type=str, default='gen_images_resized') 
    parser.add_argument('--img_size', type=int, nargs=2, default=[299, 299])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_per_label', type=int, default=108)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ckpt_steps', type=str, nargs='+', default=["070000", "080000", "090000", "100000", "110000", "120000", "130000"])
    parser.add_argument('--phase_fraction', action='store_true')
    
    args = parser.parse_args()
    set_seed(args.seed)

    # Resize real images only if resized folder is missing or empty
    if not os.path.exists(args.real_dir) or not os.listdir(args.real_dir):
        print("Resizing real images...")
        resize_and_save_grayscale_images(args.real_src_dir, args.real_dir, tuple(args.img_size), args.n_per_label)

    fid_results = []
    for ckpt_step in args.ckpt_steps:
        print(f"\n===== {ckpt_step} step evaluate start =====")
        gen_src_dir = os.path.join(args.gen_src_dir_base, ckpt_step)
        gen_dst_dir = os.path.join(args.gen_dir_base, ckpt_step)
        if not os.path.exists(gen_dst_dir) or not os.listdir(gen_dst_dir):
            print(f"Resizing fake images for step {ckpt_step} ...")
            resize_and_save_grayscale_images(gen_src_dir, gen_dst_dir, tuple(args.img_size), args.n_per_label)

        print("Loading images...")
        real_imgs = load_images(args.real_dir, args.n_per_label)
        gen_imgs = load_images(gen_dst_dir, args.n_per_label)

        print(f"Shape check [N, 1, {args.img_size[0]}, {args.img_size[1]}]: {real_imgs.shape}, {gen_imgs.shape}")

        fid = calculate_fid(real_imgs, gen_imgs, batch_size=args.batch_size)
        print(f"FID Score [{ckpt_step}]: {fid:.2f}")
        fid_results.append((ckpt_step, fid))

    print("\n== Result ==")
    for step, fid in fid_results:
        print(f"{step}: {fid:.2f}") 

    # If phase fraction analysis is requested
    if args.phase_fraction:
        best_step, best_fid = sorted(fid_results, key=lambda x: x[1])[0]
        print(f"\n==  ({best_step}, {best_fid:.2f}) ==")
        best_gen_dir = os.path.join(args.gen_src_dir_base, best_step)
        print("generator images with a low FID:", best_gen_dir)
        print("existence:", os.path.exists(best_gen_dir))
        print("exists", len(os.listdir(best_gen_dir)) if os.path.exists(best_gen_dir) else "none") 

        phase_fraction_dir = os.path.join("ferrite_fraction", best_step)
        print(f"Phase fraction analysis start on: {best_gen_dir} saving to {phase_fraction_dir}")
        phase_fraction_analysis(best_gen_dir, phase_fraction_dir, n_per_label=args.n_per_label)
        print("Phase fraction analysis done")

if __name__ == "__main__":
    main()
