import os
from PIL import Image
import numpy as np
import torch
import config
from sklearn.model_selection import train_test_split

def save_cropped_images():
    save_root = config.crop_images
    os.makedirs(save_root, exist_ok=True)

    for idx, file in enumerate(config.image_files):
        label_value = 500 + idx * 50  
        img_path = os.path.join(config.image_dir, file)
        img = Image.open(img_path).convert('L')
        img = np.array(img, dtype=np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255

        out_folder = os.path.join(save_root, str(label_value))
        os.makedirs(out_folder, exist_ok=True)

        count = 0
        for i in range(0, img.shape[1], config.stride):
            for j in range(0, img.shape[0], config.stride):
                if i + config.crop_size <= img.shape[1] and j + config.crop_size <= img.shape[0]:
                    crop = img[j:j+config.crop_size, i:i+config.crop_size]
                    out_path = os.path.join(out_folder, f"{count}.png")
                    Image.fromarray(crop.astype(np.uint8)).save(out_path)
                    count += 1

    print(f"Cropped images saved to {save_root}/(label)/")


def load_images():
    root = config.crop_images
    images = []
    labels = []

    folder_names = sorted([f for f in os.listdir(root) if f.isdigit()], key=int)
    label_map = {int(name): idx for idx, name in enumerate(folder_names)}

    for folder_name in folder_names:
        label_value = int(folder_name)
        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for fname in sorted(os.listdir(folder_path), key=lambda x: int(os.path.splitext(x)[0])):
            img_path = os.path.join(folder_path, fname)
            img = Image.open(img_path).convert('L')
            img = np.array(img, dtype=np.float32) /  255.0
            images.append(img)
            labels.append(label_map[label_value]) 

    images = np.array(images, dtype=np.float32)
    images = torch.tensor(images).unsqueeze(1)
    labels = torch.tensor(labels, dtype=torch.long) 

    return images, labels

def save_crop_image_fid():
    save_root = config.crop_images_fid
    os.makedirs(save_root, exist_ok=True)

    for idx, file in enumerate(config.image_files):
        label_value = 500 + idx * 50  
        img_path = os.path.join(config.image_dir, file)
        img = Image.open(img_path).convert('L')
        img = np.array(img, dtype=np.float32)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8) * 255

        out_folder = os.path.join(save_root, str(label_value))
        os.makedirs(out_folder, exist_ok=True)

        count = 0
        for i in range(0, img.shape[1], config.stride2):
            for j in range(0, img.shape[0], config.stride2):
                if i + config.crop_size <= img.shape[1] and j + config.crop_size <= img.shape[0]:
                    crop = img[j:j+config.crop_size, i:i+config.crop_size]
                    out_path = os.path.join(out_folder, f"{count}.png")
                    Image.fromarray(crop.astype(np.uint8)).save(out_path)
                    count += 1

    print(f"Cropped images saved to {save_root}/(label)/")

def run_prepare_data():
    save_cropped_images()       
    images, labels = load_images()
    save_crop_image_fid()


if __name__ == "__main__":
    run_prepare_data()
