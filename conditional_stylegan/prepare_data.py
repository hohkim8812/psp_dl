import argparse
import multiprocessing
import sys
import os
import numpy as np
import lmdb
from tqdm import tqdm
from io import BytesIO
from PIL import Image
from functools import partial
from torchvision import datasets
from torchvision.transforms import functional as trans_fn

# Save 128×128 cropped patches from 5 TIF files
def save_images(folder_path, output_root):
    image_files = [f'{i}.tif' for i in range(1, 6)]  # File names: 1.tif ~ 5.tif
    os.makedirs(output_root, exist_ok=True)

    for file in image_files:
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path).convert('RGB')  # Convert to RGB
        subfolder_name = str(int(file[:-4]))  # Extract number from filename
        image_output_folder = os.path.join(output_root, subfolder_name)
        os.makedirs(image_output_folder, exist_ok=True)

        image_counter = 0
        for i in range(0, img.width, 34):     # Horizontal stride
            for j in range(0, img.height, 33):  # Vertical stride
                if i + 128 <= img.width and j + 128 <= img.height:
                    crop = img.crop((i, j, i + 128, j + 128))
                    crop_filename = f'{image_counter}.png'
                    crop_path = os.path.join(image_output_folder, crop_filename)
                    crop.save(crop_path)
                    image_counter += 1

        print(f"{file} → {image_counter} patches saved in {image_output_folder}")

# Same as save_images but used for FID evaluation (larger stride)
def save_crop_image_fid(folder_path, output_root2):
    image_files = [f'{i}.tif' for i in range(1, 6)]
    os.makedirs(output_root2, exist_ok=True)

    for file in image_files:
        img_path = os.path.join(folder_path, file)
        img = Image.open(img_path).convert('RGB')
        subfolder_name = str(int(file[:-4]))
        image_output_folder = os.path.join(output_root2, subfolder_name)
        os.makedirs(image_output_folder, exist_ok=True)

        image_counter = 0
        for i in range(0, img.width, 96):
            for j in range(0, img.height, 96):
                if i + 128 <= img.width and j + 128 <= img.height:
                    crop = img.crop((i, j, i + 128, j + 128))
                    crop_filename = f'{image_counter}.png'
                    crop_path = os.path.join(image_output_folder, crop_filename)
                    crop.save(crop_path)
                    image_counter += 1

        print(f"{file} → {image_counter} patches saved in {image_output_folder}")

# Resize image and convert to JPEG bytes
def resize_and_convert(img, size, quality=100):
    img = trans_fn.resize(img, size, Image.LANCZOS)
    img = trans_fn.center_crop(img, size)
    buffer = BytesIO()
    img.save(buffer, format='jpeg', quality=quality)
    val = buffer.getvalue()
    return val

# Resize image into multiple resolutions and return as a list of bytes
def resize_multiple(img, sizes=(8, 16, 32, 64, 128, 256, 512, 1024), quality=100):
    imgs = []
    for size in sizes:
        imgs.append(resize_and_convert(img, size, quality))
    return imgs

# Worker function to read and resize a single image
def resize_worker(img_file, sizes):
    i, file, label = img_file
    img = Image.open(file)
    img = img.convert('RGB')
    out = resize_multiple(img, sizes=sizes)
    label = str(label).encode('utf-8')  # Convert label to bytes
    return i, out, label

# Save resized image bytes and labels into two LMDB databases
def prepare(transaction1, transaction2, dataset, n_worker, sizes=(8, 16, 32, 64, 128)):
    resize_fn = partial(resize_worker, sizes=sizes)

    # Sort images by label and assign index
    files = sorted(dataset.imgs, key=lambda x: x[1])
    files = [(i, file, label) for i, (file, label) in enumerate(files)]
    total = 0

    # Resize and store each image and its label into LMDB
    for file in files:
        i, imgs, label = resize_fn(file)
        for size, img in zip(sizes, imgs):
            print(i, f'{size}-{str(i).zfill(5)}')
            key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
            transaction1.put(key, img)     # Save image bytes
            transaction2.put(key, label)   # Save label
            total += 1

    total = total // 5  # Normalize for the 5 input .tif files
    transaction1.put('length'.encode('utf-8'), str(total).encode('utf-8'))
    transaction2.put('length'.encode('utf-8'), str(total).encode('utf-8'))

# Main execution script
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default='../images') # modified
    parser.add_argument('--output_root', type=str, default='output_images/128x128') # modified
    parser.add_argument('--output_root2', type=str, default='images_fid') 
    parser.add_argument('--imgout', type=str, default='images_lmdb/img.lmdb')
    parser.add_argument('--labelout', type=str, default='images_lmdb/label.lmdb')
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--img_src_path', type=str, default='output_images/128x128') # modified

    args = parser.parse_args()

    # Step 1: Crop and save patches from TIF images
    save_images(args.folder_path, args.output_root)
    save_crop_image_fid(args.folder_path, args.output_root2)

    # Step 2: Create ImageFolder dataset and save as LMDB
    imgset = datasets.ImageFolder(args.img_src_path)
    env_imgs = lmdb.open(args.imgout, map_size=2 * 1024 ** 3, readahead=False)
    env_labels = lmdb.open(args.labelout, map_size=64 * 1024 ** 2, readahead=False)
    
    with env_imgs.begin(write=True) as txn_imgs, env_labels.begin(write=True) as txn_labels:
        prepare(txn_imgs, txn_labels, imgset, args.n_worker)
