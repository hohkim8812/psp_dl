import fire
import config
import torch
from prepare_data import run_prepare_data
from train import train as run_train
from inference import run_inference

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare(
    target=config.target_col,
    image_dir=None,
    crop_size=None,
    stride=None,
    crop_images=None
):
    config.target_col = target
    if image_dir: config.image_dir = image_dir
    if crop_size: config.crop_size = int(crop_size)
    if stride: config.stride = int(stride)
    if crop_images: config.crop_images = crop_images
    run_prepare_data(target)

def train(
    target=config.target_col,
    image_dir=None,
    crop_size=None,
    stride=None,
    crop_images=None
):
    config.target_col = target
    if image_dir: config.image_dir = image_dir
    if crop_size: config.crop_size = int(crop_size)
    if stride: config.stride = int(stride)
    if crop_images: config.crop_images = crop_images
    run_train()

def inference(
    target=config.target_col,
    inference_image_dir=None
):
    config.target_col = target
    if inference_image_dir: config.inference_image_dir = inference_image_dir
    run_inference()

def both(target=config.target_col, **kwargs):
    print(f"\n Running both train and inference for target_col = {target}")
    config.target_col = target
    run_prepare_data(target)
    train(target=target, **kwargs)
    inference(target=target, **kwargs)

if __name__ == "__main__":
    fire.Fire({
        "prepare": prepare,
        "train": train,
        "inference": inference,
        "both": both
    })