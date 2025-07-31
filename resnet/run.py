import fire
import config
import torch
from prepare_data import run_prepare_data
from train import train as run_train
from inference import run_inference

# Automatically select CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare(
    target=config.target_col,
    image_dir=None,
    crop_size=None,
    stride=None,
    crop_images=None
):
    """
    Run the data preparation pipeline.

    Args:
        target (int): Index of the target column to use for labels.
        image_dir (str, optional): Override the default image directory.
        crop_size (int, optional): Size of each cropped patch.
        stride (int, optional): Sliding window stride for cropping.
        crop_images (str, optional): Directory to save cropped images.
    """
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
    """
    Train the model using prepared cropped images and labels.

    Args:
        target (int): Index of the target property column.
        image_dir (str, optional): Override the default image directory.
        crop_size (int, optional): Size of cropped image patches.
        stride (int, optional): Stride used in cropping.
        crop_images (str, optional): Directory containing cropped patches.
    """
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
    """
    Run inference on a new set of cropped images using a trained model.

    Args:
        target (int): Index of the target property to predict.
        inference_image_dir (str, optional): Path to directory of inference images.
    """
    config.target_col = target
    if inference_image_dir:
        config.inference_image_dir = inference_image_dir

    run_inference()  # Run model prediction and save results to Excel


def both(target=config.target_col, **kwargs):
    """
    Run data preparation, training, and inference in sequence.

    Args:
        target (int): Target column index.
        **kwargs: Keyword arguments passed to `train()` and `inference()`.
    """
    print(f"\n Running both train and inference for target_col = {target}")
    config.target_col = target

    run_prepare_data(target)  # Step 1: Crop images and generate labels
    train(target=target, **kwargs)  # Step 2: Train model
    inference(target=target, **kwargs)  # Step 3: Run inference


if __name__ == "__main__":
    # Use Fire to expose CLI interface with commands: prepare, train, inference, both
    fire.Fire({
        "prepare": prepare,
        "train": train,
        "inference": inference,
        "both": both
    })
