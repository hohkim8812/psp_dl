import fire
import config
from prepare_data import prepare_main
from train import main as train_main
from inference import main as inference_main
from evaluate import main as evaluate_main

def prepare_data(
    image_dir=config.image_dir,
    crop_size=config.crop_size,
    stride=config.stride,
    stride2=config.stride2,
    image_files=config.image_files,
    crop_images=config.crop_images,
    crop_images_fid=config.crop_images_fid,
):
    """
    Crop original images and prepare datasets for training and FID evaluation.

    Args:
        image_dir (str): Directory containing original images.
        crop_size (int): Size of each cropped patch.
        stride (int): Stride for training crop.
        stride2 (int): Stride for FID crop.
        image_files (list): List of image filenames.
        crop_images (str): Directory to save training patches.
        crop_images_fid (str): Directory to save FID patches.
    """
    prepare_main(image_dir, crop_size, stride, stride2, image_files, crop_images, crop_images_fid)


def train(
    n_epochs=config.n_epochs,
    d_lr=config.d_lr,
    g_lr=config.g_lr,
    d_beta1=config.d_beta1,
    d_beta2=config.d_beta2,
    g_beta1=config.g_beta1,
    g_beta2=config.g_beta2,
    latent_dim=config.latent_dim,
    n_classes=config.n_classes,
    batch_size=config.train_batch_size,
    image_size=config.image_size,
    crop_images=config.crop_images
):
    """
    Train the conditional GAN (cGAN) model.

    Args:
        n_epochs (int): Number of training epochs.
        d_lr (float): Learning rate for the discriminator.
        g_lr (float): Learning rate for the generator.
        d_beta1 (float): Beta1 for the discriminator optimizer.
        d_beta2 (float): Beta2 for the discriminator optimizer.
        g_beta1 (float): Beta1 for the generator optimizer.
        g_beta2 (float): Beta2 for the generator optimizer.
        latent_dim (int): Dimension of the latent noise vector.
        n_classes (int): Number of conditioning classes.
        batch_size (int): Batch size for training.
        image_size (int): Input image size.
        crop_images (str): Directory containing cropped training images.
    """
    train_main(n_epochs, d_lr, g_lr, d_beta1, d_beta2, g_beta1, g_beta2,
               latent_dim, n_classes, batch_size, image_size, crop_images)


def inference(
    num_images_per_label=config.num_images_per_label,
    label_list=config.label_list,
    output_dir=config.output_dir,
    latent_dim=config.latent_dim
):
    """
    Generate microstructure images using the trained generator 
    conditioned on given label values.

    Args:
        num_images_per_label (int): Number of images to generate per label.
        label_list (list): List of label values (e.g., [500, 550, ...]).
        output_dir (str): Directory to save generated images.
        latent_dim (int): Dimension of the latent noise vector.
    """
    inference_main(num_images_per_label, label_list, output_dir, latent_dim)


def evaluate(
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
    Evaluate image generation quality using Frechet Inception Distance (FID).

    Args:
        real_src_dir (str): Directory of real cropped images.
        real_dir (str): Directory of resized real images.
        gen_src_dir (str): Directory of generated images.
        gen_dir (str): Directory of resized generated images.
        img_size (tuple): Target size for FID evaluation (e.g., (299, 299)).
        batch_size (int): Batch size for FID evaluation.
        n_per_label (int): Number of images to sample per label.
        seed (int): Random seed for reproducibility.
    """
    evaluate_main(real_src_dir, real_dir, gen_src_dir, gen_dir, img_size, batch_size, n_per_label, seed)


def all_steps():
    """
    Execute the full pipeline:
    1. Data preparation
    2. Model training
    3. Image generation
    4. Image quality evaluation (FID)
    """
    print("== Step 1: Prepare Data ==")
    prepare_data()

    print("== Step 2: Train Model ==")
    train()

    print("== Step 3: Inference ==")
    inference()

    print("== Step 4: Evaluate ==")
    evaluate()


if __name__ == "__main__":
    # Expose CLI commands using fire
    fire.Fire({
        "prepare": prepare_data,    # Run data preprocessing
        "train": train,             # Train the GAN model
        "inference": inference,     # Generate synthetic images
        "evaluate": evaluate,       # Evaluate FID score
        "all": all_steps            # Execute full workflow
    })
