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
    Crop original images and prepare training and FID datasets.

    Args:
        image_dir (str): Directory with original image files.
        crop_size (int): Size of each square crop.
        stride (int): Stride for training image cropping.
        stride2 (int): Stride for FID-compatible image cropping.
        image_files (list): Filenames to load and crop.
        crop_images (str): Path to save training crops.
        crop_images_fid (str): Path to save FID crops.
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
    Train conditional GAN with provided hyperparameters.

    Args:
        n_epochs (int): Number of training epochs.
        d_lr (float): Learning rate for discriminator.
        g_lr (float): Learning rate for generator.
        d_beta1 (float): Beta1 for discriminator Adam optimizer.
        d_beta2 (float): Beta2 for discriminator Adam optimizer.
        g_beta1 (float): Beta1 for generator Adam optimizer.
        g_beta2 (float): Beta2 for generator Adam optimizer.
        latent_dim (int): Dimension of latent noise vector.
        n_classes (int): Number of conditioning class labels.
        batch_size (int): Training batch size.
        image_size (int): Height/width of input image.
        crop_images (str): Directory containing training crops.
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
    Generate synthetic images using a trained generator.

    Args:
        num_images_per_label (int): Number of images to generate per label.
        label_list (list): List of labels to condition generation on.
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
    Evaluate generated images against real images using FID.

    Args:
        real_src_dir (str): Folder with original real images.
        real_dir (str): Folder to save resized real images.
        gen_src_dir (str): Folder with generated images.
        gen_dir (str): Folder to save resized generated images.
        img_size (tuple): Target size for FID input (e.g., (299, 299)).
        batch_size (int): Batch size for Inception feature extraction.
        n_per_label (int): Number of images per label for FID.
        seed (int): Random seed for reproducibility.
    """
    evaluate_main(real_src_dir, real_dir, gen_src_dir, gen_dir, img_size, batch_size, n_per_label, seed)


def all_steps():
    """
    Run full pipeline: prepare → train → inference → evaluate.
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
    # Command-line interface entrypoint using `fire`
    fire.Fire({
        "prepare": prepare_data,    # Run data preprocessing
        "train": train,             # Train the GAN model
        "inference": inference,     # Generate images
        "evaluate": evaluate,       # Compute FID
        "all": all_steps            # Run full pipeline
    })
