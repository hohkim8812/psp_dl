import torch
import os
import config
from model import Generator
import torch.nn.functional as F
from torchvision.utils import save_image


def main(
    num_images_per_label=config.num_images_per_label,
    label_list=config.label_list,
    output_dir=config.output_dir,
    latent_dim=config.latent_dim,
    model_path="generator.pth"
):
    """
    Generate synthetic images using a trained conditional generator model.

    Args:
        num_images_per_label (int): Number of images to generate per class label.
        label_list (list): List of label values (e.g., temperatures or categories).
        output_dir (str): Root directory where generated images will be saved.
        latent_dim (int): Dimension of the noise (latent) vector.
        model_path (str): Path to the saved generator model weights.
    """
    # Select device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize generator model and load trained weights
    generator = Generator(latent_dim, len(label_list)).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()  # Set model to evaluation mode

    # Loop through each class label to generate images
    for idx, label_value in enumerate(label_list):
        # Sample latent noise vectors from standard normal distribution
        noise = torch.randn([num_images_per_label, latent_dim], device=device)

        # Create one-hot encoded class labels
        c_label = F.one_hot(
            torch.tensor([idx] * num_images_per_label),
            num_classes=len(label_list)
        ).to(torch.float32).to(device)

        # Generate fake images without computing gradients
        with torch.no_grad():
            imgs_fake = generator(noise, c_label).cpu()
            
        # Save images to class-specific subdirectory
        label_dir = os.path.join(output_dir, f"{label_value}")
        os.makedirs(label_dir, exist_ok=True)
        for i in range(num_images_per_label):
            save_image(imgs_fake[i], os.path.join(label_dir, f"{i}.png"))

    print("Image generation completed.")


if __name__ == "__main__":
    main()
