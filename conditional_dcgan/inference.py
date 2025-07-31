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
    Generate and save synthetic images using a trained generator model.

    Args:
        num_images_per_label (int): Number of images to generate per label.
        label_list (List[int]): List of label values (e.g., heat treatment conditions).
        output_dir (str): Directory to save generated images.
        latent_dim (int): Dimensionality of latent noise vector z.
        model_path (str): Path to the saved generator model (.pth).
    """
    # Set device (GPU if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize generator model and load weights
    generator = Generator(latent_dim, len(label_list)).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    # Loop over each class label
    for idx, label_value in enumerate(label_list):
        # Sample latent vectors z ~ N(0,1)
        noise = torch.randn([num_images_per_label, latent_dim], device=device)

        # Create one-hot encoded conditional labels
        c_label = F.one_hot(
            torch.tensor([idx]*num_images_per_label),
            num_classes=len(label_list)
        ).to(torch.float32).to(device)

        # Generate fake images without tracking gradients
        with torch.no_grad():
            imgs_fake = generator(noise, c_label).cpu()

        # Save each generated image into a subfolder named by label (e.g., "500", "550", ...)
        label_dir = os.path.join(output_dir, f"{label_value}")
        os.makedirs(label_dir, exist_ok=True)

        for i in range(num_images_per_label):
            save_image(imgs_fake[i], os.path.join(label_dir, f"{i}.png"))

    print("Image generation completed.")

if __name__ == "__main__":
    main()
