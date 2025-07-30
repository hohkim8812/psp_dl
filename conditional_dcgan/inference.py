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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = Generator(latent_dim, len(label_list)).to(device)
    generator.load_state_dict(torch.load(model_path, map_location=device))
    generator.eval()

    for idx, label_value in enumerate(label_list):
        noise = torch.randn([num_images_per_label, latent_dim], device=device)
        c_label = F.one_hot(
            torch.tensor([idx]*num_images_per_label),
            num_classes=len(label_list)
        ).to(torch.float32).to(device)
        with torch.no_grad():
            imgs_fake = generator(noise, c_label).cpu()
        label_dir = os.path.join(output_dir, f"{label_value}")
        os.makedirs(label_dir, exist_ok=True)
        for i in range(num_images_per_label):
            save_image(imgs_fake[i], os.path.join(label_dir, f"{i}.png"))

    print("Image generation completed.")

if __name__ == "__main__":
    main()