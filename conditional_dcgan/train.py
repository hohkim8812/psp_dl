import torch
import config
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataset import get_dataloaders
from model import Discriminator, Generator
from torch.utils.tensorboard import SummaryWriter

def main(
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
    crop_images=config.crop_images,
    model_save_path="generator.pth"
):
    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize discriminator and generator models
    discriminator = Discriminator(n_classes, image_size).to(device)
    generator = Generator(latent_dim, n_classes).to(device)

    # Setup optimizers for both networks
    d_optimizer = optim.Adam(discriminator.parameters(), d_lr, (d_beta1, d_beta2))
    g_optimizer = optim.Adam(generator.parameters(), g_lr, (g_beta1, g_beta2))

    # Binary cross-entropy loss for both real/fake classification
    criterion = nn.BCELoss().to(device)

    # TensorBoard writer (optional)
    writer = SummaryWriter()

    # Load training and test data loaders
    train_loader, test_loader = get_dataloaders(crop_images)

    # Start training loop
    for i in range(n_epochs):
        for idx, (imgs_real, label) in enumerate(train_loader):   
            # === Discriminator training ===
            discriminator.zero_grad()

            imgs_real = imgs_real.to(device)
            label_size = label.size(0)

            # Create real and fake label tensors
            real_label = torch.full([label_size, 1], 1.0, dtype=imgs_real.dtype, device=device)
            fake_label = torch.full([label_size, 1], 0.0, dtype=imgs_real.dtype, device=device)

            # Generate noise and corresponding one-hot labels
            noise = torch.randn([label_size, latent_dim], device=device)
            c_label = F.one_hot(label, n_classes).to(torch.float32).to(device)

            # Generate fake images
            imgs_fake = generator(noise, c_label)

            # Forward pass real and fake images through discriminator
            output_real = discriminator(imgs_real, c_label)
            output_fake = discriminator(imgs_fake.detach(), c_label)

            # Compute discriminator loss
            d_loss_real = criterion(output_real, real_label)
            d_loss_fake = criterion(output_fake, fake_label)
            d_loss_avg = (d_loss_real + d_loss_fake) / 2

            # Update discriminator
            d_optimizer.zero_grad()
            d_loss_avg.backward()
            d_optimizer.step()

            # === Generator training ===
            generator.zero_grad()
            g_optimizer.zero_grad()

            # Recompute discriminator prediction on fake images (updated generator)
            output_fake = discriminator(imgs_fake, c_label)

            # Generator tries to fool the discriminator
            g_loss = criterion(output_fake, real_label)

            # Update generator
            g_loss.backward()
            g_optimizer.step()

            # Print loss values every batch
            print(i, idx, d_loss_avg.item(), g_loss.item())

        # Optional: log losses to TensorBoard
        # writer.add_scalar("Loss/Discriminator", d_loss_avg.item(), i)
        # writer.add_scalar("Loss/Generator", g_loss.item(), i)

    # Save trained generator model
    torch.save(generator.state_dict(), model_save_path)
    print(f"Generator model saved as {model_save_path}")

if __name__ == "__main__":
    main()
