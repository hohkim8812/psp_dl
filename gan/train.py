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
    model_save_path="generator.pth"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    discriminator = Discriminator(n_classes, image_size).to(device)
    generator = Generator(latent_dim, n_classes).to(device)
    d_optimizer = optim.Adam(discriminator.parameters(), d_lr, (d_beta1, d_beta2))
    g_optimizer = optim.Adam(generator.parameters(), g_lr, (g_beta1, g_beta2))
    criterion = nn.BCELoss().to(device)
    writer = SummaryWriter()

    # --------------------
    # Load DataLoaders here!
    # --------------------
    train_loader, test_loader = get_dataloaders()

    for i in range(n_epochs):
        for idx, (imgs_real, label) in enumerate(train_loader):   
            discriminator.zero_grad()
            imgs_real = imgs_real.to(device)
            label_size = label.size(0)
            real_label = torch.full([label_size,1], 1.0, dtype=imgs_real.dtype, device=device)
            fake_label = torch.full([label_size,1], 0.0, dtype=imgs_real.dtype, device=device)
            noise = torch.randn([label_size, latent_dim], device=device)
            c_label = F.one_hot(label, n_classes).to(torch.float32).to(device)
            imgs_fake = generator(noise, c_label)
            output_real = discriminator(imgs_real, c_label)
            output_fake = discriminator(imgs_fake.detach(), c_label)
            d_loss_real = criterion(output_real, real_label)
            d_loss_fake = criterion(output_fake, fake_label)
            d_loss_avg = (d_loss_real + d_loss_fake)/2
            d_optimizer.zero_grad()
            d_loss_avg.backward()
            d_optimizer.step()

            generator.zero_grad()
            g_optimizer.zero_grad()
            output_fake = discriminator(imgs_fake, c_label)
            g_loss = criterion(output_fake, real_label)
            g_loss.backward()
            g_optimizer.step()

            print(i, idx, d_loss_avg.item(), g_loss.item())
        # Uncomment to log if needed
        #writer.add_scalar("Loss/Discriminator", d_loss_avg.item(), i)
        #writer.add_scalar("Loss/Generator", g_loss.item(), i)

    torch.save(generator.state_dict(), model_save_path)
    print(f"Generator model saved as {model_save_path}")

if __name__ == "__main__":
    main()
