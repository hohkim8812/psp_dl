import torch
import torch.nn as nn
import config

class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.img_size = config.image_size        
        self.init_size = config.image_size // 8  

        self.fc = nn.Linear(
            latent_dim + n_classes, 128 * self.init_size * self.init_size)

        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Upsample(scale_factor=2), nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, c_label):
        x = torch.cat([z, c_label], dim=1)  
        out = self.fc(x)
        out = out.view(-1, 128, self.init_size, self.init_size)  
        img = self.net(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, n_classes, image_size):
        super().__init__()
        self.n_classes = n_classes
        self.image_size = image_size
        self.label_linear = nn.Linear(n_classes, image_size * image_size)
        self.main = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * (image_size // 4) * (image_size // 4), 1), 
            nn.Sigmoid()
        )
    def forward(self, img, c_label):
        c_label = self.label_linear(c_label)
        c_label = c_label.view(img.shape[0], 1, self.image_size, self.image_size)
        x = torch.cat((img, c_label), dim=1)
        out = self.main(x)
        return out

