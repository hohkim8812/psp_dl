import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, n_classes, image_size):
        super().__init__()
        self.image_size = image_size
        self.label_linear = nn.Linear(n_classes, image_size * image_size)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),  
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*16*16, 1),
            nn.Sigmoid()
        )

    def forward(self, img, c_label):
        c_label = self.label_linear(c_label)
        c_label = c_label.view(img.shape[0], 1, self.image_size, self.image_size)
        x = torch.cat((img, c_label), dim=1)
        x = self.conv(x)
        out = self.fc(x)
        return out

class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(latent_dim + n_classes, 512*4*4)
        self.ups = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), 
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), 
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 8, 8, 0),
            nn.Tanh()
        )

    def forward(self, noise, c_label):
        x = torch.cat((noise, c_label), dim=1)
        x = self.linear(x)
        x = x.view(x.size(0), 512, 4, 4)
        out = self.ups(x)
        return out
