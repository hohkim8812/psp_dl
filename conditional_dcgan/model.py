import torch
import torch.nn as nn

class Discriminator(nn.Module):
    """
    Conditional Discriminator for DCGAN.

    Args:
        n_classes (int): Number of class labels (e.g., heat treatment conditions).
        image_size (int): Height and width of the input image (assumed square).

    Forward Input:
        img (Tensor): Real or generated image tensor of shape [B, 1, H, W].
        c_label (Tensor): One-hot encoded class labels of shape [B, n_classes].

    Output:
        Tensor: Probability that the input image is real, of shape [B, 1].
    """
    def __init__(self, n_classes, image_size):
        super().__init__()
        self.image_size = image_size

        # Project class label to match image spatial resolution (1 channel)
        self.label_linear = nn.Linear(n_classes, image_size * image_size)

        # Convolutional feature extractor
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1),             # Input: [B, 2, H, W]
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )

        # Fully connected layer for binary classification (real/fake)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 16 * 16, 1),
            nn.Sigmoid()
        )

    def forward(self, img, c_label):
        # Expand label to [B, 1, H, W] and concatenate with image
        c_label = self.label_linear(c_label)
        c_label = c_label.view(img.shape[0], 1, self.image_size, self.image_size)
        x = torch.cat((img, c_label), dim=1)  # Concatenate on channel axis: [B, 2, H, W]
        x = self.conv(x)
        out = self.fc(x)
        return out


class Generator(nn.Module):
    """
    Conditional Generator for DCGAN.

    Args:
        latent_dim (int): Dimension of input noise vector z.
        n_classes (int): Number of class labels (used for conditional input).

    Forward Input:
        noise (Tensor): Latent noise vector of shape [B, latent_dim].
        c_label (Tensor): One-hot encoded class labels of shape [B, n_classes].

    Output:
        Tensor: Generated image of shape [B, 1, 128, 128] (after upsampling).
    """
    def __init__(self, latent_dim, n_classes):
        super().__init__()

        # Project noise + label into initial feature map
        self.linear = nn.Linear(latent_dim + n_classes, 512 * 4 * 4)

        # Upsample with transposed convolutions to reach 128x128 resolution
        self.ups = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # Output: [B, 256, 8, 8]
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # Output: [B, 128, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 1, 8, 8, 0),    # Output: [B, 1, 128, 128]
            nn.Tanh()  # Output in range [-1, 1]
        )

    def forward(self, noise, c_label):
        # Concatenate latent vector and conditional label
        x = torch.cat((noise, c_label), dim=1)  # Shape: [B, latent_dim + n_classes]
        x = self.linear(x)
        x = x.view(x.size(0), 512, 4, 4)  # Reshape to feature map for deconv
        out = self.ups(x)
        return out
