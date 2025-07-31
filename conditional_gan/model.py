import torch
import torch.nn as nn
import config


class Generator(nn.Module):
    """
    Conditional Generator network for image generation using latent vectors and class labels.

    Architecture:
        - Input: Concatenated latent vector (z) and one-hot class label.
        - Fully connected layer reshapes input to initial feature map.
        - Three upsampling + convolutional blocks progressively generate full-size image.
        - Final activation is Tanh for output range [-1, 1].

    Args:
        latent_dim (int): Dimension of the latent vector z.
        n_classes (int): Number of discrete class labels.
    """
    def __init__(self, latent_dim, n_classes):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.img_size = config.image_size
        self.init_size = config.image_size // 8

        # Fully connected layer to expand z + label to feature map
        self.fc = nn.Linear(
            latent_dim + n_classes, 128 * self.init_size * self.init_size)

        # Upsample blocks to scale feature map to full resolution
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(True),

            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z, c_label):
        """
        Forward pass for the generator.

        Args:
            z (Tensor): Latent noise vector of shape (B, latent_dim).
            c_label (Tensor): One-hot class label vector of shape (B, n_classes).

        Returns:
            Tensor: Generated image of shape (B, 1, H, W)
        """
        x = torch.cat([z, c_label], dim=1)
        out = self.fc(x)
        out = out.view(-1, 128, self.init_size, self.init_size)
        img = self.net(out)


class Discriminator(nn.Module):
    """
    Conditional Discriminator network that classifies whether input image is real or fake.

    Architecture:
        - Class label is projected and reshaped to match image size.
        - Image and label are concatenated as separate channels.
        - 2-layer convolutional encoder followed by linear classification.

    Args:
        n_classes (int): Number of class labels (used in one-hot input).
        image_size (int): Height/width of input image (assumed square).
    """
    def __init__(self, n_classes, image_size):
        super().__init__()
        self.n_classes = n_classes
        self.image_size = image_size

        # Project label to a 2D mask same size as image
        self.label_linear = nn.Linear(n_classes, image_size * image_size)

        # Convolutional discriminator network
        self.main = nn.Sequential(
            nn.Conv2d(2, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * (image_size // 4) * (image_size // 4), 1),
            nn.Sigmoid()
        )

    def forward(self, img, c_label):
        """
        Forward pass for the discriminator.

        Args:
            img (Tensor): Input image of shape (B, 1, H, W).
            c_label (Tensor): One-hot class label of shape (B, n_classes).

        Returns:
            Tensor: Probability scalar for each input (B, 1)
        """
        # Expand class label to same spatial size as image
        c_label = self.label_linear(c_label)
        c_label = c_label.view(img.shape[0], 1, self.image_size, self.image_size)

        # Concatenate label map and image as input channels
        x = torch.cat((img, c_label), dim=1)

        # Classify real vs fake
        out = self.main(x)
        return out
