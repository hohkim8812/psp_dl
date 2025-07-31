import torch.nn as nn
import torchvision.models as models
import config

class ResNet_Model(nn.Module):
    """
    A configurable ResNet-based regression model for predicting a single target value.

    The architecture can be selected from 'resnet18', 'resnet50', or 'resnet101'
    based on the `config.resnet_variant` setting. It replaces the first convolution
    layer to support custom input channels and modifies the final fully connected
    layer for regression output.

    Attributes:
        resnet (nn.Module): The backbone ResNet model with adjusted input and output layers.
    """
    def __init__(self):
        super(ResNet_Model, self).__init__()

        # Load pretrained ResNet model according to the specified variant
        if config.resnet_variant == 'resnet18':
            self.resnet = models.resnet18(weights=config.pretrained_weights)
        elif config.resnet_variant == 'resnet50':
            self.resnet = models.resnet50(weights=config.pretrained_weights)
        elif config.resnet_variant == 'resnet101':
            self.resnet = models.resnet101(weights=config.pretrained_weights)
        else:
            raise ValueError(f"Unsupported variant: {config.resnet_variant}")  # Raise error for invalid variant

        # Replace the first convolutional layer to match the input channel count
        self.resnet.conv1 = nn.Conv2d(
            config.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Get the number of input features for the original fully connected layer
        num_ftrs = self.resnet.fc.in_features

        # Replace the final fully connected layer for single-value regression output
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        """
        Forward pass through the modified ResNet model.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)

        Returns:
            torch.Tensor: Output tensor of shape (B, 1)
        """
        return self.resnet(x)
