import torch.nn as nn
import torchvision.models as models
import config

class ResNet_Model(nn.Module):
    def __init__(self):
        super(ResNet_Model, self).__init__()
        if config.resnet_variant == 'resnet18':
            self.resnet = models.resnet18(weights=config.pretrained_weights)
        elif config.resnet_variant == 'resnet50':
            self.resnet = models.resnet50(weights=config.pretrained_weights)
        elif config.resnet_variant == 'resnet101':
            self.resnet = models.resnet101(weights=config.pretrained_weights)
        else:
            raise ValueError(f"Unsupported variant: {config.resnet_variant}")
        self.resnet.conv1 = nn.Conv2d(config.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1)

    def forward(self, x):
        return self.resnet(x)
