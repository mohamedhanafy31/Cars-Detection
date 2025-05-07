# feature_extraction/extractor.py

import torch
import torch.nn as nn
import torchvision.models as models

class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        # Use ResNet18 instead of ResNet101
        self.backbone = models.resnet18(weights=None)
        # Remove the last two layers (avgpool and fc)
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        self.out_channels = 512  # ResNet18's final conv layer outputs 512 channels

    def forward(self, x):
        return self.backbone(x)

def build_backbone():
    return ResNetBackbone()
