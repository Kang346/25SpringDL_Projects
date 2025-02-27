import torch
import torch.nn as nn
import torchvision.models as models

class ResNetCIFAR10(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetCIFAR10, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)