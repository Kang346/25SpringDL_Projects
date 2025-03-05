import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection with 1x1 convolution when downsampling
        self.downsample = downsample
        if downsample:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_channels)
        else:
            self.skip_conv = None

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.skip_bn(self.skip_conv(x))

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity  
        return F.relu(out)
    
class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()

        # Initial Convolution (C1 = 32 channels)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Residual layers with Max Pooling after each layer
        self.layer1 = self._make_layer(32, 32, num_blocks=2, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)  # Max pooling after layer1

        self.layer2 = self._make_layer(32, 64, num_blocks=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)  # Max pooling after layer2

        self.layer3 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)  # Max pooling after layer3

        self.layer4 = self._make_layer(128, 256, num_blocks=2, stride=2)

        # Average Pooling (Final layer)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)

        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1024, 800),  # First FC layer
            nn.ReLU(),
            nn.Linear(800, num_classes)  # Output layer
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, downsample=True))
        layers.append(ResidualBlock(out_channels, out_channels, stride=1, downsample=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.layer1(x)
        x = self.pool1(x)  # Apply max pooling after layer1

        x = self.layer2(x)
        x = self.pool2(x)  # Apply max pooling after layer2

        x = self.layer3(x)
        x = self.pool3(x)  # Apply max pooling after layer3

        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x