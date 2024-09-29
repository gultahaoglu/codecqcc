# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 20:29:58 2024

@author: ADMIN
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:19:04 2024

@author: ADMIN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel-wise attention."""
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(batch_size, channels, 1, 1)
        return x * y

class SpatialAttention(nn.Module):
    """Spatial Attention Module."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        assert kernel_size in (3, 7), 'Kernel size must be 3 or 7'
        padding = (kernel_size - 1) // 2  # To maintain the input size
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Compute average and max pooling along the channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Shape: [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Shape: [B, 1, H, W]
        # Concatenate along the channel dimension
        x_cat = torch.cat([avg_out, max_out], dim=1)  # Shape: [B, 2, H, W]
        # Apply convolution and sigmoid activation
        attention = self.sigmoid(self.conv(x_cat))  # Shape: [B, 1, H, W]
        return x * attention  # Element-wise multiplication

class BasicBlock_C(nn.Module):
    def __init__(self, in_planes, bottleneck_width=4, cardinality=32, stride=1, expansion=2, use_se=True):
        super(BasicBlock_C, self).__init__()
        inner_width = cardinality * bottleneck_width
        self.expansion = expansion
        self.basic = nn.Sequential(OrderedDict(
            [
                ('conv1_0', nn.Conv2d(in_planes, inner_width, 1, stride=1, bias=False)),
                ('bn1', nn.BatchNorm2d(inner_width)),
                ('act0', nn.ReLU()),
                ('conv3_0', nn.Conv2d(inner_width, inner_width, 3, stride=stride, padding=1, groups=cardinality, bias=False)),
                ('bn2', nn.BatchNorm2d(inner_width)),
                ('act1', nn.ReLU()),
                ('conv1_1', nn.Conv2d(inner_width, inner_width * self.expansion, 1, stride=1, bias=False)),
                ('bn3', nn.BatchNorm2d(inner_width * self.expansion))
            ]
        ))
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != inner_width * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, inner_width * self.expansion, 1, stride=stride, bias=False)
            )
        self.bn0 = nn.BatchNorm2d(self.expansion * inner_width)
        
        # Squeeze-and-Excitation layer (channel attention)
        self.se = SEBlock(inner_width * self.expansion) if use_se else nn.Identity()
        # Spatial Attention layer
        self.sa = SpatialAttention() if use_se else nn.Identity()

    def forward(self, x):
        out = self.basic(x)
        out = self.se(out)  # Apply the SE block (channel attention)
        out = self.sa(out)  # Apply the Spatial Attention block
        out += self.shortcut(x)
        out = F.relu(self.bn0(out))
        return out

class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, expansion=2, num_classes=2):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.expansion = expansion
        
        self.conv0 = nn.Conv2d(1, self.in_planes, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(self.in_planes)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.layer4 = self._make_layer(num_blocks[3], 2)
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive Pooling to ensure flattening works
        
        self.linear = nn.Linear(self.cardinality * self.bottleneck_width * self.expansion // 2, 256)
        self.fc_mu = nn.Linear(256, num_classes) 

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))
        out = self.pool0(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.pool1(out)  # This should reduce to (batch_size, channels, 1, 1)
        out = out.view(out.size(0), -1)  # Flatten to (batch_size, channels)
        out = self.linear(out)  # Linear layer for classification
        mu = self.fc_mu(out)
        return out, mu

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock_C(self.in_planes, self.bottleneck_width, self.cardinality, stride, self.expansion, use_se=True))
            self.in_planes = self.expansion * self.bottleneck_width * self.cardinality
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

# Example usage:
def resnext26_16x8d(num_classes=2):
    return ResNeXt(num_blocks=[2, 2, 2, 2], cardinality=16, bottleneck_width=8, num_classes=num_classes)

# Testing with the provided input shape
net = resnext26_16x8d(num_classes=2)
data = torch.rand(16, 1, 60, 750)
output, mu = net(data)
print("Output shape:", output.shape)
print("Mu shape:", mu.shape)
