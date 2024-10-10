# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 23:03:29 2024

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
    """Kanal bazlı dikkat için Sıkıştırma ve Uyarım Bloğu."""
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
    """Uzamsal dikkat bloğu."""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Ortalama havuzlama
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Maksimum havuzlama
        x = torch.cat([avg_out, max_out], dim=1)  # Kanal boyutunda birleştirme
        x = F.relu(self.bn(self.conv1(x)))
        attention = torch.sigmoid(x)
        return attention

class BasicBlock_C(nn.Module):
    def __init__(self, in_planes, bottleneck_width=4, cardinality=32, stride=1, expansion=2, use_se=True, use_sa=True):
        super(BasicBlock_C, self).__init__()
        inner_width = cardinality * bottleneck_width
        self.expansion = expansion
        out_planes = inner_width * self.expansion

        self.basic = nn.Sequential(OrderedDict(
            [
                ('conv1_0', nn.Conv2d(in_planes, inner_width, kernel_size=1, stride=1, bias=False)),
                ('bn1', nn.BatchNorm2d(inner_width)),
                ('act0', nn.ReLU(inplace=True)),
                ('conv3_0', nn.Conv2d(inner_width, inner_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)),
                ('bn2', nn.BatchNorm2d(inner_width)),
                ('act1', nn.ReLU(inplace=True)),
                ('conv1_1', nn.Conv2d(inner_width, out_planes, kernel_size=1, stride=1, bias=False)),
                ('bn3', nn.BatchNorm2d(out_planes))
            ]
        ))

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_planes)
            )
        
        # Sıkıştırma ve Uyarım katmanı
        self.se = SEBlock(out_planes) if use_se else nn.Identity()
        
        # Uzamsal Dikkat katmanı
        self.sa = SpatialAttention() if use_sa else nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.basic(x)
        residual = self.shortcut(x)
        out += residual
        out = self.se(out)  # SE bloğunu kısayol eklemesinden sonra uygula
        attention = self.sa(out)  # Spatial Attention uygulaması
        out = out * attention  # Dikkat haritasını uygula
        out = self.relu(out)
        return out

class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, expansion=2, num_classes=2):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64
        self.expansion = expansion
        
        self.conv0 = nn.Conv2d(1, self.in_planes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(num_blocks[0], stride=1)
        self.layer2 = self._make_layer(num_blocks[1], stride=2)
        self.layer3 = self._make_layer(num_blocks[2], stride=2)
        self.layer4 = self._make_layer(num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Düzleştirme için adaptif havuzlama
        self.fc = nn.Linear(self.in_planes, num_classes)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.pool0(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)  # (batch_size, channels, 1, 1) boyutuna indirger
        x = torch.flatten(x, 1)  # (batch_size, channels) boyutunda düzleştir
        x = self.fc(x)
        return x

    def _make_layer(self, num_blocks, stride):
        layers = []
        layers.append(BasicBlock_C(self.in_planes, self.bottleneck_width, self.cardinality, stride, self.expansion))
        self.in_planes = self.bottleneck_width * self.cardinality * self.expansion
        for _ in range(1, num_blocks):
            layers.append(BasicBlock_C(self.in_planes, self.bottleneck_width, self.cardinality, expansion=self.expansion))
        return nn.Sequential(*layers)

# Örnek kullanım:
def resnext50(num_classes=2):
    return ResNeXt(num_blocks=[3, 4, 6, 3], cardinality=32, bottleneck_width=4, num_classes=num_classes)

# Modeli oluşturma ve test etme
if __name__ == "__main__":
    net = resnext50(num_classes=2)
    data = torch.rand(16, 1, 224, 224)  # Giriş boyutu [batch_size, channels, height, width]
    output = net(data)
    print(output.shape)  # Beklenen çıktı boyutu: [16, 2]
