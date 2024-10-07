import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torch.nn.init as init

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global max pooling

        # MLP layers
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.avg_pool(x)
        avg_out = self.fc1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.fc2(avg_out)

        max_out = self.max_pool(x)
        max_out = self.fc1(max_out)
        max_out = self.relu(max_out)
        max_out = self.fc2(max_out)

        out = avg_out + max_out
        out = self.sigmoid(out)
        return out

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(out)
        out = self.sigmoid(out)
        return out

class SelfAttention(nn.Module):
    def __init__(self, hidden_size, mean_only=False):
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size
        self.att_weights = nn.Parameter(torch.Tensor(1, hidden_size))
        self.mean_only = mean_only
        init.kaiming_uniform_(self.att_weights)

    def forward(self, inputs):
        # inputs: (batch_size, seq_length, hidden_size)
        batch_size, seq_length, hidden_size = inputs.size()
        assert hidden_size == self.hidden_size, "Input hidden size does not match SelfAttention hidden size."

        # Compute attention weights
        weights = torch.bmm(inputs, self.att_weights.unsqueeze(0).expand(batch_size, -1, -1).permute(0, 2, 1))
        attentions = F.softmax(torch.tanh(weights), dim=1)
        weighted = inputs * attentions.expand_as(inputs)

        if self.mean_only:
            return weighted.sum(1)  # (batch_size, hidden_size)
        else:
            noise = 1e-5 * torch.randn_like(weighted)
            avg_repr = weighted.sum(1)  # (batch_size, hidden_size)
            std_repr = (weighted + noise).std(1)  # (batch_size, hidden_size)
            representations = torch.cat((avg_repr, std_repr), dim=1)  # (batch_size, 2 * hidden_size)
            return representations

class BasicBlock_C(nn.Module):
    def __init__(self, in_planes, bottleneck_width=4, cardinality=32, stride=1, expansion=2, use_se=True, reduction=16):
        super(BasicBlock_C, self).__init__()
        inner_width = cardinality * bottleneck_width
        self.expansion = expansion
        planes = inner_width * expansion  # Correctly compute planes

        self.basic = nn.Sequential(OrderedDict(
            [
                ('conv1_0', nn.Conv2d(in_planes, inner_width, kernel_size=1, stride=1, bias=False)),
                ('bn1', nn.BatchNorm2d(inner_width)),
                ('act0', nn.ReLU()),
                ('conv3_0', nn.Conv2d(inner_width, inner_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)),
                ('bn2', nn.BatchNorm2d(inner_width)),
                ('act1', nn.ReLU()),
                ('conv1_1', nn.Conv2d(inner_width, planes, kernel_size=1, stride=1, bias=False)),
                ('bn3', nn.BatchNorm2d(planes))
            ]
        ))

        self.channel = ChannelAttention(planes)  # Channel Attention Module
        self.spatial = SpatialAttention()        # Spatial Attention Module

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False)
            )
        self.bn0 = nn.BatchNorm2d(planes)

        self.out_channels = planes  # Store planes for later use

    def forward(self, x):
        out = self.basic(x)
        CBAM_Cout = self.channel(out)
        out = out * CBAM_Cout
        CBAM_Sout = self.spatial(out)
        out = out * CBAM_Sout
        out += self.shortcut(x)
        out = F.relu(self.bn0(out))
        return out

class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, expansion=2, num_classes=2, mean_only=False):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.expansion = expansion
        self.in_planes = 64

        self.conv0 = nn.Conv2d(1, self.in_planes, kernel_size=3, stride=1, padding=1)
        self.bn0 = nn.BatchNorm2d(self.in_planes)
        self.pool0 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(num_blocks[0], stride=1)
        self.layer2 = self._make_layer(num_blocks[1], stride=2)
        self.layer3 = self._make_layer(num_blocks[2], stride=2)
        self.layer4 = self._make_layer(num_blocks[3], stride=2)

        # Self-Attention Module
        self.attention = SelfAttention(hidden_size=self.in_planes, mean_only=mean_only)
        if mean_only:
            self.fc = nn.Linear(self.in_planes, 256)
        else:
            self.fc = nn.Linear(2 * self.in_planes, 256)  # Adjust input size based on attention output
        self.fc_mu = nn.Linear(256, num_classes)

    def forward(self, x):
        out = F.relu(self.bn0(self.conv0(x)))  # Initial convolution and activation
        out = self.pool0(out)                  # Initial pooling
        out = self.layer1(out)                 # Layer 1
        out = self.layer2(out)                 # Layer 2
        out = self.layer3(out)                 # Layer 3
        out = self.layer4(out)                 # Layer 4

        # Aggregate over the spatial dimensions (width dimension)
        out = out.mean(dim=3)  # Now out has shape (batch_size, channels, height)
        out = out.permute(0, 2, 1)  # Shape becomes (batch_size, height, channels)

        # Apply Self-Attention over the sequence length (height)
        stats = self.attention(out)  # Output shape depends on mean_only

        out = self.fc(stats)
        mu = self.fc_mu(out)
        return out, mu

    def _make_layer(self, num_blocks, stride):
        layers = []
        for i in range(num_blocks):
            block = BasicBlock_C(
                in_planes=self.in_planes,
                bottleneck_width=self.bottleneck_width,
                cardinality=self.cardinality,
                stride=stride if i == 0 else 1,
                expansion=self.expansion,
                use_se=True
            )
            layers.append(block)
            self.in_planes = block.out_channels  # Update in_planes to the output channels
        return nn.Sequential(*layers)
