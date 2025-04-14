import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights

class ResNet2D(nn.Module):
    def __init__(self, model_type="resnet18", num_classes=5, pretrained=True, dropout=0):
        """
        ResNet2D model for ECG classification, treating the 12-lead ECG as a 2D input.
        
        Args:
            model_type (str): ResNet architecture type.
            num_classes (int): Number of output classes.
            pretrained (bool): If True, loads ImageNet weights. If False, random init.
            dropout (float)
        """
        super(ResNet2D, self).__init__()

        # Define available weights
        weights_dict = {
            "resnet18": ResNet18_Weights.DEFAULT,
            "resnet34": ResNet34_Weights.DEFAULT,
            "resnet50": ResNet50_Weights.DEFAULT,
            "resnet101": ResNet101_Weights.DEFAULT,
            "resnet152": ResNet152_Weights.DEFAULT,
        }

        # Load ResNet with weights if specified
        if pretrained:
            self.resnet = getattr(models, model_type)(weights=weights_dict[model_type])
        else:
            self.resnet = getattr(models, model_type)(weights=None)

        # Modify first convolution layer to accept (1, 12, time) input instead of RGB (3, H, W)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(12, 7), stride=(1, 2), padding=(0, 3), bias=False)

        # If using pretrained weights, adapt the first layer
        if pretrained:
            with torch.no_grad():
                pretrained_weights = self.resnet.conv1.weight.mean(dim=1, keepdim=True)  # Convert 3-channel to 1-channel
                self.resnet.conv1.weight = nn.Parameter(pretrained_weights)

        # Remove global average pooling (to preserve temporal structure)
        self.resnet.avgpool = nn.Identity()

        self.feature_dim, self.time_dim = self._get_feature_map_size(model_type)
        self.dropout = nn.Dropout(dropout)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.feature_dim * self.time_dim, num_classes)

    def _get_feature_map_size(self, model_type):
        feature_dims = {
            "resnet18": 512,
            "resnet34": 512,
            "resnet50": 2048,
            "resnet101": 2048,
            "resnet152": 2048,
        }
        time_dims = {
            "resnet18": 32,
            "resnet34": 32,
            "resnet50": 32,
            "resnet101": 32,
            "resnet152": 32,
        }
        return feature_dims[model_type], time_dims[model_type]

    def forward(self, x):
        x = self.resnet.conv1(x)  
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.dropout(x) 
        x = self.resnet.layer4(x)

        # Output is (Batch, 512, 1, 32), flatten it before FC
        x = self.flatten(x)  # (Batch, 512 * 32)
        x = self.fc(x)  # (Batch, Num_Classes)

        return x

# Explicitly define 2D ResNet variants for direct import
resnet18 = lambda pretrained=True, **kwargs: ResNet2D(model_type="resnet18", pretrained=pretrained, **kwargs)
resnet34 = lambda pretrained=True, **kwargs: ResNet2D(model_type="resnet34", pretrained=pretrained, **kwargs)
resnet50 = lambda pretrained=True, **kwargs: ResNet2D(model_type="resnet50", pretrained=pretrained, **kwargs)
resnet101 = lambda pretrained=True, **kwargs: ResNet2D(model_type="resnet101", pretrained=pretrained, **kwargs)
resnet152 = lambda pretrained=True, **kwargs: ResNet2D(model_type="resnet152", pretrained=pretrained, **kwargs)

# 1D ResNet BasicBlock
class BasicBlock1D(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, kernel_size=3, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
    
class Bottleneck1D(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, kernel_size=3, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet1D Architecture
class ResNet1D(nn.Module):
    def __init__(self, block, layers, num_classes=5, input_channels=12, dropout=0):
        """
        ResNet1D model for ECG classification, adapted from: https://github.com/helme/ecg_ptbxl_benchmarking/tree/master
        """
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 1D ResNet Variants
resnet1d_variants = {
    "resnet1d18": lambda **kwargs: ResNet1D(BasicBlock1D, [2, 2, 2, 2], **kwargs),
    "resnet1d34": lambda **kwargs: ResNet1D(BasicBlock1D, [3, 4, 6, 3], **kwargs),
    "resnet1d50": lambda **kwargs: ResNet1D(Bottleneck1D, [3, 4, 6, 3], **kwargs),
    "resnet1d101": lambda **kwargs: ResNet1D(Bottleneck1D, [3, 4, 23, 3], **kwargs),
    "resnet1d152": lambda **kwargs: ResNet1D(Bottleneck1D, [3, 8, 36, 3], **kwargs),
}

# Explicitly define 1D ResNet variants for direct import
resnet1d18 = resnet1d_variants["resnet1d18"]
resnet1d34 = resnet1d_variants["resnet1d34"]
resnet1d50 = resnet1d_variants["resnet1d50"]
resnet1d101 = resnet1d_variants["resnet1d101"]
resnet1d152 = resnet1d_variants["resnet1d152"]
