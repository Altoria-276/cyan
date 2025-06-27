import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, num_classes: int = 1000, pretrained: bool = True):
        super().__init__()
        # 初始化基础模型
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)

        # 特征提取部分（包含全局平均池化层）
        self.features = nn.Sequential(*list(self.model.children())[:-1], nn.Flatten())

        # 单独保留全连接层
        self.fc = self.model.fc

        # 适配自定义类别数
        if num_classes != 1000:
            self.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> tuple:
        features = self.features(x)
        outputs = self.fc(features)
        return features, outputs
