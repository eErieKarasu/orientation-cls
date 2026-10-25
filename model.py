import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


def build_model(num_classes=4, pretrained=True):
    """构建 MobileNetV3-Small 方向分类模型。

    Args:
        num_classes: 分类数（默认 4 个方向）
        pretrained: 是否加载 ImageNet 预训练权重

    Returns:
        nn.Module
    """
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)

    # 替换分类头为 4 类输出
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
