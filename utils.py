import torch
from PIL import Image
from torchvision import transforms


def get_device():
    """按优先级检测可用设备: CUDA > MPS > CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def letterbox(image, target_size=224):
    """保持宽高比缩放，灰色填充到正方形。

    Args:
        image: PIL.Image 对象
        target_size: 目标正方形边长

    Returns:
        PIL.Image 正方形图片
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.BILINEAR)

    canvas = Image.new("RGB", (target_size, target_size), (114, 114, 114))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    canvas.paste(image, (paste_x, paste_y))
    return canvas


class LetterboxTransform:
    """可用于 torchvision transforms.Compose 的 letterbox 封装。"""

    def __init__(self, target_size=224):
        self.target_size = target_size

    def __call__(self, image):
        return letterbox(image, self.target_size)


def get_transform(training=False):
    """返回预处理 transforms。

    训练时仅使用 ColorJitter / Normalize 等安全增强，
    **禁止** RandomRotation / RandomFlip，以保证方向标签语义正确。
    """
    steps = [LetterboxTransform(224)]

    if training:
        steps.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2))

    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose(steps)
