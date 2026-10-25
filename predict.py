"""推理脚本。

支持单张图片和批量文件夹推理，输出标准 JSON 格式结果。
"""

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from model import build_model
from utils import get_device, get_transform

LABEL_TO_ANGLE = {0: 0, 1: 90, 2: 180, 3: 270}
MODEL_PATH = Path(__file__).parent / "best_model.pth"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def load_model(device):
    model = build_model(num_classes=4, pretrained=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


def predict_single(model, image_path, transform, device):
    img = Image.open(image_path).convert("RGB")

    t0 = time.perf_counter()
    inp = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(1)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    return {
        "filename": Path(image_path).name,
        "angle": LABEL_TO_ANGLE[pred.item()],
        "confidence": round(conf.item(), 4),
        "device_used": str(device),
        "processing_time_ms": round(elapsed_ms, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="表格方向分类推理")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=str, help="单张图片路径")
    group.add_argument("--folder", type=str, help="图片文件夹路径")
    args = parser.parse_args()

    device = get_device()
    print(f"使用设备: {device}", flush=True)

    model = load_model(device)
    transform = get_transform(training=False)

    if args.image:
        result = predict_single(model, args.image, transform, device)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        folder = Path(args.folder)
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not images:
            print(f"[Error] {folder} 下未找到图片")
            return
        results = []
        for img_path in images:
            results.append(predict_single(model, img_path, transform, device))
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
