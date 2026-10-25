"""数据集构建脚本。

读取 data/raw_upright/ 下的正向图片，自动生成 0/90/180/270 度旋转版本，
按 8:2 比例划分到 data/processed/train 和 data/processed/val。
"""

import os
import random
import shutil
from pathlib import Path

from PIL import Image
from tqdm import tqdm

RAW_DIR = Path(__file__).parent / "data" / "raw_upright"
OUT_DIR = Path(__file__).parent / "data" / "processed"
# label -> (文件夹名, 旋转角度)
CLASSES = {0: ("0", 0), 1: ("1", 90), 2: ("2", 180), 3: ("3", 270)}
SPLIT_RATIO = 0.8  # train 占比
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_images(src_dir: Path):
    return [p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS]


def main():
    images = collect_images(RAW_DIR)
    if not images:
        print(f"[Error] {RAW_DIR} 下未找到图片，请先放入正向表格图片。")
        return

    print(f"找到 {len(images)} 张原始图片")

    # 清空旧数据
    for split in ("train", "val"):
        for folder_name, _ in CLASSES.values():
            d = OUT_DIR / split / folder_name
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    # 随机划分
    random.shuffle(images)
    split_idx = int(len(images) * SPLIT_RATIO)
    splits = {
        "train": images[:split_idx],
        "val": images[split_idx:],
    }

    # 确保 val 至少有 1 张
    if len(splits["val"]) == 0 and len(splits["train"]) > 1:
        splits["val"] = [splits["train"].pop()]

    stats = {s: {fn: 0 for fn, _ in CLASSES.values()} for s in ("train", "val")}

    for split_name, file_list in splits.items():
        for img_path in tqdm(file_list, desc=f"生成 {split_name}"):
            img = Image.open(img_path).convert("RGB")
            stem = img_path.stem

            for label, (folder_name, angle) in CLASSES.items():
                # PIL.rotate 逆时针旋转，expand=True 保留完整内容
                rotated = img.rotate(-angle, expand=True) if angle != 0 else img.copy()
                out_name = f"{stem}_r{angle}.jpg"
                out_path = OUT_DIR / split_name / folder_name / out_name
                rotated.save(out_path, quality=95)
                stats[split_name][folder_name] += 1

    # 打印统计
    angle_labels = {"0": "0", "1": "90", "2": "180", "3": "270"}
    print("\n===== 数据集统计 =====")
    for split_name in ("train", "val"):
        total = sum(stats[split_name].values())
        print(f"\n{split_name} ({total} 张):")
        for folder_name, angle_str in angle_labels.items():
            print(f"  类别{folder_name} ({angle_str}°): {stats[split_name][folder_name]} 张")

    print("\n数据集构建完成 ✓")


if __name__ == "__main__":
    main()
