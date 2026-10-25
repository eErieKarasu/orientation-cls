"""训练脚本。

使用 MobileNetV3-Small 在自动生成的数据集上训练方向分类模型。
"""

import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from model import build_model
from utils import get_device, get_transform

DATA_DIR = Path(__file__).parent / "data" / "processed"
SAVE_PATH = Path(__file__).parent / "best_model.pth"

# 超参数
EPOCHS = 30
LR = 1e-3
BATCH_SIZE = 32


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def main():
    device = get_device()
    print(f"使用设备: {device}")

    # 数据加载
    train_ds = ImageFolder(str(DATA_DIR / "train"), transform=get_transform(training=True))
    val_ds = ImageFolder(str(DATA_DIR / "val"), transform=get_transform(training=False))

    print(f"训练集: {len(train_ds)} 张 | 验证集: {len(val_ds)} 张")
    print(f"类别映射: {train_ds.class_to_idx}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 模型
    model = build_model(num_classes=4, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    print(f"\n开始训练 (epochs={EPOCHS}, lr={LR}, batch_size={BATCH_SIZE})")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        elapsed = time.time() - t0

        marker = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            marker = " *best*"

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"{elapsed:.1f}s{marker}"
        )

    print("-" * 60)
    print(f"训练完成 | 最佳验证准确率: {best_acc:.4f}")
    print(f"模型已保存至: {SAVE_PATH}")


if __name__ == "__main__":
    main()
