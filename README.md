# orientation-cls

基于 PyTorch / MobileNetV3-Small 的表格图片方向分类工具，识别图片相对正向顺时针旋转的 0°、90°、180°、270°，支持模型训练、单张及文件夹预测、ONNX 导出与推理。

## 安装

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

ONNX 导出及推理需要额外依赖：

```bash
python -m pip install numpy onnx onnxruntime
```

## 使用已有模型

仓库包含 `best_model.pth`、`best_model.onnx` 和 `best_model.onnx.data`。保留 ONNX 文件及其配套数据文件在同一目录。

```bash
python predict.py --image /path/to/image.jpg
python predict.py --folder /path/to/images
python predict_onnx.py --image /path/to/image.jpg
python predict_onnx.py --folder /path/to/images
```

预测结果包含文件名、方向角度、置信度、设备或后端及处理耗时。脚本会先打印设备或后端信息，再打印 JSON 结果。PyTorch 自动选择 CUDA、Apple MPS 或 CPU。输出角度表示输入图片的方向；例如预测 90° 时，可逆时针旋转 90° 恢复正向。当前脚本只预测方向，不自动保存回正后的图片。

## 构建数据集和训练

将正向表格图片放入 `data/raw_upright/`：

```bash
mkdir -p data/raw_upright
# 将正向图片复制到上述目录后执行：
python build_dataset.py
python train.py
```

数据构建脚本先按原图划分训练集和验证集（约 8:2），再生成四种旋转版本，避免同一原图的不同方向进入两个集合。重新构建会清空 `data/processed/train` 和 `data/processed/val` 下四个类别目录中的旧数据。

模型使用 ImageNet 预训练权重，首次训练需要下载权重。默认训练 30 个 epoch，学习率为 0.001，batch size 为 32；验证集准确率提升时保存为 `best_model.pth`。

图片保持宽高比缩放并以灰色填充到 224×224，再使用 ImageNet 均值和标准差归一化。训练增强使用颜色扰动，避免旋转或翻转改变方向标签。

## 导出 ONNX

```bash
python export_onnx.py
python export_onnx.py --output /path/to/model.onnx
```

导出使用 opset 17，输入名称为 `image`，形状为 `[batch_size, 3, 224, 224]`，输出名称为 `logits`，支持动态 batch。

## 文件结构

| 文件 | 用途 |
| --- | --- |
| `build_dataset.py` | 生成四方向训练集和验证集 |
| `model.py` | 构建 MobileNetV3-Small 四分类模型 |
| `train.py` | 训练并保存最佳模型 |
| `predict.py` | PyTorch 单张及文件夹推理 |
| `export_onnx.py` | 导出 ONNX 模型 |
| `predict_onnx.py` | ONNX Runtime 推理 |
| `utils.py` | 设备选择、缩放填充及预处理 |

训练图片、生成的数据集、本机配置和虚拟环境不纳入版本控制。仓库未提供模型准确率报告，使用前应通过自己的数据验证效果。
