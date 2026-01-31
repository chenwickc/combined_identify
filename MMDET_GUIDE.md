# MMDetection 目标检测指南

## 📌 概述

**MMDetection** 是一个开源的目标检测工具箱，包含50+个预训练模型，可用于各种检测任务。本项目使用MMDetection进行路况识别的微调训练。

## 🧠 MMDetection 内置模型

MMDetection内置了多个高性能的目标检测模型：

| 模型 | 架构 | 速度 | 精度 | 说明 |
|------|------|------|------|------|
| **RTMDet** | 单阶段 | ⚡⚡⚡ 快 | ⭐⭐⭐⭐⭐ 高 | **推荐**，平衡性最好 |
| YOLOv3 | 单阶段 | ⚡⚡ 中等 | ⭐⭐⭐ 中等 | 经典模型，稳定可靠 |
| Faster R-CNN | 两阶段 | ⚡ 慢 | ⭐⭐⭐⭐ 较高 | 精度高，推理时间长 |
| RetinaNet | 单阶段 | ⚡⚡ 中等 | ⭐⭐⭐⭐ 较高 | 焦点损失函数 |
| EfficientDet | 单阶段 | ⚡⚡⚡ 快 | ⭐⭐⭐⭐ 较高 | 轻量级模型 |

### 为什么选择RTMDet？

- ✅ **速度快**：实时检测能力，适合边缘设备
- ✅ **精度高**：在多个基准数据集上领先
- ✅ **训练快**：收敛速度快，需要较少epoch
- ✅ **内存低**：显存需求小，适合有限资源
- ✅ **易部署**：支持TensorRT、ONNX等多种格式

## 📁 项目结构

```
combined_identify/
├── dataset/                          # 数据集（YOLO格式）
│   ├── dataset.yaml                 # 数据集配置
│   ├── train/
│   │   ├── images/
│   │   ├── labels/
│   │   └── annotations.json         # 转换后的COCO格式
│   ├── val/
│   │   └── ...
│   └── test/
│       └── ...
│
├── convert_dataset.py               # YOLO → COCO格式转换脚本
├── mmdet_finetune.py                # 配置生成脚本
├── train_mmdet.py                   # 训练脚本（主程序）
├── predict_mmdet.py                 # 推理脚本
└── runs/
    └── mmdet/
        └── rtmdet_finetune/         # 训练结果目录
            ├── best_*.pth           # 最佳模型
            ├── mmdet_config.py      # 配置文件
            └── logs/                # 训练日志
```

## 🚀 快速开始

### 第1步：转换数据集格式

MMDetection需要COCO格式的标注文件，先转换你的YOLO格式数据集：

```bash
.venv\Scripts\activate
python convert_dataset.py
```

**输出：**
```
train集:
  ✅ 图像数: XXX
  ✅ 标注数: XXX
  ✅ 类别数: 17
```

### 第2步：训练模型

使用RTMDet模型进行微调：

```bash
python train_mmdet.py
```

**训练参数：**
- 模型：RTMDet (backbone: CSPResNet-50)
- 优化器：SGD
- 学习率：0.01（余弦退火调度）
- 批大小：16
- 训练轮数：50 epoch
- 显存需求：约6GB

**训练时间估计：**
- GPU显存 8GB：~2-3小时
- GPU显存 12GB：~1-2小时
- CPU：~24小时（不推荐）

### 第3步：推理和评估

使用训练好的模型进行预测：

```bash
python predict_mmdet.py
```

**输出：**
```
检测结果:
---
Truck                | 置信度: 0.9234 | 位置: (120, 150, 400, 350)
Bicycle              | 置信度: 0.8765 | 位置: (50, 200, 150, 450)
...
总检测数: 25
```

## 📊 模型评估指标

MMDetection自动计算以下指标：

### 基本指标
- **Precision@0.5** - 置信度阈值0.5的精确度
- **Recall@0.5** - 置信度阈值0.5的召回率
- **F1 Score** - Precision和Recall的调和平均

### 标准COCO指标
- **AP (Average Precision)** - 平均精确度
- **AP@0.5** - IoU=0.5时的AP（严格模式）
- **AP@0.75** - IoU=0.75时的AP（更严格）
- **AP@0.5:0.95** - IoU从0.5到0.95的平均AP（最严格）
- **AR (Average Recall)** - 平均召回率

### 按类别的指标
每个类别单独计算上述指标，便于分析模型在不同物体上的表现

## 🔧 自定义训练

### 修改训练参数

编辑 `train_mmdet.py` 中的配置：

```python
# 训练轮数
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=100,  # 增加epoch数
    val_interval=10
)

# 学习率
optim_wrapper = dict(
    optimizer=dict(
        type='SGD',
        lr=0.02,  # 提高学习率
        momentum=0.937,
        weight_decay=0.0005
    )
)

# 批大小（根据显存调整）
train_dataloader = dict(
    batch_size=32,  # 增加批大小
    num_workers=4,
    ...
)
```

### 使用其他模型

修改配置文件中的 `model` 部分：

```python
# 使用YOLOv3
model = dict(
    type='YOLOV3',
    backbone=dict(type='Darknet', depth=53, ...),
    ...
)

# 使用Faster R-CNN
model = dict(
    type='FasterRCNN',
    backbone=dict(type='ResNet', depth=50, ...),
    ...
)
```

## 📈 监控训练过程

训练日志保存在 `runs/mmdet/rtmdet_finetune/` 中：

```
├── 20250131_120000/
│   ├── vis_data/
│   │   ├── scalars.json       # 训练指标
│   │   └── image/             # 可视化结果
│   └── events.out.tfevents   # TensorBoard事件
├── best_bbox_mAP_epoch_XX.pth # 最佳模型
└── epoch_XX.pth               # 中间检查点
```

## 💡 常见问题

### Q1: 显存不足怎么办？

```python
# 减小批大小
batch_size = 8  # 从16改为8

# 减小输入图像大小
dict(type='Resize', scale=(480, 480), keep_ratio=True)  # 从640改为480

# 启用梯度累积
accumulative_counts = 2
```

### Q2: 模型收敛缓慢？

```python
# 增加学习率
lr = 0.02  # 从0.01改为0.02

# 增加数据增强
dict(type='RandomShift', prob=0.8, max_shift_px=64)
dict(type='RandomRotate', prob=0.5, degree=15)

# 增加训练轮数
max_epochs = 100
```

### Q3: 推理速度太慢？

```python
# 使用更小的输入图像
scale=(480, 480)

# 量化模型
# 使用MMDeploy进行INT8量化或蒸馏

# 增加NMS预检查
nms_pre=5000  # 从30000改为5000
```

### Q4: 如何部署模型？

```bash
# 1. 导出为ONNX格式
python -m mim export mmdet checkpoint.pth \
  --cfg mmdet_config.py \
  --opset-version 11

# 2. 使用MMDeploy进行优化部署
# 3. 转换为TensorRT或其他格式
```

## 📚 相关资源

- [MMDetection官方文档](https://mmdetection.readthedocs.io/)
- [RTMDet论文](https://arxiv.org/abs/2212.07784)
- [模型库](https://github.com/open-mmlab/mmdetection/tree/main/configs)
- [数据集格式](https://mmdetection.readthedocs.io/en/latest/user_guides/dataset_prepare.html)

## 🔍 对比YOLOv26x和MMDetection

| 特性 | YOLOv26x | MMDetection |
|------|---------|-------------|
| 框架 | ultralytics | OpenMMLab |
| 易用性 | ⭐⭐⭐⭐⭐ 最简单 | ⭐⭐⭐⭐ 中等 |
| 灵活性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ 高 |
| 模型选择 | 有限 | ⭐⭐⭐⭐⭐ 丰富 |
| 推理速度 | 快 | 快 |
| 精度 | 高 | 高 |
| 社区支持 | 大 | 大 |
| 部署工具 | 完善 | 完善 |

## ✅ 检查清单

在开始训练前，检查以下事项：

- [ ] 虚拟环境激活
- [ ] MMDetection已安装
- [ ] CUDA和PyTorch版本兼容
- [ ] 数据集已转换为COCO格式
- [ ] 数据集文件夹结构正确
- [ ] GPU显存足够（至少6GB）

---

**祝你训练顺利！** 🚀
