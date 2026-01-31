# YOLOv26x 路况识别训练指南

## 项目概述

本项目使用YOLOv26x模型进行**路况识别**任务，识别摄像头拍摄图片中的关键物体，包括：
- 车辆类型：卡车、SUV、轿车、公交车等
- 交通信号：红绿灯（红、绿、橙）
- 其他物体：行人、摩托车、自行车等

## 数据集信息

**数据集结构：**
```
dataset/
├── dataset.yaml          # 数据集配置文件
├── train/images/         # 训练集图像
├── val/images/           # 验证集图像
└── test/images/          # 测试集图像
```

**识别类别（共17个）：**
```
0: Bicycle (自行车)
1: Bus (公交车)
2: Green (绿灯)
3: Micro (微车)
4: Motorcycle (摩托车)
5: Orange (橙灯)
6: Pedestrian (行人)
7: Pickup (皮卡)
8: Red (红灯)
9: SUV (SUV)
10: Sedan (轿车)
11: Tank truck (油罐车)
12: Tow Truck (拖车)
13: Trailer truck (拖挂车)
14: Truck (卡车)
15: Unknow (未知)
16: Van (面包车)
```

## 环境要求

- Python 3.12.12
- PyTorch 2.10.0 + CUDA 12.6
- ultralytics 8.4.9

## 脚本说明

### 1. `train.py` - 完整训练脚本

**功能：** 完整的训练、评估流程
- 检查GPU环境
- 加载数据集配置
- 训练YOLOv26x模型
- 评估模型性能
- 保存评估指标到JSON

**使用方法：**
```bash
.venv\Scripts\activate
python train.py
```

**参数配置：**
- `epochs`: 训练轮数（默认50）
- `batch_size`: 批处理大小（默认16，可根据显存调整）
- `img_size`: 输入图像大小（默认640）

### 2. `train_quick.py` - 快速测试脚本

**功能：** 快速验证训练流程（适合快速测试）
- 使用较少的epoch (20)
- 快速得到结果
- 用于调试参数

**使用方法：**
```bash
.venv\Scripts\activate
python train_quick.py
```

### 3. `predict.py` - 推理/预测脚本

**功能：** 使用训练好的模型进行预测
- 单图像预测
- 视频预测
- 批量预测

**使用方法：**
```bash
.venv\Scripts\activate
python predict.py
```

## 评估指标说明

### Precision（精确率）
- **定义：** 在所有预测为正的样本中，正确预测的比例
- **公式：** Precision = TP / (TP + FP)
- **含义：** 越高越好，表示误报率低

### Recall（召回率）
- **定义：** 在所有真实正样本中，被正确预测的比例
- **公式：** Recall = TP / (TP + FN)
- **含义：** 越高越好，表示漏检率低

### F-1 Score（F1分数）
- **定义：** Precision和Recall的调和平均数
- **公式：** F1 = 2 × (Precision × Recall) / (Precision + Recall)
- **含义：** 综合评估模型性能，0-1之间

### mAP（平均精确度）
- **mAP@0.5：** IoU阈值为0.5时的平均精确度
- **mAP@0.5:0.95：** IoU从0.5到0.95的平均精确度

## 训练建议

### GPU显存考虑
```
- 显存 < 4GB：batch_size=8, epochs=30
- 显存 4-8GB：batch_size=16, epochs=50
- 显存 > 8GB：batch_size=32, epochs=100
```

### 超参数调整

**学习率：** 默认0.01（可在train.py中修改）
**优化器：** SGD（稳定性好）或Adam（收敛快）
**数据增强：** 默认开启（旋转、缩放、翻转等）

## 输出文件

训练完成后，结果保存在 `runs/detect/train_yolov26x/` 中：

```
runs/detect/train_yolov26x/
├── weights/
│   ├── best.pt          # 最佳模型（推荐使用）
│   └── last.pt          # 最后一个epoch的模型
├── results.csv          # 训练过程指标
├── results.png          # 训练曲线图
└── confusion_matrix.png # 混淆矩阵
```

## 模型使用示例

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('runs/detect/train_yolov26x/weights/best.pt')

# 预测图像
results = model.predict(source='image.jpg', conf=0.5)

# 预测视频
results = model.predict(source='video.mp4', conf=0.5)

# 实时预测（网络摄像头）
results = model.predict(source=0, conf=0.5)
```

## 常见问题

### 显存不足怎么办？
- 减小batch_size（如从16改为8）
- 减小img_size（如从640改为480）
- 启用混合精度训练（fp16=True）

### 模型收敛缓慢怎么办？
- 增加学习率（lr0=0.02）
- 调整优化器为Adam
- 增加数据增强强度

### 推理时速度慢怎么办？
- 减小输入图像大小
- 使用INT8或FP16量化
- 考虑使用更小的模型（yolov8m等）

## 其他资源

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [YOLOv26x 模型卡](https://github.com/ultralytics/ultralytics)

## 注意事项

1. **数据集准备：** 确保dataset文件夹中有images和labels两个子目录
2. **标注格式：** YOLO格式（txt文件，每行一个检测框）
3. **GPU驱动：** 确保CUDA和cuDNN正确安装
4. **显存清理：** 长时间训练后可能需要清空缓存

---

**祝你训练顺利！** 🚀
