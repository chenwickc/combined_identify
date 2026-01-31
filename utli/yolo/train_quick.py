"""
简化版训练脚本 - 快速测试
"""

from ultralytics import YOLO
import torch

# 检查GPU
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print()

# 加载模型
model = YOLO('yolo26x.pt')

# 训练
results = model.train(
    data='dataset/dataset.yaml',
    epochs=20,  # 快速测试用20个epoch
    imgsz=640,
    batch=16,
    patience=5,
    device=0,
    project='runs/detect',
    name='train_yolov26x_quick',
    save=True,
    verbose=True,
)

# 评估
print("\n评估模型...")
metrics = model.val(
    data='dataset/dataset.yaml',
    imgsz=640,
    batch=16,
    device=0,
)

# 输出关键指标
print("\n========== 评估结果 ==========")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")
print(f"F-1 Score: {metrics.box.f1:.4f}")
print(f"mAP@0.5: {metrics.box.map50:.4f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
print("==============================")
