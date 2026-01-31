"""
YOLOv26x 路况识别模型训练脚本
目标：训练一个模型来识别摄像头拍摄图片中的关键物体
如卡车、红绿灯、行人、自行车等

考察指标：Precision、Recall、F-1 Score
"""

import os
import json
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml


def check_environment():
    """检查GPU和PyTorch环境"""
    print("=" * 60)
    print("环境检查")
    print("=" * 60)
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU 显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("=" * 60)
    print()


def load_dataset_config(dataset_path):
    """加载数据集配置"""
    config_file = os.path.join(dataset_path, 'dataset.yaml')
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print("数据集配置信息：")
    print(f"  - 类别数: {config['nc']}")
    print(f"  - 类别名称: {config['names']}")
    print()
    
    return config


def train_model(dataset_yaml_path, model_name='yolo26x.pt', epochs=50, batch_size=16, img_size=640):
    """
    训练YOLOv26x模型
    
    参数:
        dataset_yaml_path: 数据集配置文件路径
        model_name: 预训练模型名称
        epochs: 训练轮数
        batch_size: 批大小
        img_size: 输入图像大小
    """
    print("=" * 60)
    print("开始训练 YOLOv26x 模型")
    print("=" * 60)
    print(f"模型: {model_name}")
    print(f"数据集: {dataset_yaml_path}")
    print(f"训练轮数: {epochs}")
    print(f"批大小: {batch_size}")
    print(f"图像大小: {img_size}x{img_size}")
    print("=" * 60)
    print()
    
    # 加载预训练模型
    model = YOLO(model_name)
    
    # 训练模型
    results = model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        patience=10,  # 早停，10个epoch无改进则停止
        device=0,  # 使用GPU设备0
        project='runs/detect',  # 结果保存目录
        name='train_yolov26x',  # 结果文件夹名称
        save=True,  # 保存检查点
        save_period=10,  # 每10个epoch保存一次
        exist_ok=False,  # 不覆盖现有目录
        pretrained=True,  # 使用预训练权重
        optimizer='SGD',  # 优化器：SGD, Adam, etc
        lr0=0.01,  # 初始学习率
        lrf=0.01,  # 最终学习率（初始学习率的比例）
        momentum=0.937,  # SGD动量
        weight_decay=0.0005,  # 权重衰减
        warmup_epochs=3,  # 预热轮数
        warmup_momentum=0.8,  # 预热初始动量
        box=7.5,  # 盒子损失增益
        cls=0.5,  # 分类损失增益
        obj=1.0,  # 目标损失增益
        hsv_h=0.015,  # 图像HSV-H增强
        hsv_s=0.7,  # 图像HSV-S增强
        hsv_v=0.4,  # 图像HSV-V增强
        degrees=10.0,  # 旋转增强（度数）
        translate=0.1,  # 平移增强（比例）
        scale=0.5,  # 缩放增强
        flipud=0.0,  # 垂直翻转概率
        fliplr=0.5,  # 水平翻转概率
        mosaic=1.0,  # 马赛克增强概率
        mixup=0.0,  # Mixup增强概率
        cache='ram',  # 数据缓存：'ram', 'disk', None
        workers=4,  # 数据加载工作进程数
        verbose=True,  # 详细输出
    )
    
    return results


def evaluate_model(model_path, dataset_yaml_path, img_size=640):
    """
    评估模型性能
    
    参数:
        model_path: 训练后的模型路径
        dataset_yaml_path: 数据集配置文件路径
        img_size: 输入图像大小
    """
    print("\n" + "=" * 60)
    print("评估模型性能")
    print("=" * 60)
    print()
    
    # 加载训练后的模型
    model = YOLO(model_path)
    
    # 在测试集上评估
    metrics = model.val(
        data=dataset_yaml_path,
        imgsz=img_size,
        device=0,
        batch=16,
        patience=0,
        verbose=True,
    )
    
    print("\n评估结果摘要：")
    print(f"  - Precision: {metrics.box.mp:.4f}")
    print(f"  - Recall: {metrics.box.mr:.4f}")
    print(f"  - F-1 Score: {metrics.box.f1:.4f}")
    print(f"  - mAP@0.5: {metrics.box.map50:.4f}")
    print(f"  - mAP@0.5:0.95: {metrics.box.map:.4f}")
    print()
    
    return metrics


def predict_and_visualize(model_path, image_path):
    """
    使用训练后的模型进行预测
    
    参数:
        model_path: 训练后的模型路径
        image_path: 图像路径
    """
    print("\n" + "=" * 60)
    print("进行预测")
    print("=" * 60)
    print()
    
    model = YOLO(model_path)
    
    # 预测
    results = model.predict(
        source=image_path,
        conf=0.5,  # 置信度阈值
        iou=0.5,  # IOU阈值
        device=0,
        verbose=True,
    )
    
    return results


def save_metrics_to_json(metrics, output_path='metrics.json'):
    """保存评估指标到JSON文件"""
    metrics_dict = {
        'precision': float(metrics.box.mp),
        'recall': float(metrics.box.mr),
        'f1_score': float(metrics.box.f1),
        'mAP@0.5': float(metrics.box.map50),
        'mAP@0.5:0.95': float(metrics.box.map),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_dict, f, indent=4, ensure_ascii=False)
    
    print(f"指标已保存到: {output_path}")


def main():
    """主函数"""
    
    # 检查环境
    check_environment()
    
    # 数据集配置文件路径
    dataset_yaml_path = 'dataset/dataset.yaml'
    
    # 检查数据集文件是否存在
    if not os.path.exists(dataset_yaml_path):
        print(f"错误：数据集配置文件不存在: {dataset_yaml_path}")
        return
    
    # 加载数据集配置
    config = load_dataset_config('dataset')
    
    # 训练模型
    print("开始训练...")
    results = train_model(
        dataset_yaml_path=dataset_yaml_path,
        model_name='yolo26x.pt',
        epochs=50,  # 可根据需要调整
        batch_size=16,  # 根据显存大小调整
        img_size=640
    )
    
    # 获取最佳模型路径
    best_model_path = os.path.join('runs/detect/train_yolov26x', 'weights', 'best.pt')
    
    # 评估模型
    if os.path.exists(best_model_path):
        print(f"\n最佳模型路径: {best_model_path}")
        metrics = evaluate_model(best_model_path, dataset_yaml_path)
        
        # 保存评估指标
        save_metrics_to_json(metrics)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"结果保存在: runs/detect/train_yolov26x/")
    print()


if __name__ == '__main__':
    main()
