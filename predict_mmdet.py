"""
MMDetection 推理脚本
使用训练好的模型进行预测
"""

import os
import cv2
import numpy as np
from mmdet.apis import inference_detector, init_detector
from pathlib import Path


def inference_image(config_file, checkpoint_file, image_path, score_thr=0.5, device='cuda:0'):
    """
    对单个图像进行推理
    
    参数:
        config_file: 模型配置文件
        checkpoint_file: 训练好的模型权重文件
        image_path: 输入图像路径
        score_thr: 置信度阈值
        device: 使用的设备 ('cuda:0' 或 'cpu')
    """
    
    print(f"初始化模型...")
    model = init_detector(
        config_file,
        checkpoint_file,
        device=device
    )
    
    print(f"读取图像: {image_path}")
    img = cv2.imread(image_path)
    
    if img is None:
        print(f"错误: 无法读取图像 {image_path}")
        return None
    
    print(f"进行推理...")
    result = inference_detector(model, image_path)
    
    # 获取检测结果
    print(f"\n检测结果:")
    print("-" * 60)
    
    pred_instances = result.pred_instances
    
    # 加载类别名称
    class_names = model.dataset_meta['classes']
    
    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    
    detection_count = 0
    for bbox, score, label in zip(bboxes, scores, labels):
        if score >= score_thr:
            class_name = class_names[label]
            x1, y1, x2, y2 = bbox
            print(f"{class_name:20s} | 置信度: {score:.4f} | 位置: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")
            detection_count += 1
    
    print("-" * 60)
    print(f"总检测数: {detection_count}")
    
    # 可视化结果
    return result, img, model


def visualize_result(result, img, model, output_path='output.jpg', score_thr=0.5):
    """
    可视化检测结果
    """
    
    pred_instances = result.pred_instances
    class_names = model.dataset_meta['classes']
    
    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    labels = pred_instances.labels.cpu().numpy()
    
    # 绘制检测框
    for bbox, score, label in zip(bboxes, scores, labels):
        if score >= score_thr:
            x1, y1, x2, y2 = bbox.astype(int)
            class_name = class_names[label]
            
            # 绘制矩形框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制文本标签
            text = f"{class_name} {score:.2f}"
            cv2.putText(
                img,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
    
    # 保存图像
    cv2.imwrite(output_path, img)
    print(f"\n✅ 结果已保存: {output_path}")
    
    return img


def batch_inference(config_file, checkpoint_file, image_dir, output_dir='outputs', score_thr=0.5, device='cuda:0'):
    """
    批量推理文件夹中的所有图像
    """
    
    print(f"初始化模型...")
    model = init_detector(config_file, checkpoint_file, device=device)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in image_extensions
    ]
    
    print(f"找到 {len(image_files)} 个图像\n")
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        output_path = os.path.join(output_dir, f'result_{image_file}')
        
        print(f"处理: {image_file}")
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"  ⚠️  无法读取图像")
            continue
        
        result = inference_detector(model, image_path)
        visualize_result(result, img.copy(), model, output_path, score_thr)
        print()


if __name__ == '__main__':
    
    print("=" * 70)
    print("MMDetection 推理脚本")
    print("=" * 70)
    print()
    
    # 配置文件和模型权重
    config_file = 'runs/mmdet/rtmdet_finetune/mmdet_config.py'
    checkpoint_file = 'runs/mmdet/rtmdet_finetune/best_*.pth'
    
    # 检查文件是否存在
    if not os.path.exists(config_file):
        print(f"错误: 配置文件不存在 {config_file}")
        print("请先运行: python train_mmdet.py")
        exit(1)
    
    # 查找最佳模型文件
    checkpoint_dir = os.path.dirname(checkpoint_file)
    checkpoint_files = [
        f for f in os.listdir(checkpoint_dir) if f.startswith('best_') and f.endswith('.pth')
    ] if os.path.exists(checkpoint_dir) else []
    
    if not checkpoint_files:
        print(f"错误: 未找到模型权重文件")
        print("请先运行: python train_mmdet.py")
        exit(1)
    
    checkpoint_file = os.path.join(checkpoint_dir, checkpoint_files[0])
    
    print(f"配置文件: {config_file}")
    print(f"模型权重: {checkpoint_file}")
    print()
    
    # 推理测试集
    print("批量推理测试集...")
    test_image_dir = 'dataset/test/images'
    output_dir = 'outputs/mmdet_results'
    
    if os.path.exists(test_image_dir):
        batch_inference(
            config_file,
            checkpoint_file,
            test_image_dir,
            output_dir,
            score_thr=0.5,
            device='cuda:0'  # 改为 'cpu' 如果没有GPU
        )
    else:
        print(f"测试集目录不存在: {test_image_dir}")
