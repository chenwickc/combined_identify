"""
推理脚本 - 使用训练后的模型进行预测
"""

import os
from ultralytics import YOLO
from pathlib import Path

def predict_image(model_path, image_path, conf=0.5):
    """
    对单个图像进行预测
    """
    model = YOLO(model_path)
    
    results = model.predict(
        source=image_path,
        conf=conf,
        iou=0.5,
        device=0,
        verbose=True,
    )
    
    # 显示检测结果
    for result in results:
        print(f"\n检测到的对象:")
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            print(f"  - {class_name}: {confidence:.4f}")


def predict_video(model_path, video_path, conf=0.5, output_path='output.mp4'):
    """
    对视频进行预测
    """
    model = YOLO(model_path)
    
    results = model.predict(
        source=video_path,
        conf=conf,
        iou=0.5,
        device=0,
        save=True,
        verbose=True,
    )


def batch_predict(model_path, image_dir, conf=0.5):
    """
    批量预测文件夹中的所有图像
    """
    model = YOLO(model_path)
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in image_extensions
    ]
    
    print(f"找到 {len(image_files)} 个图像文件\n")
    
    for image_file in image_files:
        image_path = os.path.join(image_dir, image_file)
        print(f"处理: {image_file}")
        predict_image(model_path, image_path, conf)


if __name__ == '__main__':
    # 使用训练好的模型
    model_path = 'runs/detect/train_yolov26x/weights/best.pt'
    
    # 检查模型是否存在
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在: {model_path}")
        print("请先运行 train.py 或 train_quick.py 来训练模型")
        exit(1)
    
    # 示例：预测测试集中的图像
    test_image_dir = 'dataset/test/images'
    
    if os.path.exists(test_image_dir):
        print("批量预测测试集图像...")
        batch_predict(model_path, test_image_dir, conf=0.5)
    else:
        print(f"测试图像目录不存在: {test_image_dir}")
