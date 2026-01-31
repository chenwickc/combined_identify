"""
数据集格式转换脚本
将YOLO格式的标注转换为COCO格式（MMDetection所需）
"""

import os
import json
import cv2
from pathlib import Path
from tqdm import tqdm
import yaml


def yolo_to_coco(dataset_dir, output_dir='dataset'):
    """
    将YOLO格式数据集转换为COCO格式
    
    YOLO格式：
    - images/
    - labels/ (txt文件，每行: class_id center_x center_y width height)
    
    COCO格式：
    - annotations.json (包含所有标注信息)
    """
    
    # 加载类别信息
    yaml_path = os.path.join(dataset_dir, 'dataset.yaml')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    class_names = {i: name for i, name in config['names'].items()}
    print(f"类别总数: {len(class_names)}")
    print(f"类别: {list(class_names.values())[:5]}...")
    print()
    
    # 处理训练/验证/测试集
    for split in ['train', 'val', 'test']:
        print(f"\n处理{split}集...")
        print("=" * 60)
        
        images_dir = os.path.join(dataset_dir, split, 'images')
        labels_dir = os.path.join(dataset_dir, split, 'labels')
        
        if not os.path.exists(images_dir):
            print(f"跳过: {images_dir} 不存在")
            continue
        
        # 初始化COCO格式数据结构
        coco_data = {
            'images': [],
            'annotations': [],
            'categories': [
                {'id': int(k), 'name': v}
                for k, v in sorted(class_names.items(), key=lambda x: int(x[0]))
            ]
        }
        
        image_id = 0
        annotation_id = 0
        
        # 获取所有图像文件
        image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
        ])
        
        print(f"找到 {len(image_files)} 个图像")
        
        # 处理每个图像
        for image_file in tqdm(image_files, desc=f"转换{split}集"):
            image_path = os.path.join(images_dir, image_file)
            label_file = image_file.rsplit('.', 1)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            # 读取图像信息
            img = cv2.imread(image_path)
            if img is None:
                print(f"警告: 无法读取图像 {image_path}")
                continue
            
            height, width, _ = img.shape
            
            # 添加图像信息
            coco_data['images'].append({
                'id': image_id,
                'file_name': image_file,
                'height': height,
                'width': width
            })
            
            # 读取标注
            if os.path.exists(label_path):
                with open(label_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        
                        class_id = int(parts[0])
                        center_x = float(parts[1]) * width
                        center_y = float(parts[2]) * height
                        box_width = float(parts[3]) * width
                        box_height = float(parts[4]) * height
                        
                        # 转换为COCO格式 (x_min, y_min, width, height)
                        x_min = center_x - box_width / 2
                        y_min = center_y - box_height / 2
                        
                        # 确保坐标在有效范围内
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        box_width = min(box_width, width - x_min)
                        box_height = min(box_height, height - y_min)
                        
                        if box_width <= 0 or box_height <= 0:
                            continue
                        
                        # 添加标注
                        coco_data['annotations'].append({
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': class_id,
                            'bbox': [x_min, y_min, box_width, box_height],
                            'area': box_width * box_height,
                            'iscrowd': 0
                        })
                        
                        annotation_id += 1
            
            image_id += 1
        
        # 保存COCO格式的JSON文件
        output_path = os.path.join(dataset_dir, split, 'annotations.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(coco_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 已保存: {output_path}")
        print(f"   - 图像数: {len(coco_data['images'])}")
        print(f"   - 标注数: {len(coco_data['annotations'])}")


def verify_conversion(dataset_dir):
    """验证转换结果"""
    print("\n" + "=" * 60)
    print("验证转换结果...")
    print("=" * 60)
    
    for split in ['train', 'val', 'test']:
        json_path = os.path.join(dataset_dir, split, 'annotations.json')
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"\n{split.upper()}集:")
            print(f"  ✅ 图像数: {len(data['images'])}")
            print(f"  ✅ 标注数: {len(data['annotations'])}")
            print(f"  ✅ 类别数: {len(data['categories'])}")


if __name__ == '__main__':
    print("=" * 60)
    print("YOLO → COCO 数据集格式转换")
    print("=" * 60)
    print()
    
    dataset_dir = 'dataset'
    
    # 转换数据集
    yolo_to_coco(dataset_dir)
    
    # 验证转换
    verify_conversion(dataset_dir)
    
    print("\n" + "=" * 60)
    print("✅ 转换完成！")
    print("=" * 60)
    print("\n现在可以使用以下命令训练模型:")
    print("  python mmdet_finetune.py")
