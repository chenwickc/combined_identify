"""
MMDetection 微调脚本 - 路况识别
基于MMDetection内置的预训练模型进行微调
"""

import os
import json
from pathlib import Path
from mmdet.apis import train_detector
from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner
import torch
import yaml


def create_mmdet_config(dataset_yaml_path, num_classes=17, model_type='rtmdet'):
    """
    创建MMDetection配置
    
    参数:
        dataset_yaml_path: YOLO格式的数据集配置文件
        num_classes: 类别数量
        model_type: 模型类型 ('rtmdet', 'yolov3', 'fasterrcnn' 等)
    """
    
    # 加载YOLO格式的数据集配置
    with open(dataset_yaml_path, 'r', encoding='utf-8') as f:
        yolo_config = yaml.safe_load(f)
    
    # 获取数据路径（相对于数据集目录）
    dataset_dir = os.path.dirname(dataset_yaml_path)
    train_path = os.path.join(dataset_dir, yolo_config['train'].lstrip('./'))
    val_path = os.path.join(dataset_dir, yolo_config['val'].lstrip('./'))
    test_path = os.path.join(dataset_dir, yolo_config['test'].lstrip('./'))
    
    # 将YOLO的类别映射转换为列表
    class_names = [yolo_config['names'][str(i)] for i in range(num_classes)]
    
    print(f"数据集配置信息:")
    print(f"  训练集路径: {train_path}")
    print(f"  验证集路径: {val_path}")
    print(f"  测试集路径: {test_path}")
    print(f"  类别数: {num_classes}")
    print(f"  类别: {class_names[:5]}... (共{len(class_names)}类)")
    print()
    
    # 选择模型架构
    if model_type == 'rtmdet':
        model_config = dict(
            type='RTMDet',
            backbone=dict(
                type='CSPResNet',
                depth=101,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                num_classes=None,
                with_cp=True,
                style='pytorch'
            ),
            neck=dict(
                type='PAFPN',
                in_channels=[256, 512, 1024],
                out_channels=256,
                num_outs=5
            ),
            bbox_head=dict(
                type='RTMDetHead',
                num_classes=num_classes,
                in_channels=256,
                feat_channels=256,
                stacked_convs=2,
                featmap_strides=[8, 16, 32, 64, 128],
                anchors=((8, 8), (16, 16), (32, 32), (64, 64), (128, 128)),
                share_conv=True,
                share_cls=True,
                share_reg=True,
                strides=[8, 16, 32, 64, 128],
                loss_cls=dict(type='QualityFocalLoss', use_sigmoid=True, beta=2.0, loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
                loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25)
            ),
            train_cfg=dict(
                assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
                allowed_border=-1,
                pos_weight=-1,
                debug=False
            ),
            test_cfg=dict(
                nms_pre=30000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.6),
                max_per_img=300
            )
        )
    elif model_type == 'yolov3':
        model_config = dict(
            type='YOLOV3',
            backbone=dict(
                type='Darknet',
                depth=53,
                num_stages=5,
                out_indices=(2, 3, 4),
                frozen_stages=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=True,
                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
            ),
            neck=dict(
                type='YOLOV3Neck',
                num_scales=3,
                in_channels=[1024, 512, 256],
                out_channels=[512, 256, 128]
            ),
            bbox_head=dict(
                type='YOLOV3Head',
                num_classes=num_classes,
                in_channels=[512, 256, 128],
                out_channels=[1024, 512, 256],
                anchor_generator=dict(
                    type='YOLOAnchorGenerator',
                    base_sizes=[[(116, 90), (156, 198), (373, 326)],
                                 [(30, 61), (62, 45), (59, 119)],
                                 [(10, 13), (16, 30), (33, 23)]],
                    strides=[32, 16, 8]
                ),
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_conf=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_xy=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=2.0),
                loss_wh=dict(type='MSELoss', loss_weight=2.0)
            ),
            train_cfg=dict(
                assigner=dict(
                    type='GridAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0
                )
            ),
            test_cfg=dict(
                nms_pre=1000,
                min_bbox_size=0,
                score_thr=0.05,
                nms=dict(type='nms', iou_threshold=0.45),
                max_per_img=100
            )
        )
    else:  # fasterrcnn
        model_config = dict(
            type='FasterRCNN',
            backbone=dict(
                type='ResNet',
                depth=50,
                num_stages=4,
                out_indices=(0, 1, 2, 3),
                frozen_stages=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                norm_eval=True,
                style='pytorch',
                init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
            ),
            neck=dict(
                type='FPN',
                in_channels=[256, 512, 1024, 2048],
                out_channels=256,
                num_outs=5
            ),
            rpn_head=dict(
                type='RPNHead',
                in_channels=256,
                feat_channels=256,
                anchor_generator=dict(
                    type='AnchorGenerator',
                    scales=[8],
                    ratios=[0.5, 1.0, 2.0],
                    strides=[4, 8, 16, 32, 64]
                ),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[.0, .0, .0, .0],
                    target_stds=[1.0, 1.0, 1.0, 1.0]
                ),
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)
            ),
            roi_head=dict(
                type='StandardRoIHead',
                bbox_roi_extractor=dict(
                    type='SingleRoIExtractor',
                    roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                    out_channels=256,
                    featmap_strides=[4, 8, 16, 32]
                ),
                bbox_head=dict(
                    type='Shared2FCBBoxHead',
                    in_channels=256,
                    fc_out_channels=1024,
                    roi_feat_size=7,
                    num_classes=num_classes,
                    bbox_coder=dict(
                        type='DeltaXYWHBBoxCoder',
                        target_means=[0., 0., 0., 0.],
                        target_stds=[0.1, 0.1, 0.2, 0.2]
                    ),
                    reg_class_agnostic=False,
                    loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                    loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)
                )
            ),
            train_cfg=dict(
                rpn=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.7,
                        neg_iou_thr=0.3,
                        min_pos_iou=0.3,
                        match_low_quality=True,
                        ignore_iof_thr=-1
                    ),
                    sampler=dict(
                        type='RandomSampler',
                        num=256,
                        pos_fraction=0.5,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=False
                    ),
                    allowed_border=-1,
                    pos_weight=-1,
                    debug=False
                ),
                rcnn=dict(
                    assigner=dict(
                        type='MaxIoUAssigner',
                        pos_iou_thr=0.5,
                        neg_iou_thr=0.5,
                        min_pos_iou=0.5,
                        match_low_quality=False,
                        ignore_iof_thr=-1
                    ),
                    sampler=dict(
                        type='RandomSampler',
                        num=512,
                        pos_fraction=0.25,
                        neg_pos_ub=-1,
                        add_gt_as_proposals=True
                    ),
                    pos_weight=-1,
                    debug=False
                )
            ),
            test_cfg=dict(
                rpn=dict(
                    nms_pre=1000,
                    max_per_img=1000,
                    nms=dict(type='nms', iou_threshold=0.7),
                    min_bbox_size=0
                ),
                rcnn=dict(
                    score_thr=0.05,
                    nms=dict(type='nms', iou_threshold=0.5),
                    max_per_img=100
                )
            )
        )
    
    return model_config, class_names


def create_dataset_config(train_path, val_path, test_path, class_names, batch_size=16):
    """创建数据集配置"""
    dataset_config = dict(
        train=dict(
            type='CocoDataset',
            data_root='',
            ann_file=os.path.join(train_path, '../annotations.json'),
            img_prefix=train_path,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Resize',
                    img_scale=(640, 640),
                    keep_ratio=True
                ),
                dict(
                    type='RandomFlip',
                    flip_ratio=0.5
                ),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True
                ),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
            ]
        ),
        val=dict(
            type='CocoDataset',
            data_root='',
            ann_file=os.path.join(val_path, '../annotations.json'),
            img_prefix=val_path,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(640, 640),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True
                        ),
                        dict(type='Pad', size_divisor=32),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]
                )
            ]
        ),
        test=dict(
            type='CocoDataset',
            data_root='',
            ann_file=os.path.join(test_path, '../annotations.json'),
            img_prefix=test_path,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(
                    type='MultiScaleFlipAug',
                    img_scale=(640, 640),
                    flip=False,
                    transforms=[
                        dict(type='Resize', keep_ratio=True),
                        dict(
                            type='Normalize',
                            mean=[123.675, 116.28, 103.53],
                            std=[58.395, 57.12, 57.375],
                            to_rgb=True
                        ),
                        dict(type='Pad', size_divisor=32),
                        dict(type='ImageToTensor', keys=['img']),
                        dict(type='Collect', keys=['img'])
                    ]
                )
            ]
        )
    )
    
    return dataset_config


def main():
    """主函数"""
    
    print("=" * 70)
    print("MMDetection 路况识别 - 模型微调")
    print("=" * 70)
    print()
    
    # 检查GPU
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print()
    
    # 配置参数
    dataset_yaml_path = 'dataset/dataset.yaml'
    num_classes = 17
    model_type = 'rtmdet'  # 可选: 'rtmdet', 'yolov3', 'fasterrcnn'
    epochs = 50
    batch_size = 16
    
    print(f"选定模型: {model_type.upper()}")
    print(f"训练轮数: {epochs}")
    print(f"批大小: {batch_size}")
    print()
    
    # 创建模型配置
    model_config, class_names = create_mmdet_config(
        dataset_yaml_path,
        num_classes=num_classes,
        model_type=model_type
    )
    
    print("=" * 70)
    print("模型配置创建成功！")
    print("=" * 70)
    print()
    print("💡 提示：")
    print("  1. MMDetection内置了多个预训练模型")
    print("  2. 当前配置基于标准模型架构")
    print("  3. 需要将YOLO格式数据集转换为COCO格式")
    print("  4. 建议使用RTMDet模型（快速且精度高）")
    print()
    
    print("✅ 配置准备完成！")
    print()
    print("下一步:")
    print("  1. 转换数据集格式（YOLO → COCO JSON）")
    print("  2. 使用Runner进行训练")
    print("  3. 评估模型性能")


if __name__ == '__main__':
    main()
