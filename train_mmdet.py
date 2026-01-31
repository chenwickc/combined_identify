"""
MMDetection 完整训练脚本 - 基于RTMDet模型
"""

import os
import torch
from mmdet.apis import set_random_seed, train_detector
from mmdet.models import build_detector
from mmengine.config import Config, ConfigDict
from mmengine.runner import Runner
from mmengine.optim.scheduler import LinearLR
import yaml


def create_config_file(output_path='mmdet_config.py'):
    """创建MMDetection配置文件"""
    
    config_content = """
# 模型配置
model = dict(
    type='RTMDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        bgr_to_rgb=False),
    backbone=dict(
        type='CSPResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        num_classes=None,
        with_cp=True,
        style='pytorch'),
    neck=dict(
        type='PAFPN',
        in_channels=[256, 512, 1024],
        out_channels=256,
        num_outs=5),
    bbox_head=dict(
        type='RTMDetHead',
        num_classes=17,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        featmap_strides=[8, 16, 32, 64, 128],
        anchors=((8, 8), (16, 16), (32, 32), (64, 64), (128, 128)),
        share_conv=True,
        share_cls=True,
        share_reg=True,
        strides=[8, 16, 32, 64, 128],
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(
            type='GIoULoss',
            loss_weight=2.0),
        loss_dfl=dict(
            type='DistributionFocalLoss',
            loss_weight=0.25)),
    train_cfg=dict(
        assigner=dict(
            type='DynamicSoftLabelAssigner',
            topk=13),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=30000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(
            type='nms',
            iou_threshold=0.6),
        max_per_img=300))

# 数据集配置
dataset_type = 'CocoDataset'
data_root = 'dataset/'

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomShift',
        prob=0.5,
        max_shift_px=32),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(
        type='Normalize',
        mean=[103.53, 116.28, 123.675],
        std=[57.375, 57.12, 58.395],
        to_rgb=False),
    dict(type='Pad', size_divisor=32),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/images/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=None))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val/annotations.json',
        data_prefix=dict(img='val/images/'),
        pipeline=test_pipeline,
        backend_args=None))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='test/annotations.json',
        data_prefix=dict(img='test/images/'),
        pipeline=test_pipeline,
        backend_args=None))

# 评估配置
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'val/annotations.json',
    metric='bbox',
    format_only=False,
    backend_args=None)

test_evaluator = val_evaluator

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=0.01,
        momentum=0.937,
        weight_decay=0.0005))

# 学习率调度
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=500),
    dict(
        type='CosineAnnealingLR',
        eta_min_ratio=0.05,
        begin=0,
        end=50,
        by_epoch=True,
        convert_to_iter_based=True)
]

# 训练循环配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=50,
    val_interval=10)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 默认配置
default_scope = 'mmdet'
default_hooks = dict(
    timer=dict(type='TimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        save_best='auto',
        max_keep_ckpts=3),
    sampler_seed=dict(type='SamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ 配置文件已创建: {output_path}")
    return output_path


def main():
    """主函数"""
    
    print("=" * 70)
    print("MMDetection RTMDet 模型微调 - 路况识别")
    print("=" * 70)
    print()
    
    # 检查环境
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print()
    
    # 检查数据集
    print("检查数据集...")
    dataset_dir = 'dataset'
    
    for split in ['train', 'val', 'test']:
        json_file = os.path.join(dataset_dir, split, 'annotations.json')
        if not os.path.exists(json_file):
            print(f"⚠️  {split}集的annotations.json不存在")
            print(f"请先运行: python convert_dataset.py")
            return
        else:
            print(f"✅ {split}集: annotations.json 存在")
    print()
    
    # 创建配置文件
    config_file = create_config_file()
    print()
    
    # 加载配置
    print("加载配置...")
    config = Config.fromfile(config_file)
    
    # 设置工作目录
    work_dir = 'runs/mmdet/rtmdet_finetune'
    config.work_dir = work_dir
    
    # 设置随机种子
    set_random_seed(42)
    
    print(f"工作目录: {work_dir}")
    print()
    
    # 创建运行器并训练
    print("=" * 70)
    print("开始训练...")
    print("=" * 70)
    print()
    
    runner = Runner.from_cfg(config)
    runner.train()
    
    print()
    print("=" * 70)
    print("✅ 训练完成！")
    print("=" * 70)
    print()
    print(f"结果保存在: {work_dir}")
    print()
    print("评估结果:")
    print("  - 查看训练日志: logs/")
    print("  - 最佳模型: best_*.pth")
    print("  - 训练中间检查点: epoch_*.pth")


if __name__ == '__main__':
    main()
