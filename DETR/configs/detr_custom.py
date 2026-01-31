# Base config
_base_ = 'mmdet::detr/detr_r50_8xb2-150e_coco.py'

# Dataset
data_root = './'  # 根据你的项目路径调整
classes = (
    'Bicycle','Bus','Green','Micro','Motorcycle','Orange','Pedestrian',
    'Pickup','Red','SUV','Sedan','Tank truck','Tow Truck','Trailer truck',
    'Truck','Unknow','Van'
)

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/images/'),
        metainfo=dict(classes=classes)
    )
)

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/images/'),
        metainfo=dict(classes=classes)
    )
)

test_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='test/images/'),
        metainfo=dict(classes=classes)
    )
)

# 使用预训练权重
load_from = 'checkpoints/detr_r50_8xb2-150e_coco_20221023_153551-436d03e8.pth'

# GPU配置，两张5090
optim_wrapper = dict(type='OptimWrapper', optimizer=dict(type='AdamW', lr=1e-4, weight_decay=1e-4))
train_cfg = dict(max_epochs=50, val_interval=1)
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))

# 测试配置
test_evaluator = dict(type='CocoMetric', ann_file='annotations/test.json', metric='bbox')
val_evaluator = dict(type='CocoMetric', ann_file='annotations/val.json', metric='bbox')