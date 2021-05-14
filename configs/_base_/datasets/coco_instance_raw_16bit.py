dataset_type = 'CocoDataset'
data_root = 'data/coco/v1/'
img_norm_cfg = dict(
    mean=[1064.83866493, 2287.40086903, 1304.45303096], std=[ 6854.36373512, 13586.33207643,  6648.28956547], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile16Bit'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='ResizeRAW', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlipRAW', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile16Bit'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='ResizeRAW', keep_ratio=True),
            dict(type='RandomFlipRAW'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
