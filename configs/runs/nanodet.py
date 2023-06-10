model = dict(
    type='SingleStageDetector',
    backbone=dict(
        type='ShuffleNetV2',
        model_size='0.5x',
        out_stages=[2, 3, 4],
        activation='LeakyReLU'),
    neck=dict(
        type='MMENanoDetPAN',
        in_channels=[48, 96, 192],
        out_channels=96,
        start_level=0,
        num_outs=3),
    bbox_head=dict(
        type='NanoDetHead',
        num_classes=7,
        in_channels=96,
        stacked_convs=2,
        feat_channels=96,
        prior_box_scale=5,
        strides=[8, 16, 32],
        act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        loss_dfl=dict(
            type='DistributionFocalLoss', loss_weight=0.25, reg_max=7)),
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))

### Dataset ###

# Modify dataset type and path
dataset_type = 'DACDataset'
data_root = '/home/melina/gpu_starter_2023/data/dac/train/'

val_dataloader = dict(
    persistent_workers=True,
    drop_last=False)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomAffine',
        max_rotate_degree=0,
        max_translate_ratio=0.2,
        scaling_ratio_range=(0.5, 1.5),
        max_shear_degree=0),
    dict(type='Resize', img_scale=(640, 352), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        ),
        #file_client_args=dict(
        #    backend='memcached',
        #    server_list_cfg=
        #    '/mnt/lustre/share/memcached_client/server_list.conf',
        #    client_cfg='/mnt/lustre/share/memcached_client/client.conf')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 352),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root +
        'debug.txt',
        img_prefix=data_root + 
        'JPEGImages',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root +
        'debug.txt',
        img_prefix=data_root + 
        'JPEGImages',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root +
        'debug.txt',
        img_prefix=data_root +
        'JPEGImages',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='mAP')
#evaluation = dict(interval=1, metric='bbox')

optimizer = dict(type='SGD', lr=0.56, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1e-05,
    step=[13, 16, 17])
runner = dict(type='EpochBasedRunner', max_epochs=18)
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl', port=25968)
find_unused_parameters = True
log_level = 'INFO'
load_from = '/home/melina/gpu_starter_2023/nanodet.pth'
resume_from = None
workflow = [('train', 1)]
work_dir = 'workdir/nanodet_0.5x/0.5x_warp'
gpu_ids = range(1)
seed = 0
device = 'cuda'
