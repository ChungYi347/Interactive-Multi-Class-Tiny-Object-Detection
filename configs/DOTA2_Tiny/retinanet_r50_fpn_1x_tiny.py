n_class = 8
mark_size = 1
lr=0.01
weight_decay=0.0001
# Retina: RetinaNetRbbox, Others: RetinaNetOBB ...
model_name='RetinaNetOBBC3Det'
user_input_loss_enable=False
user_input_loss='CrossEntropyLoss'

# model settings
model = dict(
    type=model_name,
    pretrained='modelzoo://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    rbbox_head=dict(
        type='RetinaHeadRbbox',
        num_classes=n_class+1,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))

if model_name in ['RetinaNetOBBLateFusion', 'RetinaNetOBBC3Det']:
    model['userinput_backbone'] = dict(
        type='ResNet',
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        input_channel=n_class,
        style='pytorch'
    )
    model['userinput_neck'] = dict(
        type='FPN',
        in_channels=[64, 128, 256, 512],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5)


# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssignerCy',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='py_cpu_nms_poly_fast', iou_thr=0.5),
    max_per_img = 100)
# dataset settings
dataset_type = 'DOTA2Dataset_v2_Tiny_UserInput' if model_name not in ['RetinaNetRbbox'] \
                     else 'DOTA2Dataset_v2_Tiny'
data_root = '/path/to/dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train1024/DOTA2_train1024_tiny.json',
        img_prefix=data_root + 'train1024/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=False,
        with_label=True),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val1024/DOTA2_val1024_tiny.json',
        img_prefix=data_root + 'val1024/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=True,
        with_crowd=False,
        with_label=True),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test1024/DOTA2_test1024_tiny.json',
        img_prefix=data_root + 'test1024/images',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0,
        with_mask=False,
        with_crowd=False,
        with_label=False,
        test_mode=True))

if model_name not in ['RetinaNetRbbox']:
    if user_input_loss_enable:
        model['bbox_head']['type'] = 'RetinaHeadRbboxUserinput'
        model['bbox_head']['loss_userinput'] = dict(type=user_input_loss, use_sigmoid=False, loss_weight=1.0)
        model['bbox_head']['mark_size'] = mark_size
        for data_type in ['train', 'val', 'test']:
            data[data_type]['mark_size'] = mark_size
    else:
        for data_type in ['train', 'val', 'test']:
            data[data_type]['mark_size'] = mark_size

if model_name in ['RetinaNetOBBEarlyFusion']:
    model['backbone']['input_channel'] = 3 + n_class

if model_name == "RetinaNetRbbox":
    _total_epochs = 24 
    _step = [16, 22]
else:
    _total_epochs = 36
    _step = [24, 33]

# optimizer
optimizer = dict(type='SGD', lr=float(lr), momentum=0.9, weight_decay=float(weight_decay))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=_step)
val_interval = 1
checkpoint_config = dict(interval=val_interval)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHookVal')
    ])
# yapf:enable
# runtime settings
total_epochs = _total_epochs
dist_params = dict(backend='nccl')
log_level = 'INFO'
if model_name in ['RetinaNetRbbox']:
    work_dir = f'./work_dirs/faster_rcnn_obb_r50_fpn_1x_dota2_tiny_{model_name}_{lr}_{weight_decay}'
else:
    work_dir = f'./work_dirs/faster_rcnn_obb_r50_fpn_1x_dota2_tiny_{model_name}_{user_input_loss}_{lr}_{weight_decay}'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]