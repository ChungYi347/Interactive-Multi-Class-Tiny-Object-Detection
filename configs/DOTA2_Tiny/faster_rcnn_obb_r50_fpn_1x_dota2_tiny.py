n_class = 8
mark_size = 1
lr=0.01
weight_decay=0.0001
model_name='FasterRCNNOBBC3Det'
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
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_scales=[8],
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[4, 8, 16, 32, 64],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    bbox_roi_extractor=dict(
        type='SingleRoIExtractor',
        roi_layer=dict(type='RoIAlign', out_size=7, sample_num=2),
        out_channels=256,
        featmap_strides=[4, 8, 16, 32]),
    bbox_head=dict(
        type='SharedFCBBoxHeadRbbox',
        num_fcs=2,
        in_channels=256,
        fc_out_channels=1024,
        roi_feat_size=7,
        num_classes=n_class+1,
        target_means=[0., 0., 0., 0., 0.],
        target_stds=[0.1, 0.1, 0.2, 0.2, 0.1],
        reg_class_agnostic=False,
        with_module=False,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)))

if model_name in ['FasterRCNNOBBLateFusion', 'FasterRCNNOBBC3Det']:
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
        num_outs=5)


# model training and testing settings
train_cfg = dict(
    rpn=dict(
        assigner=dict(
            type='MaxIoUAssignerCy',
            pos_iou_thr=0.7,
            neg_iou_thr=0.3,
            min_pos_iou=0.3,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=256,
            pos_fraction=0.5,
            neg_pos_ub=-1,
            add_gt_as_proposals=False),
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    rpn_proposal=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        assigner=dict(
            type='MaxIoUAssignerCy',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0.5,
            ignore_iof_thr=-1),
        sampler=dict(
            type='RandomSampler',
            num=512,
            pos_fraction=0.25,
            neg_pos_ub=-1,
            add_gt_as_proposals=True),
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=2000,
        nms_post=2000,
        max_num=2000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr = 0.05, nms = dict(type='py_cpu_nms_poly_fast', iou_thr=0.1), max_per_img = 2000)
# soft-nms is also supported for rcnn testing
    # e.g., nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05)
)
# dataset settings
dataset_type = 'DOTA2Dataset_v2_Tiny_UserInput' if model_name not in ['FasterRCNNOBB'] \
                     else 'DOTA2Dataset_v2_Tiny'
data_root = '/path/to/dataset'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train1024/DOTA2_train1024_tiny.json',
        img_prefix=data_root + 'train1024/images/',
        img_scale=(1024, 1024),
        img_norm_cfg=img_norm_cfg,
        size_divisor=32,
        flip_ratio=0.5,
        with_mask=True,
        with_crowd=True,
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
        with_crowd=True,
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
        with_label=False,
        test_mode=True))

if model_name not in ['FasterRCNNOBB']:
    if user_input_loss_enable:
        model['bbox_head']['type'] = 'SharedFCBBoxHeadRbboxUserinput'
        model['bbox_head']['loss_userinput'] = dict(type=user_input_loss, use_sigmoid=False, loss_weight=1.0)
        model['bbox_head']['mark_size'] = mark_size
        for data_type in ['train', 'val', 'test']:
            data[data_type]['mark_size'] = mark_size
    else:
        for data_type in ['train', 'val', 'test']:
            data[data_type]['mark_size'] = mark_size

if model_name in ['FasterRCNNOBBEarlyFusion']:
    model['backbone']['input_channel'] = 3 + n_class

if model_name == "FasterRCNNOBB":
    _total_epochs = 12  
    _step = [8, 11]
else:
    _total_epochs = 24
    _step = [16, 22]

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

val_interval = 4
checkpoint_config = dict(interval=val_interval)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHookVal'),
        dict(type='TensorboardLoggerHookVal')
    ])
# yapf:enable
# runtime settings
total_epochs = _total_epochs
dist_params = dict(backend='nccl')
log_level = 'INFO'
if model_name in ['FasterRCNNOBB']:
    work_dir = f'./work_dirs/faster_rcnn_obb_r50_fpn_1x_dota2_tiny_{model_name}_{lr}_{weight_decay}'
else:
    work_dir = f'./work_dirs/faster_rcnn_obb_r50_fpn_1x_dota2_tiny_{model_name}_{user_input_loss}_{lr}_{weight_decay}'
load_from = None
resume_from = None
workflow = [('train', 1), ('val', 1)]