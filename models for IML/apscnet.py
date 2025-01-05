norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='VizEncoderDecoder',
    pretrained='convnext_ade.pth',
    backbone=dict(
        type='ConvNeXt',
        in_chans=3,
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        out_indices=[0, 1, 2, 3]),
    decode_head=dict(
        type='TestRFB1HA1CLab',
        in_channels=[128, 256, 512, 1024],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        align_corners=False,
        loss_decode=dict(type='OhemCrossEntropy', loss_weight=1.0),),
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

dataset_type = 'Tianchi2022Dataset'
data_root = './'
imgsize = 512
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

tamper_comp = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    # dict(type='CASIA1'),
    dict(type='Resize', img_scale=[(imgsize, imgsize), (int(imgsize*1.5), int(imgsize*1.5))], multiscale_mode='range', keep_ratio=False),
    # dict(type='Resize', ratio_range=(0.5, 4.0), min_size=256),
    dict(type='RandomCrop', crop_size=(imgsize, imgsize), cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(
        type='Albu',
        transforms=[
            dict(type='RandomRotate90', p=0.5),
            dict(
                type='ImageCompression',
                quality_lower=50,
                quality_upper=100,
                p=0.9),
            dict(type='RGBShift', p=0.3),
        ]),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    # dict(type='Pad', size=(512, 512), pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

coco_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeCOCO', img_scale=[(imgsize, imgsize), (int(imgsize*1.5), int(imgsize*1.5))], multiscale_mode='range', keep_ratio=False),
    dict(type='RandomCrop', crop_size=(imgsize, imgsize), cat_max_ratio=1.0),
    dict(type='RandomFlip', prob=0.0),
    dict(type='COCO'),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='RESIZE', img_scale=(512, 512), multiscale_mode='value', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

cm = dict(
    type='TamperDatajpgpng',
    data_root=data_root,
    img_dir='cm/imgs',
    ann_dir='cm/masks',
    pipeline=tamper_comp,
)

bcm = dict(
    type='TamperDatajpgpng',
    data_root=data_root,
    img_dir='bcm/imgs',
    ann_dir='bcm/masks',
    pipeline=tamper_comp,
)

sp = dict(
    type='TamperDatajpgpng',
    data_root=data_root,
    img_dir='sp/imgs',
    ann_dir='sp/masks',
    pipeline=tamper_comp,
)

casia2 = dict(
    type = 'RepeatDataset',
    times = 80,
    dataset = dict(
    type='TamperDatajpgpng',
    data_root=data_root,
    img_dir='normed/CASIA2/imgs',
    ann_dir='normed/CASIA2/masks',
    pipeline=tamper_comp,
    )
)

SPG = dict(
    type = 'RepeatDataset',
    times = 6,
    dataset = dict(
    type='TamperDatajpgpng',
    data_root=data_root,
    img_dir='MIML_Part1/imgs',
    ann_dir='MIML_Part1/masks',
    pipeline=tamper_comp,
    )
)

SDG = dict(
            type = 'RepeatDataset',
                times = 4,
                    dataset = dict(
                            type='TamperDatajpgpng',
                                data_root=data_root,
                                    img_dir='MIML_Part2/imgs',
                                        ann_dir='MIML_Part2/masks',
                                            pipeline=tamper_comp,
                                                )
                    )

coco = dict(
    type = 'RepeatDataset',
    times = 3,
    dataset = dict(
    type='TamperCOCO',
    data_root=data_root,
    img_dir='train2017/',
    ann_dir='train2017/',
    pipeline=coco_pipeline,
))

casia1 = dict(
    type='TamperDatajpgpng',
    data_root=data_root,
    img_dir='casia1/imgs',
    ann_dir='casia1/masks',
    pipeline=test_pipeline,
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=6,
    train=[cm, bcm, sp, SPG, SDG, coco, casia2],
    val=casia1,
    test=casia1,
)

log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook', by_epoch=False)])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'convnext_ade.pth'
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True

optimizer = dict(
    constructor='LearningRateDecayOptimizerConstructor',
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(decay_rate=0.9, decay_type='stage_wise', num_layers=12))
# optimizer_config = dict(type='Fp16OptimizerHook', loss_scale=512.)
# fp16 = dict()
optimizer_config = dict(type='OptimizerHook')
lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=1e-06,
    by_epoch=False)
runner = dict(type='IterBasedRunner', max_iters=160000)
checkpoint_config = dict(by_epoch=False, interval=2000)
evaluation = dict(interval=2000, metric=['mIoU','mFscore'], pre_eval=True, save_best='F1score')
gpu_ids = range(0, 8)
auto_resume = False
