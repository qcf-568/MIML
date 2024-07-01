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

dataset_type = 'Dataset'
data_root = './'
imgsize = 512
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

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
    train=casia1,
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
