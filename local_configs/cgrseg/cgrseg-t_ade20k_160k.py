_base_ = [
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py',
]
norm_cfg = dict(type='SyncBN', requires_grad=True)

EfficientFormer_width = {
    'L': (40, 80, 192, 384),  # 26m 83.3% 6attn
    'S2': (32, 64, 144, 288),  # 12m 81.6% 4attn dp0.02
    'S1': (32, 48, 120, 224),  # 6.1m 79.0
    'S0': (32, 48, 96, 176),  # 75.0 75.7
}

EfficientFormer_depth = {
    'L': (5, 5, 15, 10),  # 26m 83.3%
    'S2': (4, 4, 12, 8),  # 12m
    'S1': (3, 3, 9, 6),  # 79.0
    'S0': (2, 2, 6, 4),  # 75.7
}

EfficientFormer_expansion_ratios = {
    'L': (4, 4, (4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4), (4, 4, 4, 3, 3, 3, 3, 4, 4, 4)),
    'S2': (4, 4, (4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4), (4, 4, 3, 3, 3, 3, 4, 4)),
    'S1': (4, 4, (4, 4, 3, 3, 3, 3, 4, 4, 4), (4, 4, 3, 3, 4, 4)),
    'S0': (4, 4, (4, 3, 3, 3, 4, 4), (4, 3, 3, 4)),
}

model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='efficientformerv2_s1_feat',
        style='pytorch',
        init_cfg=dict(type='Pretrained',checkpoint='checkpoint/eformer_s1_450.pth',),
    ),
    decode_head=dict(
        type='CGRSeg',
        in_channels=[48, 120, 224],
        in_index=[1, 2, 3],
        channels=128,
        dropout_ratio=0.1,
        num_classes=150,
        is_dw=True,
        dw_size=11,
        neck_size=11,
        next_repeat=5,
        square_kernel_size=3,
        ratio=1, 
        module='RCA',
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
    # test_cfg = dict(mode='slide', crop_size=(512, 512), stride=(512, 512))
    )

optimizer = dict(_delete_=True, type='AdamW', lr=0.00012, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=4,
          workers_per_gpu=8)
evaluation = dict(interval=4000)
find_unused_parameters=True

