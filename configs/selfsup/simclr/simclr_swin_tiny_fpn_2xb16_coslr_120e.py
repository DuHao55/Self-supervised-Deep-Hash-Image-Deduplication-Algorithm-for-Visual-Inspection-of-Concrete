_base_ = [
    '../_base_/datasets/crack_image_simclr.py',
    '../_base_/schedules/lars_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

# model settings
model = dict(
    type='SimCLR_Hash',
    data_preprocessor=dict(
        mean=(123.675, 116.28, 103.53),
        std=(58.395, 57.12, 57.375),
        bgr_to_rgb=True),
    backbone=dict(
        type='SimCLRSwinTransformer_FPN',
        arch='T',
        img_size=256,
        frozen_parameter=False,#frozen backbone
        stage_cfgs=dict(block_cfgs=dict(window_size=6)),
        # init_cfg=dict(
        # type="Pretrained",
        # checkpoint='/home/znck/PycharmProjects/mmselfsup-main/mmselfsup-main/checkpoints_full/simmim_mask_fpn_original_32_500/backbone.pth'
        #)
),
    neck=dict(
        type='NonLinearNeck',  # SimCLR non-linear neck
        in_channels=768,
        hid_channels=768,
        out_channels=160,#hash value's total bit
        num_layers=2,
        with_avg_pool=True),
    head=dict(
        type='ContrastiveHashHead',
        loss=dict(type='mmcls.CrossEntropyLoss'),
        temperature=0.1,
        beta=1),
)

# optimizer
optimizer = dict(
    type='AdamW', lr=1e-5, betas=(0.9, 0.999), eps=1e-8)
                                   #现在一个iteration加载的样本数/人家以前一个iteration加载的样本数（样本数等于几个gpu*batch_size)）

# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR', T_max=110, by_epoch=True, begin=10, end=120)
]

train_cfg = dict(max_epochs=120)
# runtime settings
default_hooks = dict(
    # 1 means 1 epoch save once,only keeps the latest 3 checkpoints
    checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))
custom_hooks = [dict(type='SetEpochInfoHook')] # my own