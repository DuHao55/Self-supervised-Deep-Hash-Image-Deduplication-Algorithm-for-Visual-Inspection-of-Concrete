_base_ = [
    '../_base_/models/simmim_swin-base.py',
    '../_base_/datasets/crack_image_simmim.py',
    '../_base_/schedules/adamw_coslr-200e_in1k.py',
    '../_base_/default_runtime.py',
]

model = dict(
    backbone=dict(
        type='SimMIMSwinTransformer_FPN',
        arch='T',
        img_size=256,
        stage_cfgs=dict(block_cfgs=dict(window_size=6))),
    neck=dict(type='SimMIMNeck', in_channels=96 * 2**3, encoder_stride=32),#input(N,C∗encoder_stride∗encoder_stride,H,W)
                                                                  #PixelShuffle reshape->(N,C,H∗encoder_stride,W∗encoder_stride)
    head=dict(
        type='SimMIMHead',
        patch_size=4,
        loss=dict(type='SimMIMReconstructionLoss', encoder_in_channels=3)))

# optimizer wrapper
optimizer = dict(
    type='AdamW', lr=1e-5, betas=(0.9, 0.999), eps=1e-8)
                                   #现在一个iteration加载的样本数/人家以前一个iteration加载的样本数（样本数等于几个gpu*batch_size)）
# learning rate scheduler
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6 / 2e-4,
        by_epoch=True,
        begin=0,
        end=10,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=790,
        eta_min=1e-6,
        by_epoch=True,
        begin=10,
        end=800,
        convert_to_iter_based=True)
]

# schedule
train_cfg = dict(max_epochs=800)

# runtime
default_hooks = dict(logger=dict(type='LoggerHook', interval=100),
                     checkpoint=dict(type='CheckpointHook', interval=10, max_keep_ckpts=3))  #10,20,30,40,50,60,save 40,50,60
