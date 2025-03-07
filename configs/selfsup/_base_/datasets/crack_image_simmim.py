# dataset settings
dataset_type = 'mmcls.CustomDataset'
data_root = '/home/duhaohao/PycharmProjects/mmselfsup-main/mmselfsup-main/data'  #指向dataset(dataset->｛namedirs｝->pics)
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 256), backend='pillow'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='SimMIMMaskGenerator',
        input_size=256,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6),
    dict(
        type='PackSelfSupInputs',
        algorithm_keys=['mask'],
        meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=16,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),#shuffle将序列的所有元素随机排序
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        #ann_file='meta/train.txt',
        #data_prefix=dict(img_path=''),
        pipeline=train_pipeline))

# for visualization
vis_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 256), backend='pillow'),
    dict(
        type='SimMIMMaskGenerator',
        input_size=256,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6),
    dict(
        type='PackSelfSupInputs',
        algorithm_keys=['mask'],
        meta_keys=['img_path'])
]
