# dataset settings
dataset_type = 'mmcls.CustomDataset'
data_root = '/home/duhaohao/PycharmProjects/mmselfsup-main/mmselfsup-main/data'  #指向dataset(dataset->｛namedirs｝->pics)

view_pipeline = [
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomApply',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        prob=0.8),
    dict(type='RandomGaussianBlur', sigma_min=0.1, sigma_max=2.0, prob=0.5),

]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(256, 256), backend='pillow'),
    dict(type='MultiView', num_views=2, transforms=[view_pipeline]),
    dict(type='PackSelfSupInputs',
         #algorithm_keys=['mask'],
         meta_keys=['img_path'])
]

train_dataloader = dict(
    batch_size=16,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        #ann_file='meta/train.txt',
        #data_prefix=dict(img_path='train/'),
        pipeline=train_pipeline))
