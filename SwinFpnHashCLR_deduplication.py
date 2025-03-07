import random
from argparse import ArgumentParser
import numpy as np
import torch
import os
from mmengine.dataset import Compose, default_collate
from mmselfsup.apis import inference_model, init_model
from natsort import natsorted  
import pandas as pd

def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Model config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='The random seed for visualization')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    print('Model loaded.')

    # make random mask reproducible (comment out to make it change)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model.cfg.test_dataloader = dict(
        dataset=dict(pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='Resize', scale=(512, 512), backend='pillow'),
            dict(type='PackSelfSupInputs', meta_keys=['img_path'])
        ]))

    vis_pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)
    output_img_size=vis_pipeline.transforms[1].scale[0]

    initial_prefix = "_h0_"
    image_folder = '/home/duhaohao/PycharmProjects/mmselfsup-main/mmselfsup-main/data/Img2_convert_jpg'
    num=0
    results_df = pd.DataFrame(columns=['Prefix', '32 OTSU Threshold','64 OTSU Threshold','96 OTSU Threshold', '128 OTSU Threshold', 'length 32 remove', 'length 64 remove', 'length 96 remove', 'length 128 remove'])
    while True:
        image_files = [file for file in os.listdir(image_folder) if file.startswith(initial_prefix)]
        hash_distances_32 = []
        hash_distances_64 = []
        hash_distances_96 = []
        hash_distances_128 = []

      
        image_files = natsorted(image_files)

        if not image_files:
            break

        interval=1
        length_i=len(image_files)-interval

        for i in range(0, length_i, 2): #for i in range(0, a, 1),not include a
            img_path_1 = os.path.join(image_folder, image_files[i])
            img_path_2 = os.path.join(image_folder, image_files[i + interval])    #+1 is two;  +2 is three

            data = dict(img_path=img_path_1)
            data = vis_pipeline(data)
            data = default_collate([data])
            hash1 = inference_model(model, img_path_1, type='predict')

            hash1 = hash1.view(-1).tolist()
            n = len(hash1) // 32
            split_a = [hash1[j:j + n] for j in range(0, len(hash1), n)]
            hash1_32 = []
            hash1_64 = []
            hash1_96 = []
            hash1_128 = []
            for my_list in split_a:
                hash1_32.append(my_list[0])
                hash1_64.extend(my_list[1:3])
                hash1_96.extend(my_list[3:6])
                hash1_128.extend(my_list[6:10])

            data = dict(img_path=img_path_2)
            data = vis_pipeline(data)
            data = default_collate([data])
            hash2 = inference_model(model, img_path_2, type='predict')

            hash2 = hash2.view(-1).tolist()
            n = len(hash2) // 32
            split_b = [hash2[j:j + n] for j in range(0, len(hash2), n)]
            hash2_32 = []
            hash2_64 = []
            hash2_96 = []
            hash2_128 = []
            for my_list in split_b:
                hash2_32.append(my_list[0])
                hash2_64.extend(my_list[1:3])
                hash2_96.extend(my_list[3:6])
                hash2_128.extend(my_list[6:10])
            # Calculate Hamming distance between hash1 and hash2
            hamming_distance_32 = 0
            for i in range(len(hash1_32)):
                if hash1_32[i] != hash2_32[i]:
                    hamming_distance_32 += 1

            print(f'Images: {os.path.basename(img_path_1)} and {os.path.basename(img_path_2)}')
            print(f'Hamming 32 distance: {hamming_distance_32}')
            hash_distances_32.append(hamming_distance_32)
            # 使用大津法找到佳阈值
            optimal_32_threshold = otsu_threshold(hash_distances_32)

         # Calculate Hamming distance between hash1 and hash2

            hamming_distance_64 = 0
            for i in range(len(hash1_64)):
                if hash1_64[i] != hash2_64[i]:
                    hamming_distance_64 += 1

            print(f'Images: {os.path.basename(img_path_1)} and {os.path.basename(img_path_2)}')
            print(f'Hamming 64 distance: {hamming_distance_64}')
            hash_distances_64.append(hamming_distance_64)
           
            optimal_64_threshold = otsu_threshold(hash_distances_64)

            hamming_distance_96 = 0
            for i in range(len(hash1_96)):
                if hash1_96[i] != hash2_96[i]:
                    hamming_distance_96 += 1

            # Calculate Hamming distance between hash1 and hash2

            print(f'Images: {os.path.basename(img_path_1)} and {os.path.basename(img_path_2)}')
            print(f'Hamming 96 distance: {hamming_distance_96}')
            hash_distances_96.append(hamming_distance_96)
            
            optimal_96_threshold = otsu_threshold(hash_distances_96)

            hamming_distance_128 = 0
            for i in range(len(hash1_128)):
                if hash1_128[i] != hash2_128[i]:
                    hamming_distance_128 += 1
            # Calculate Hamming distance between hash1 and hash2

            print(f'Images: {os.path.basename(img_path_1)} and {os.path.basename(img_path_2)}')
            print(f'Hamming 128 distance: {hamming_distance_128}')
            hash_distances_128.append(hamming_distance_128)
           
            optimal_128_threshold = otsu_threshold(hash_distances_128)


        dist_32_indices = [i for i, x in enumerate(hash_distances_32) if x <= optimal_32_threshold]
        dist_64_indices = [i for i, x in enumerate(hash_distances_64) if x <= optimal_64_threshold]
        dist_96_indices = [i for i, x in enumerate(hash_distances_96) if x <= optimal_96_threshold]
        dist_128_indices = [i for i, x in enumerate(hash_distances_128) if x <= optimal_128_threshold]


        results_df = results_df._append({
            'Prefix': initial_prefix,
            '32 OTSU Threshold': optimal_32_threshold,
            '64 OTSU Threshold': optimal_64_threshold,
            '96 OTSU Threshold': optimal_96_threshold,
            '128 OTSU Threshold': optimal_128_threshold,
            'length 32 remove': len(dist_32_indices),
            'length 64 remove': len(dist_64_indices),
            'length 96 remove': len(dist_96_indices),
            'length 128 remove' :len(dist_128_indices),

        }, ignore_index=True)
        
        results_df.to_excel('320_no_fpn_no_pretrain_0.1_heng'+'_'+str(output_img_size)+'.xlsx', index=False)

        num = num + len(hash_distances_32)*2
        print(num)
        
        try:
            prefix_number_start = initial_prefix.index("_h") + 2
            prefix_number_end = initial_prefix.index("_", prefix_number_start)
            current_number = int(initial_prefix[prefix_number_start:prefix_number_end])
            new_number = current_number + 7
            initial_prefix = f"{initial_prefix[:prefix_number_start]}{new_number}{initial_prefix[prefix_number_end:]}"
        except ValueError:
            
            break

def otsu_threshold(data):
   
    total_mean = sum(data) / len(data)
    total_var = sum((x - total_mean) ** 2 for x in data) / len(data)

    max_variance = 0
    optimal_threshold = 0

    for threshold in range(1, max(data)):
        class1 = [x for x in data if x <= threshold]
        class2 = [x for x in data if x > threshold]

        if not class1 or not class2:
            continue

        mean1 = sum(class1) / len(class1)
        mean2 = sum(class2) / len(class2)

       
        between_variance = len(class1) * len(class2) * (mean1 - mean2) ** 2 / (len(data) - 1)

        if between_variance > max_variance:
            max_variance = between_variance
            optimal_threshold = threshold

    return optimal_threshold

if __name__ == '__main__':
    main()

