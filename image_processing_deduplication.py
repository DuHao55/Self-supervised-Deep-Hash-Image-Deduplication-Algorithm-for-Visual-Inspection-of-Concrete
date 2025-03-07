# -*- coding: utf-8 -*-
import pandas as pd
import cv2
import time
import os
import numpy as np
from natsort import natsorted  
import imagehash
from PIL import Image

def pHash(img, leng=16, wid=16):
    img = cv2.resize(img, (leng, wid))
    #cv2.imshow("resize",img)
    #cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_roi = dct[0:8, 0:8]
    avreage = np.mean(dct_roi)
    phash_01 = (dct_roi > avreage) + 0
    phash_list = phash_01.reshape(1, -1)[0].tolist()
    hash = ''.join([str(x) for x in phash_list])
    return hash


def dHash(img, leng=9, wid=8):
    img = cv2.resize(img, (leng, wid))
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    hash = []
    for i in range(wid):
        for j in range(wid):
            if image[i, j] > image[i, j + 1]:
                hash.append(1)
            else:
                hash.append(0)
    return hash


def aHash(img, leng=8, wid=8):
    img = cv2.resize(img, (leng, wid))
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    avreage = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] >= avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash


def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num

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
    initial_prefix = "_h0_"
    image_folder = '/home/duhaohao/PycharmProjects/mmselfsup-main/mmselfsup-main/data/Img2_convert_jpg'

    results_df = pd.DataFrame(columns=['Prefix', 'ahash OTSU Threshold','phash OTSU Threshold','dhash OTSU Threshold', 'whash OTSU Threshold','length ahash remove', 'length phash remove', 'length dhash remove', 'length whash remove'])
    while True:
        image_files = [file for file in os.listdir(image_folder) if file.startswith(initial_prefix)]
        ahash_hamming_distances = []
        phash_hamming_distances = []
        dhash_hamming_distances = []
        whash_hamming_distances = []
       
        image_files = natsorted(image_files)

        if not image_files:
            break

        interval=1
        length_i=len(image_files)-interval

        for i in range(0, length_i, 2): #for i in range(0, a, 1),not include a
            img_path_1 = os.path.join(image_folder, image_files[i])
            img_path_2 = os.path.join(image_folder, image_files[i + interval])    #+1 is two;  +2 is three

            img_1=cv2.imread(img_path_1)

            hash1_ahash = aHash(img_1)
            hash1_phash = pHash(img_1)
            hash1_dhash = dHash(img_1)
            hash1_whash = imagehash.whash((Image.fromarray(cv2.cvtColor(img_1,cv2.COLOR_BGR2RGB))),hash_size=8,
                                          mode ='haar', image_scale=16)
            hash1_whash=hash1_whash.hash.flatten().tolist()
            hash1_whash = [int(x) for x in hash1_whash]

            img_2 = cv2.imread(img_path_2)


            hash2_ahash = aHash(img_2)
            hash2_phash = pHash(img_2)
            hash2_dhash = dHash(img_2)
            hash2_whash = imagehash.whash((Image.fromarray(cv2.cvtColor(img_2, cv2.COLOR_BGR2RGB))), hash_size=8,
                                          mode='haar', image_scale=16)
            hash2_whash = hash2_whash.hash.flatten().tolist()
            hash2_whash = [int(x) for x in hash2_whash]


            ahash_hamming_distance = Hamming_distance(hash1_ahash,hash2_ahash)
            phash_hamming_distance = Hamming_distance(hash1_phash,hash2_phash)
            dhash_hamming_distance = Hamming_distance(hash1_dhash,hash2_dhash)
            whash_hamming_distance = Hamming_distance(hash1_whash, hash2_whash)


            print(f'Images: {os.path.basename(img_path_1)} and {os.path.basename(img_path_2)}')
            print(f'ahash Hamming distance: {ahash_hamming_distance}')
            print(f'phash Hamming distance: {phash_hamming_distance}')
            print(f'dhash Hamming distance: {dhash_hamming_distance}')
            print(f'whash Hamming distance: {whash_hamming_distance}')

            ahash_hamming_distances.append(ahash_hamming_distance)
            phash_hamming_distances.append(phash_hamming_distance)
            dhash_hamming_distances.append(dhash_hamming_distance)
            whash_hamming_distances.append(whash_hamming_distance)

            
            ahash_optimal_threshold = otsu_threshold(ahash_hamming_distances)
            phash_optimal_threshold = otsu_threshold(phash_hamming_distances)
            dhash_optimal_threshold = otsu_threshold(dhash_hamming_distances)
            whash_optimal_threshold = otsu_threshold(whash_hamming_distances)

        print("aHash Hamming distances for pairs of images:", ahash_hamming_distances)
        print("pHash Hamming distances for pairs of images:", phash_hamming_distances)
        print("dHash Hamming distances for pairs of images:", dhash_hamming_distances)
        print("wHash Hamming distances for pairs of images:", whash_hamming_distances)

        print("Hamming distances for aHash Otsu_threshold:", ahash_optimal_threshold)
        print("Hamming distances for pHash Otsu_threshold:", phash_optimal_threshold)
        print("Hamming distances for dHash Otsu_threshold:", dhash_optimal_threshold)
        print("Hamming distances for wHash Otsu_threshold:", whash_optimal_threshold)

        ahash_dist_indices = [i for i, x in enumerate(ahash_hamming_distances) if x <= ahash_optimal_threshold]
        phash_dist_indices = [i for i, x in enumerate(phash_hamming_distances) if x <= phash_optimal_threshold]
        dhash_dist_indices = [i for i, x in enumerate(dhash_hamming_distances) if x <= dhash_optimal_threshold]
        whash_dist_indices = [i for i, x in enumerate(whash_hamming_distances) if x <= whash_optimal_threshold]

        results_df = results_df._append({
            'Prefix': initial_prefix,
            'ahash OTSU Threshold': ahash_optimal_threshold,
            'length ahash remove': len(ahash_dist_indices),
            'phash OTSU Threshold': phash_optimal_threshold,
            'length phash remove': len(phash_dist_indices),
            'dhash OTSU Threshold': dhash_optimal_threshold,
            'length dhash remove': len(dhash_dist_indices),
            'whash OTSU Threshold': whash_optimal_threshold,
            'length whash remove': len(whash_dist_indices),

        }, ignore_index=True)
        
        results_df.to_excel('muti_'+'heng_original'+'_'+'a_p_d_whash_results'+'.xlsx', index=False)


     
        try:
            prefix_number_start = initial_prefix.index("_h") + 2
            prefix_number_end = initial_prefix.index("_", prefix_number_start)
            current_number = int(initial_prefix[prefix_number_start:prefix_number_end])
            new_number = current_number + 7
            initial_prefix = f"{initial_prefix[:prefix_number_start]}{new_number}{initial_prefix[prefix_number_end:]}"
        except ValueError:
            
            break