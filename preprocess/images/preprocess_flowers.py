'''
Flowers Data Setup

1. Download the original dataset:
   * http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
2. Download the segmentation images:
   * http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102segmentations.tgz
3. Download the raw textual data from:
   * https://github.com/reedscot/cvpr2016
4. Download the textual embedding data:
   * https://github.com/reedscot/icml2016

After downloading all the necessary data, run this script to preprocess the
original data and dump it on the data folder (check parameters for further
information).

You must also preprocess the textual data for the flowers dataset using
FastText. Place the preprocessed FastText embeddings in the same folder.
'''
import os
import argparse
from tqdm import tqdm

from glob import glob

import torch
import torchfile
import numpy as np
from PIL import Image
import pickle



def main(args):
    tqdm.write('Preprocessing train data')
    dump_flowers_split(args.dir_imgs, args.dir_txts, args.dir_embs,
            args.dir_segs, 'trainval', args.data_dump_path)

    tqdm.write('Preprocessing test data')
    dump_flowers_split(args.dir_imgs, args.dir_txts, args.dir_embs,
            args.dir_segs, 'test', args.data_dump_path)



def dump_flowers_split(dir_imgs, dir_txts, dir_embs, dir_segs, split, data_dump_path):
    tqdm.write('Loading data')
    data = extract_flowers_info(dir_imgs, dir_txts, dir_embs, dir_segs, split)

    num_instances = len(data)

    filenames = []
    class_info = np.empty(num_instances, dtype=np.uint8)
    imgs76 = np.empty((num_instances, 76, 76, 3), dtype=np.uint8)
    imgs304 = np.empty((num_instances, 304, 304, 3), dtype=np.uint8)
    segs76 = np.empty((num_instances, 76, 76, 1), dtype=np.uint8)
    segs304 = np.empty((num_instances, 304, 304, 1), dtype=np.uint8)
    ccr_embs = np.empty((num_instances, 10, 1024))

    tqdm.write('Preprocessing instances')
    for idx, datum in enumerate(tqdm(data)):
        img = Image.open(datum['path_img'])
        img76 = np.array(img.resize((76, 76)), dtype=np.uint8)
        img304 = np.array(img.resize((304, 304)), dtype=np.uint8)

        seg = Image.open(datum['path_seg'])
        seg76 = np.array(seg.resize((76, 76)), dtype=np.uint8)
        seg76 = np.all(seg76 != [0, 0, 254], axis=-1)
        seg76 = np.expand_dims(seg76, axis=2)
        seg304 = np.array(seg.resize((304, 304)), dtype=np.uint8)
        seg304 = np.all(seg304 != [0, 0, 254], axis=-1)
        seg304 = np.expand_dims(seg304, axis=2)

        imgs76[idx] = img76
        imgs304[idx] = img304
        segs76[idx] = seg76
        segs304[idx] = seg304
        filenames.append(datum['name_img'])
        class_info[idx] = datum['class_index']
        ccr_embs[idx] = datum['embs']

    if split == 'trainval': split = 'train'
    path_save = os.path.join(data_dump_path, split)
    os.makedirs(os.path.join(path_save), exist_ok=True)

    tqdm.write('Saving preprocessed instances')
    with open(os.path.join(path_save, '76images.pickle'), 'wb') as f:
        pickle.dump(imgs76, f)
    with open(os.path.join(path_save, '76segmentations.pickle'), 'wb') as f:
        pickle.dump(segs76, f)
    with open(os.path.join(path_save, '304images.pickle'), 'wb') as f:
        pickle.dump(imgs304, f)
    with open(os.path.join(path_save, '304segmentations.pickle'), 'wb') as f:
        pickle.dump(segs304, f)
    with open(os.path.join(path_save, 'class_info.pickle'), 'wb') as f:
        pickle.dump(class_info, f)
    with open(os.path.join(path_save, 'filenames.pickle'), 'wb') as f:
        pickle.dump(filenames, f)
    with open(os.path.join(path_save, 'char-CNN-RNN-embeddings.pickle'), 'wb') as f:
        pickle.dump(ccr_embs, f)



def extract_flowers_info(dir_imgs, dir_txts, dir_embs, dir_segs, split):
    data = []
    split_classes_file = os.path.join(dir_txts, split+'classes.txt')
    classes = [line.rstrip('\n') for line in open(split_classes_file)]

    for cls in classes:
        dir_cls_embs = os.path.join(dir_embs, cls)
        dir_cls_txts = os.path.join(dir_txts, 'text_c10', cls)

        paths_data = sorted(glob(dir_cls_embs + '/*.t7'))
        paths_txt = sorted(glob(dir_cls_txts + '/*.txt'))

        for path_datum, path_raw_txts in zip(paths_data, paths_txt):
            path_datum = os.path.join(dir_cls_embs, path_datum)
            path_raw_txts = os.path.join(dir_cls_txts, path_raw_txts)

            datum = torchfile.load(path_datum)
            name_img = datum[b'img'].decode('utf-8').split('/')[-1][:-4]
            path_img = os.path.join(dir_imgs, 'jpg', name_img + '.jpg')
            name_seg = name_img.replace('image', 'segmim')
            path_seg = os.path.join(dir_segs, 'segmim', name_seg + '.jpg')

            idx = name_img.split('_')[-1]
            cls_idx = int(cls.split('_')[-1])

            embs = datum[b'txt']

            #with open(path_raw_txts, 'r') as f:
            #    txts = [line.strip() for line in f.readlines()]

            data.append({
                'index': idx,
                'class_name': cls,
                'class_index': cls_idx,
                'name_img': name_img,
                'path_img': path_img,
                'path_seg': path_seg,
                'embs': embs
                #'path_txts': path_txts,
                #'txts': txts
            })


    return data



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dir_imgs = 102flowers
    # dir_txts = cvpr2016_flowers
    # dir_embs = flowers_icml
    # dir_segs = 102segmentations
    parser.add_argument('--dir_imgs', type=str, required=True,
            help='Path to image files')
    parser.add_argument('--dir_txts', type=str, required=True,
            help='Path to text files')
    parser.add_argument('--dir_embs', type=str, required=True,
            help='Path to embedding files')
    parser.add_argument('--dir_segs', type=str, required=True,
            help='Path to segmentation files')
    parser.add_argument('--data_dump_path', type=str, required=True,
            help='Path to dump extracted data')

    args = parser.parse_args()
    main(args)
