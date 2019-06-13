#!/usr/bin/env python3

import os
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf
import pickle
from PIL import Image


########
# PATHS
########
# data_path = 'data' # set path to training set images
output_path = 'fid_stats.npz' # path for where to store the statistics
# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you
inception_path = None
print("check for inception model..", end=" ", flush=True)
inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
print("ok")

# loads all images into memory (this might require a lot of RAM!)
print("load images..", end=" " , flush=True)
print()

pickle_img_path = '/home/mauricio/datasets/birds/train/76images.pickle'
with open(pickle_img_path, 'rb') as f:
            images76 = pickle.load(f)
            images76 = np.array(images76)
            images = np.empty((images76.shape[0], 64, 64, 3))
            for i, img in enumerate(images76):
            	img = Image.fromarray(img)
            	img = img.resize((64, 64), resample=Image.BILINEAR)
            	img = np.array(img)
            	images[i] = img

print(images.shape)

# image_list = glob.glob(os.path.join(data_path, '*.jpg'))
# images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])

print("%d images found and loaded" % len(images))

print("create inception graph..", end=" ", flush=True)
fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
print("ok")

print("calculte FID stats..", end=" ", flush=True)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
    np.savez_compressed(output_path, mu=mu, sigma=sigma)
print("finished")