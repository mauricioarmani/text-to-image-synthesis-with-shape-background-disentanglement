#!/usr/bin/env python3
from __future__ import absolute_import, division, print_function
import os
import glob
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf
import h5py
import json

tf.app.flags.DEFINE_string('image_folder','',"""Path where to load the images """)
tf.app.flags.DEFINE_string('h5_file','',"""Path where to load the images """)
FLAGS = tf.app.flags.FLAGS

# Paths
# image_path = '/home/mauricio/MyGan/Results/birds/GAN_64_birds_testing_num_10/GAN_64_birds_G_epoch_200.h5' # set path to some generated images
image_path = os.path.join(FLAGS.image_folder, FLAGS.h5_file)
stats_path = 'evaluation/fid/fid_stats.npz' # training set statistics
inception_path = fid.check_or_download_inception(None) # download inception network


# loads all images into memory (this might require a lot of RAM!)
# image_list = glob.glob(os.path.join(datapath, '*.jpg'))
# images = np.array([imread(str(fn)).astype(np.float32) for fn in files])

f = h5py.File(image_path, 'r')
images = f['output_64']

# load precalculated training set statistics
f = np.load(stats_path)
mu_real, sigma_real = f['mu'][:], f['sigma'][:]
f.close()

fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)

fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
print('Image path:', image_path)
print("FID: %s" % fid_value)

json_path = os.path.join(FLAGS.image_folder ,FLAGS.h5_file[:-3] + '_FID.json')
json.dump({'FID': fid_value}, open(json_path,'w'), indent=True)