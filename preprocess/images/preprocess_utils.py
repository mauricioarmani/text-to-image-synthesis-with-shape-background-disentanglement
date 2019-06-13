"""
Some codes from
https://github.com/openai/improved-gan/blob/master/imagenet/utils.py
"""
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import errno
from PIL import Image


def get_image(image_path, image_size, is_crop=False, bbox=None):
    global index
    out = transform(imread(image_path), image_size, is_crop, bbox)
    return out


def custom_crop(img, bbox):
    imsiz = img.shape  # [height, width, channel]
    center_x = int((2 * bbox[0] + bbox[2]) / 2)
    center_y = int((2 * bbox[1] + bbox[3]) / 2)
    R = int(np.maximum(bbox[2], bbox[3]) * 0.75)
    y1 = np.maximum(0, center_y - R)
    y2 = np.minimum(imsiz[0], center_y + R)
    x1 = np.maximum(0, center_x - R)
    x2 = np.minimum(imsiz[1], center_x + R)
    img_cropped = img[y1:y2, x1:x2]
    return img_cropped


def transform(image, image_size, is_crop, bbox):
    if is_crop:
        image = custom_crop(image, bbox)
    image = Image.fromarray(image.astype('uint8'))
    transformed_image =\
        image.resize(size=(image_size, image_size), resample=Image.BICUBIC)
    return np.array(transformed_image)


def imread(path):
    img = Image.open(path)
    img = np.array(img)
    if len(img.shape) != 2:
            img = img[:,:,0]
    if len(img.shape) == 0:
        raise ValueError(path + " got loaded as a dimensionless array!")
    return img.astype(np.float)


def colorize(img):
    if img.ndim == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)
        img = np.concatenate([img, img, img], axis=2)
    if img.shape[2] == 4:
        img = img[:, :, 0:3]
    return img


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise