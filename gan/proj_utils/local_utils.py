import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import shutil
import scipy.misc as misc


def mkdirs(folders, erase=False):
    if type(folders) is not list:
        folders = [folders]
    for fold in folders:
        if not os.path.exists(fold):
            os.makedirs(fold)
        else:
            if erase:
                shutil.rmtree(fold)
                os.makedirs(fold)
                
def writeImg(array, savepath):
    im = Image.fromarray(array)
    im.save(savepath)

def imresize_shape(img, outshape):
    if len(img.shape) == 3:
        if img.shape[0] == 1 or img.shape[0] == 3:
            transpose_img = np.transpose(img, (1,2,0))
            _img =  imresize_shape(transpose_img, outshape)
            return np.transpose(_img, (2,0, 1))
    if len(img.shape) == 4: 
        img_out = []
        for this_img in img:
            img_out.append( imresize_shape(this_img, outshape) ) 
        return np.stack(img_out, axis=0)

    img = img.astype(np.float32)
    outshape = (int(outshape[1]) , int(outshape[0]))
    
    temp = misc.imresize(img, size=outshape, interp='bilinear').astype(float)

    if len(img.shape) == 3 and img.shape[2] == 1:
        temp = np.reshape(temp, temp.shape + (1,))
    return temp

def IndexH5(h5_array, indices):
    read_list = []
    for idx in indices:
        read_list.append(h5_array[idx])
    return np.stack(read_list, 0)

def save_images(X_list, save_path=None, save=True, dim_ordering='tf'):

    # X_list: list of X
    # X: B*C*H*W

    X = X_list[0]
    n_samples = X.shape[0]
    nh = n_samples
    nw = len(X_list)

    if X.ndim == 4:
        # BCHW -> BHWC
        if dim_ordering == 'tf':
            pass
        else:
            for idx, X in enumerate(X_list):
                X_list[idx] = X.transpose(0, 2, 3, 1)

        X = X_list[0]
        h, w, c = X[0].shape[:3]
        hgap, wgap = int(0.1*h), int(0.1*w)
        img = np.zeros(((h+hgap)*nh - hgap, (w+wgap)*nw-wgap, c)) -1

    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        c = 0
        hgap, wgap = int(0.1*h), int(0.1*w)
        img = np.zeros(((h+hgap)*nh - hgap, (w+wgap)*nw - wgap)) - 1
    else:
        assert 0, 'you have wrong number of dimension input {}'.format(X.ndim)

    for n, x_tuple in enumerate(zip(*X_list)):
        i = n
        for j, x in enumerate(x_tuple):
            rs, cs = i*(h+hgap), j*(w+wgap)
            img[rs:rs+h, cs:cs+w] = x

    if c == 1:
        img = img[:, :, 0]

    if save:
        img = (img + 1)/2 * 255 # img
        save_image = img.astype(np.uint8)
        writeImg(save_image, save_path)
    return img

def imshow(img, size=None):
    if size is not None:
        plt.figure(figsize = size)
    else:
        plt.figure()
    plt.imshow(img)
    plt.show()

def normalize_img(X):
    min_, max_ = np.min(X), np.max(X)
    X = (X - min_)/ (max_ - min_ + 1e-9)
    X = X*255
    return X.astype(np.uint8) 

def Indexflow(Totalnum, batch_size, random=True):
    numberofchunk = int(Totalnum + batch_size - 1)// int(batch_size)   # the floor
    #Chunkfile = np.zeros((batch_size, row*col*channel))
    totalIndx = np.arange(Totalnum).astype(np.int)
    if random is True:
        totalIndx = np.random.permutation(totalIndx)
        
    chunkstart = 0
    for chunkidx in range(int(numberofchunk)):
        thisnum = min(batch_size, Totalnum - chunkidx*batch_size)
        thisInd = totalIndx[chunkstart: chunkstart + thisnum]
        chunkstart += thisnum
        yield thisInd       