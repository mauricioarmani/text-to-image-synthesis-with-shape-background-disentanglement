import sys
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import random
import pickle
from scipy import signal
from scipy.ndimage.filters import convolve

proj_root = '.'
sys.path.insert(0, proj_root)
data_root = 'data'
model_root = 'models'

from gan.data_loader import BirdsDataset
from gan.networks import Generator
from gan.networks import ImgEncoder
from gan.proj_utils.local_utils import save_images
from gan.proj_utils.torch_utils import to_torch, to_numpy, to_binary, roll


from PIL import Image


def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def SSIM(img1, img2, seg1, seg2, max_val=255, filter_size=11,
                       filter_sigma=1.5, k1=0.01, k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.
  
    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
  
    Arguments:
      img1: Numpy array holding the first RGB image batch.
      img2: Numpy array holding the second RGB image batch.
      max_val: the dynamic range of the images (i.e., the difference between the
        maximum the and minimum allowed values).
      filter_size: Size of blur kernel to use (will be reduced for small 
      images).
      filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
        for small images).
      k1: Constant used to maintain stability in the SSIM calculation (0.01 in
        the original paper).
      k2: Constant used to maintain stability in the SSIM calculation (0.03 in
        the original paper).
  
    Returns:
      Pair containing the mean SSIM and contrast sensitivity between `img1` and
      `img2`.
  
    Raises:
      RuntimeError: If input images don't have the same shape or don't have four
        dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).',
                           img1.shape, img2.shape)
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d',
                           img1.ndim)

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        s_window = np.ones_like(window) / (filter_size*filter_size)
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
        seg1 = signal.convolve(seg1, s_window, mode='valid')
        seg2 = signal.convolve(seg2, s_window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12
    
    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    bkg_seg1 = (seg1 <= 0.008) # aprox. threshold for 1 pixel
    bkg_seg2 = (seg2 <= 0.008) # aprox. threshold for 1 pixel

    mask = (bkg_seg1 & bkg_seg2)

    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2

    ssim = (((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2))

    mask_ssim = (ssim * mask).sum()/(mask.sum() * 3) # 3 channels

    return mask_ssim

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Gans')

    parser.add_argument('--epoch', type=int, default=0,  
                        help='load from epoch')
    parser.add_argument('--model', type=str, default='',  
                        help='model name')
    parser.add_argument('--batch_size', type=int, default=10,  
                        help='batch_size')
    parser.add_argument('--align',    type=str,  choices=['shape', 'background', 'all', 'none'], 
                        help='Which concept to align during generation.')
    args = parser.parse_args()

    epoch = args.epoch
    model_name = args.model
    batch_size = args.batch_size
    align = args.align

    # set file name
    file = 'epoch_%d' % epoch

    sample_name = file
    png_file = file + '.png'
    txt_file = file + '.txt' 
    z_file = file + '.pickle'

    # cfgs
    data_name = 'birds'
    emb_dim = 128
    scode_dim = 1024 # segmentation enconded dim

    # folders
    datadir = os.path.join(data_root, data_name)
    model_name = '{}_{}'.format(model_name, data_name)
    model_folder = os.path.join(model_root, model_name)

    # NNs
    netG  = Generator(tcode_dim=512, scode_dim=scode_dim, emb_dim=emb_dim, hid_dim=128)
    netEs = ImgEncoder(num_chan=1, out_dim=scode_dim)
    netEb = ImgEncoder(num_chan=3, out_dim=scode_dim)

    # Dataset
    dataset    = BirdsDataset(datadir, mode='test')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # load models from checkpoint
    G_weightspath  = os.path.join(model_folder, 'G_epoch{}.pth'.format(epoch))
    D_weightspath  = os.path.join(model_folder, 'D_epoch{}.pth'.format(epoch))
    Es_weightspath = os.path.join(model_folder, 'Es_epoch{}.pth'.format(epoch))
    Eb_weightspath = os.path.join(model_folder, 'Eb_epoch{}.pth'.format(epoch))

    netG.load_state_dict(torch.load(G_weightspath))
    netEs.load_state_dict(torch.load(Es_weightspath))
    netEb.load_state_dict(torch.load(Eb_weightspath))

    # set to GPU
    netG  = netG.cuda()
    netEs = netEs.cuda()
    netEb = netEb.cuda()

    scores = []
    with torch.no_grad():
        for i in range(100):
            print('%d/100' % i)
            # get batch of test samples
            images, _, segs, txt_data, txt_len, captions, *_ = next(iter(dataloader))

            images = images.cuda()
            segs = segs.cuda()    
            txt_data = txt_data.cuda()
            bimages = images

            netG.eval()
            netEs.eval()
            netEb.eval()

            # alignment
            if align == 'shape':
                bimages = roll(images, 2, dim=0) # for text and seg mismatched backgrounds
                bsegs   = roll(segs, 2, dim=0) # for text and seg mismatched backgrounds
            elif align == 'background':
                segs = roll(segs, 1, dim=0) # for text mismatched segmentations
            elif align == 'all':
                bimages = images.clone()
                bsegs   = segs.clone()
            elif align == 'none':
                bimages = roll(images, 2, dim=0) # for text and seg mismatched backgrounds
                segs    = roll(segs, 1, dim=0) # for text mismatched segmentations
                bsegs   = roll(segs, 2, dim=0) # for text and seg mismatched backgrounds

            np_segs    = to_numpy(segs)
            np_bsegs   = to_numpy(bsegs)
            np_images  = to_numpy(images)
            np_bimages = to_numpy(bimages)

            segs_code = netEs(segs)
            bkgs_code = netEb(bimages)
            
            *_, f_images, z_list = netG(txt_data, txt_len, segs_code, bkgs_code)
            
            np_fakes = to_numpy(f_images)

            for x, b, s, sb in zip(np_fakes, np_bimages, np_segs, np_bsegs):
                x  = (x.transpose(1,2,0) + 1)/2. * 255.
                b  = (b.transpose(1,2,0) + 1)/2. * 255.
                s  = s.transpose(1,2,0)
                sb = sb.transpose(1,2,0)
                ssim = SSIM(x[np.newaxis,:,:,:], b[np.newaxis,:,:,:], s[np.newaxis,:,:,:], sb[np.newaxis,:,:,:])
                # ssim = SSIM(x[np.newaxis,:,:,:], x[np.newaxis,:,:,:], s[np.newaxis,:,:,:], s[np.newaxis,:,:,:])
                if not np.isnan(ssim):
                    scores.append(ssim)

        print('SSSIM = %f +- %f'(np.array(scores).mean(),np.array(scores).std()))
