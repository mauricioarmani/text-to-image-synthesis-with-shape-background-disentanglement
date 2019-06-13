import sys
import os
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import random
import pickle

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

def lerp(val, low, high):
    return ((1.0-val)*low) + (val*high)

def slerp(val, low, high):
    omega = np.arccos(np.dot(low/np.linalg.norm(low),
        high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high

def get_interpolations(low, high, latent_dim, nb_latents, nb_interp, mode='lerp', continuity=False):
    # adjust number of latents if we want continuity
    if continuity: nb_latents -= 1

    # get interpolation method
    if mode == 'lerp':
        interp_fn = lerp
    elif mode == 'slerp':
        interp_fn = slerp

    # sample first latent vector (left)
    first = low.copy() # saved for last continuity later
    latent_interps = np.empty(shape=(0, latent_dim), dtype=np.float32)
    for _ in range(nb_latents):
        # sample second latent vector (right)
        # interpolation mixture
        interp_vals = np.linspace(0, 1, num=nb_interp)
        # generate interpolations
        latent_interp = np.array([interp_fn(v,low,high) for v in interp_vals],
                dtype=np.float32)
        # save results
        latent_interps = np.vstack((latent_interps, latent_interp))
        # maintain continuity between samples
        low = high

    # check if we want to keep continuity between first and last image
    if continuity:
        last_interp = np.array([interp_fn(v,high,first) for v in interp_vals],
                dtype=np.float32)
        latent_interps = np.vstack((latent_interps, last_interp))

    return latent_interps 

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Gans')

    parser.add_argument('--epoch', type=int, default=0,  
                        help='load from epoch')
    parser.add_argument('--model', type=str, default='',  
                        help='model name')
    parser.add_argument('--nb_interp', type=int, default=10,  
                        help='nb_interp')
    parser.add_argument('--interpolate', type=str, default='shape', choices=['text','shape','background'], 
                        help='Noise to be interpolated')
    parser.add_argument('--fx1_id', type=int, default=0,  
                        help='Fixed noise 1 id')
    parser.add_argument('--fx2_id', type=int, default=1,  
                        help='Fixed noise 1 id')
    parser.add_argument('--mv1_id', type=int, default=2,  
                        help='Moving noise low id')
    parser.add_argument('--mv2_id', type=int, default=3,  
                        help='Moving noise high id')

    args = parser.parse_args()

    epoch = args.epoch
    model_name = args.model
    nb_interp = args.nb_interp
    interpolate = args.interpolate
    fx1_id = args.fx1_id
    fx2_id = args.fx2_id
    mv1_id = args.mv1_id
    mv2_id = args.mv2_id

    # set file name
    file = 'epoch_%d' % epoch

    sample_name = file
    png_file = file + '_%s_interpolation.png' % interpolate
    z_file = file + '.pickle'

    # cfgs
    data_name = 'birds'
    emb_dim = 128
    scode_dim = 1024 # segmentation enconded dim

    # folders
    model_name = '{}_{}'.format(model_name, data_name)
    model_folder = os.path.join(model_root, model_name)
    sample_folder = os.path.join(model_folder, sample_name)

    # open noise tensor
    with open(os.path.join(sample_folder, z_file), 'br') as f:
        z_list = pickle.load(f)

    # Load generator model
    netG  = Generator(tcode_dim=512, scode_dim=scode_dim, emb_dim=emb_dim, hid_dim=128)
    G_weightspath  = os.path.join(model_folder, 'G_epoch{}.pth'.format(epoch))
    netG.load_state_dict(torch.load(G_weightspath))
    netG = netG.cuda()
    netG.eval()

    if interpolate == 'text':
        mv_noise  = 0
        fx_noise1 = 1
        fx_noise2 = 2
    elif interpolate == 'shape':
        fx_noise1 = 0
        mv_noise  = 1
        fx_noise2 = 2
    elif interpolate == 'background':
        fx_noise1 = 0
        fx_noise2 = 1
        mv_noise  = 2

    # interpolate shape from two instances
    low  = np.array(z_list[mv_noise][mv1_id].data.cpu())
    high = np.array(z_list[mv_noise][mv2_id].data.cpu())
    numpy_interp = get_interpolations(low, high, 128, 1, nb_interp, mode='lerp')
    z_interp = torch.from_numpy(numpy_interp).cuda()

    # generate samples
    f_images = torch.FloatTensor(nb_interp, 3, 64, 64)
    vis_samples = [None] * nb_interp
    for i, z_mv_i in enumerate(z_interp):
        # fixed
        z_f1_i = z_list[fx_noise1][fx1_id].unsqueeze(0)
        z_f2_i = z_list[fx_noise2][fx2_id].unsqueeze(0)

        # moving
        z_mv_i = z_mv_i.unsqueeze(0)

        if interpolate == 'text':
            f_image, _ = netG(z_list=[z_mv_i, z_f1_i, z_f2_i])

        elif interpolate == 'shape':
            f_image, _ = netG(z_list=[z_f1_i, z_mv_i, z_f2_i])

        elif interpolate == 'background':
            f_image, _ = netG(z_list=[z_f1_i, z_f2_i, z_mv_i])

        np_fake = to_numpy(f_image)
        vis_samples[i] = np_fake

    # save images
    save_images(vis_samples, save=not sample_folder == '', 
        save_path=os.path.join(sample_folder, png_file), dim_ordering='th')

    print('Images and captions saved at %s' % sample_folder)