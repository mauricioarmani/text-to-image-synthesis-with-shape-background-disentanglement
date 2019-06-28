import sys
import os
import numpy as np
import torch
import argparse
import pickle

proj_root = '.'
sys.path.insert(0, proj_root)
data_root = 'data'
model_root = 'models'

from gan.networks import Generator
from gan.proj_utils.local_utils import save_images
from gan.proj_utils.torch_utils import to_numpy


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
    parser.add_argument('--fx1_id', type=int, default=None,  
                        help='Fixed noise 1 id')
    parser.add_argument('--fx2_id', type=int, default=None,  
                        help='Fixed noise 2 id')
    parser.add_argument('--mv_ida', type=int, default=None,  
                        help='Moving noise low id')
    parser.add_argument('--mv_idb', type=int, default=None,  
                        help='Moving noise high id')

    args = parser.parse_args()

    epoch = args.epoch
    model_name = args.model
    nb_interp = args.nb_interp
    interpolate = args.interpolate
    fx1_id = args.fx1_id
    fx2_id = args.fx2_id
    mv_ida = args.mv_ida
    mv_idb = args.mv_idb

    if not fx1_id:
        fx1_id = np.random.randint(0,160)
    if not fx2_id:
        fx2_id = np.random.randint(0,160)
    if not mv_ida:
        mv_ida = np.random.randint(0,160)
    if not mv_idb:
        mv_idb = np.random.randint(0,160)

    # set file name
    file = 'epoch_%d' % epoch
    sample_name = file
    z_file = file + '.pickle'

    # cfg
    data_name = 'birds'

    # folders
    model_name = '{}_{}'.format(model_name, data_name)
    model_folder = os.path.join(model_root, model_name)
    sample_folder = os.path.join(model_folder, sample_name)

    # open noise tensor
    with open(os.path.join(sample_folder, z_file), 'br') as f:
        z_list = pickle.load(f)

    # Load generator model
    netG  = Generator(tcode_dim=512, scode_dim=1024, emb_dim=128, hid_dim=128)
    G_weightspath  = os.path.join(model_folder, 'G_epoch{}.pth'.format(epoch))
    netG.load_state_dict(torch.load(G_weightspath))
    netG = netG.cuda()
    netG.eval()

    if interpolate == 'text':
        mv_noise  = 0 # text
        fx_noise1 = 1 # shape
        fx_noise2 = 2 # bkg
        png_file = '1Dinterp__text=%dto%d__shape=%d__background=%d.png' % (mv_ida, mv_idb, fx1_id, fx2_id)

    elif interpolate == 'shape':
        fx_noise1 = 0 # text
        mv_noise  = 1 # shape
        fx_noise2 = 2 # bkg
        png_file = '1Dinterp__text=%d__shape=%dto%d__background=%d.png' % (fx1_id, mv_ida, mv_idb, fx2_id)

    elif interpolate == 'background':
        fx_noise1 = 0 # text
        fx_noise2 = 1 # shape
        mv_noise  = 2 # bkg
        png_file = '1Dinterp__text=%d__shape=%d__background=%dto%d.png' % (fx1_id, fx2_id, mv_ida, mv_idb)

    # get interpolations
    n1a = z_list[mv_noise][mv_ida].unsqueeze(0)
    n1b = z_list[mv_noise][mv_idb].unsqueeze(0)
    n2_ = z_list[fx_noise1][fx1_id].unsqueeze(0)
    n3_ = z_list[fx_noise2][fx2_id].unsqueeze(0)

    # generate samples
    vis_samples = [None] * nb_interp
    for x in range(nb_interp):
        # 1D interpolation
        cx = x/(nb_interp)
        z_mv = n1a * (1-cx) + n1b * (cx)

        # get fixed sample
        z_f1  = n2_
        z_f2  = n3_

        if interpolate == 'text':
            f_image, _ = netG(z_list=[z_mv, z_f1, z_f2])

        elif interpolate == 'shape':
            f_image, _ = netG(z_list=[z_f1, z_mv, z_f2])

        elif interpolate == 'background':
            f_image, _ = netG(z_list=[z_f1, z_f2, z_mv])

        np_fake = to_numpy(f_image)
        vis_samples[x] = np_fake

    # save images
    save_images(vis_samples, save=not sample_folder == '', 
        save_path=os.path.join(sample_folder, png_file), dim_ordering='th')

    print('Images saved at %s' % sample_folder)