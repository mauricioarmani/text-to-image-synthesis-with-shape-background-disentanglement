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
from gan.proj_utils.torch_utils import to_numpy, to_torch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Gans')

    parser.add_argument('--epoch', type=int, default=0,  
                        help='load from epoch')
    parser.add_argument('--model', type=str, default='',  
                        help='model name')
    parser.add_argument('--nb_interp', type=int, default=10,  
                        help='nb_interp')
    parser.add_argument('--fix', type=str, default='shape', choices=['text','shape','background'], 
                        help='Noise to be fixed')
    parser.add_argument('--fx_id', type=int, default=0,  
                        help='Fixed noise 1 id')
    parser.add_argument('--mv1_ida', type=int, default=None,  
                        help='Moving noise low id')
    parser.add_argument('--mv1_idb', type=int, default=None,  
                        help='Moving noise high id')
    parser.add_argument('--mv2_ida', type=int, default=None,  
                        help='Moving noise low id')
    parser.add_argument('--mv2_idb', type=int, default=None,  
                        help='Moving noise high id')    

    args = parser.parse_args()

    epoch = args.epoch
    model_name = args.model
    nb_interp = args.nb_interp
    fix = args.fix
    fx_id = args.fx_id
    mv1_ida = args.mv1_ida
    mv1_idb = args.mv1_idb
    mv2_ida = args.mv2_ida
    mv2_idb = args.mv2_idb

    if not fx_id:
        fx_id = np.random.randint(0,100)
    if not mv1_ida:
        mv1_ida = np.random.randint(0,100)
    if not mv1_idb:
        mv1_idb = np.random.randint(0,100)
    if not mv2_ida:
        mv2_ida = np.random.randint(0,100)
    if not mv2_idb:
        mv2_idb = np.random.randint(0,100)

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

    if fix == 'text':
        fx_noise  = 0 # text
        mv_noise1 = 1 # shape
        mv_noise2 = 2 # bkg
        png_file = '2Dinterp__text=%d__shape=%dto%d__background=%dto%d.png' % (fx_id, mv1_ida, mv1_idb, mv2_ida, mv2_idb)
        
    elif fix == 'shape':
        mv_noise1 = 0 # text
        fx_noise  = 1 # shape
        mv_noise2 = 2 # bkg
        png_file = '2Dinterp__text=%dto%d__shape=%d__background=%dto%d.png' % (mv1_ida, mv1_idb, fx_id, mv2_ida, mv2_idb)

    elif fix == 'background':
        mv_noise1 = 0 # text
        mv_noise2 = 1 # shape
        fx_noise  = 2 # bkg
        png_file = '2Dinterp__text=%dto%d__shape=%dto%d__background=%d.png' % (mv1_ida, mv1_idb, mv2_ida, mv2_idb, fx_id)

    # get interpolations
    n1a = z_list[mv_noise1][mv1_ida].unsqueeze(0)
    n1b = z_list[mv_noise1][mv1_idb].unsqueeze(0)
    n2a = z_list[mv_noise2][mv2_ida].unsqueeze(0)
    n2b = z_list[mv_noise2][mv2_idb].unsqueeze(0)
    n3_ = z_list[fx_noise][fx_id].unsqueeze(0)

    # generation
    vis_samples = [None] * nb_interp
    for x in range(nb_interp):
        column = np.empty(shape=(nb_interp, 3, 64, 64), dtype=np.float32)
        for y in range(nb_interp):
            # 2D interpolation
            cx = x/(nb_interp)
            cy = y/(nb_interp)
            z_mv1 = n1a * (1-cx) + n1b * (cx)
            z_mv2 = n2a * (1-cy) + n2b * (cy)

            # get sample
            z_fx  = n3_

            if fix == 'text':
                f_image, _ = netG(z_list=[z_fx, z_mv1, z_mv2])

            elif fix == 'shape':
                f_image, _ = netG(z_list=[z_mv1, z_fx, z_mv2])

            elif fix == 'background':
                f_image, _ = netG(z_list=[z_mv1, z_mv2, z_fx])

            np_fake = to_numpy(f_image)
            column[y] = np_fake

        vis_samples[x] = column

    # save images
    save_images(vis_samples, save=not sample_folder == '', 
        save_path=os.path.join(sample_folder, png_file), dim_ordering='th')

    print('Images and captions saved at %s' % sample_folder)