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
    parser.add_argument('--nb_interp', type=int, default=5,  
                        help='nb_interp')
    parser.add_argument('--mv1_ida', type=int, default=None,  
                        help='Moving noise low id')
    parser.add_argument('--mv1_idb', type=int, default=None,  
                        help='Moving noise high id')
    parser.add_argument('--mv2_ida', type=int, default=None,  
                        help='Moving noise low id')
    parser.add_argument('--mv2_idb', type=int, default=None,  
                        help='Moving noise high id')
    parser.add_argument('--mv3_ida', type=int, default=None,  
                        help='Moving noise low id')
    parser.add_argument('--mv3_idb', type=int, default=None,  
                        help='Moving noise high id')    

    args = parser.parse_args()

    epoch = args.epoch
    model_name = args.model
    nb_interp = args.nb_interp
    mv1_ida = args.mv1_ida
    mv1_idb = args.mv1_idb
    mv2_ida = args.mv2_ida
    mv2_idb = args.mv2_idb
    mv3_ida = args.mv3_ida
    mv3_idb = args.mv3_idb

    if not mv1_ida:
        mv1_ida = np.random.randint(0,100)
    if not mv1_idb:
        mv1_idb = np.random.randint(0,100)
    if not mv2_ida:
        mv2_ida = np.random.randint(0,100)
    if not mv2_idb:
        mv2_idb = np.random.randint(0,100)
    if not mv3_ida:
        mv3_ida = np.random.randint(0,100)
    if not mv3_idb:
        mv3_idb = np.random.randint(0,100)

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

    # Load model
    netG  = Generator(tcode_dim=512, scode_dim=1024, emb_dim=128, hid_dim=128)
    G_weightspath  = os.path.join(model_folder, 'G_epoch{}.pth'.format(epoch))
    netG.load_state_dict(torch.load(G_weightspath))
    netG = netG.cuda()
    netG.eval()

    # get noise
    n1a = z_list[0][mv1_ida].unsqueeze(0)
    n1b = z_list[0][mv1_idb].unsqueeze(0)
    n2a = z_list[1][mv2_ida].unsqueeze(0)
    n2b = z_list[1][mv2_idb].unsqueeze(0)
    n3a = z_list[2][mv3_ida].unsqueeze(0)
    n3b = z_list[2][mv3_idb].unsqueeze(0)

    # generation
    for z in range(nb_interp):
        vis_samples = [None] * nb_interp
        for x in range(nb_interp):
            column = np.empty(shape=(nb_interp, 3, 64, 64), dtype=np.float32)
            for y in range(nb_interp):
                # 3D interpolation
                cx = x/(nb_interp)
                cy = y/(nb_interp)
                cz = z/(nb_interp)

                z_mv1 = n1a * (1-cx) + n1b * (cx)
                z_mv2 = n2a * (1-cy) + n2b * (cy)
                z_mv3 = n3a * (1-cz) + n3b * (cz)

                # get sample
                f_image, _ = netG(z_list=[z_mv1, z_mv2, z_mv3])
                np_fake = to_numpy(f_image)

                column[y] = np_fake

            vis_samples[x] = column

        # save images
        png_file = '3Dinterp__text=%dto%d__shape=%dto%d__background=%dto%d-%d.png' % \
            (mv1_ida, mv1_idb, mv2_ida, mv2_idb, mv3_ida, mv3_idb, z)

        save_images(vis_samples, save=not sample_folder == '', 
            save_path=os.path.join(sample_folder, png_file), dim_ordering='th')

    print('Images saved at %s' % sample_folder)