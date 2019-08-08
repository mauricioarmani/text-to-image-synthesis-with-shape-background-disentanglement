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
    parser.add_argument('--fix_seed', action='store_true',
                        help='Fix seed.')
    parser.add_argument('--idx', type=int, default=0, 
                        help='idx')
    args = parser.parse_args()

    epoch = args.epoch
    model_name = args.model
    batch_size = args.batch_size
    align = args.align
    fix_seed = args.fix_seed
    idx = args.idx

    if fix_seed:
        seed = 1231251
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # set file name
    file = 'epoch_%d' % epoch

    sample_name = file
    png_file = file + '%d.png' % idx
    txt_file = file + '.txt' 
    z_file = file + '.pickle'

    # folders
    datadir = os.path.join(data_root, 'birds')
    model_name = '{}_{}'.format(model_name, 'birds')
    model_folder = os.path.join(model_root, model_name)
    sample_folder = os.path.join(model_folder, sample_name)

    # create sample folder
    if not os.path.exists(sample_folder):
        os.mkdir(sample_folder)

    # NNs
    netG  = Generator(tcode_dim=512, scode_dim=1024, emb_dim=128, hid_dim=128)
    netEs = ImgEncoder(num_chan=1, out_dim=1024)
    netEb = ImgEncoder(num_chan=3, out_dim=1024)

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

    # # get batch of test samples
    images, _, segs, txt_data, txt_len, captions, *_ = next(iter(dataloader))

    images = images.cuda()
    segs = segs.cuda()    
    txt_data = txt_data.cuda()
    bimages = images

    netG.eval()
    netEs.eval()
    netEb.eval()

    
    # bimages = roll(images, i+1, dim=0) # for text and seg mismatched backgrounds
    np_bimages = to_numpy(bimages)
    np_segs    = np.repeat(to_numpy(segs), 3, 1) * 2 - 1

    n = 4

    vis_samples = [None for i in range(1+n)]

    vis_samples[0] = np_segs
    for i in range(n):
        # generate testing results
        segs_code = netEs(segs)
        bkgs_code = netEb(bimages[i].unsqueeze(0).repeat(batch_size,1,1,1))

        *_, f_images, z_list= netG(txt_data, txt_len, segs_code, bkgs_code)
        
        np_fakes = to_numpy(f_images)

        vis_samples[0]   = np.concatenate((np.expand_dims(np.zeros_like(np_bimages[i])-1, 0), np_segs))
        vis_samples[i+1] = np.concatenate((np.expand_dims(np_bimages[i],0), np_fakes))

    # save images
    save_images(vis_samples, save=not sample_folder == '', save_path=os.path.join(
        sample_folder, png_file), dim_ordering='th')

    # save captions
    # with open(os.path.join(sample_folder, txt_file), 'w') as f:
        # for cap in captions:
            # f.write(cap + '\n')

    print('Images and captions saved at %s' % sample_folder)