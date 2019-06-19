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

    # seed = 1231251
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)

    parser = argparse.ArgumentParser(description='Gans')

    parser.add_argument('--epoch', type=int, default=0,  
                        help='load from epoch')
    parser.add_argument('--model', type=str, default='',  
                        help='model name')
    parser.add_argument('--n_samples', type=int, default=5,  
                        help='Number of samples per embedding')
    parser.add_argument('--batch_size', type=int, default=10,  
                        help='batch_size')
    args = parser.parse_args()

    epoch = args.epoch
    model_name = args.model
    n_samples = args.n_samples
    batch_size = args.batch_size

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
    sample_folder = os.path.join(model_folder, sample_name)

    # create sample folder
    if not os.path.exists(sample_folder):
        os.mkdir(sample_folder)

    # NNs
    netG  = Generator(tcode_dim=512, scode_dim=scode_dim, emb_dim=emb_dim, hid_dim=128)
    netEs = ImgEncoder(num_chan=1, out_dim=scode_dim)
    netEb = ImgEncoder(num_chan=3, out_dim=scode_dim)

    # Dataset
    dataset = BirdsDataset(datadir, mode='train')
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
    images, _, segs, txt_data, txt_len, captions = next(iter(dataloader))

    netG.eval()
    netEs.eval()
    netEb.eval()

    segs = segs.cuda()
    txt_data = txt_data.cuda()

    # mismatch segmentations and backgrounds
    bimages = roll(images, 2, dim=0).cuda() # for text and seg mismatched backgrounds
    bsegs   = roll(segs, 2, dim=0).cuda()   # background segmentations
    segs    = roll(segs, 1, dim=0).cuda()   # for text mismatched segmentations

    # np to save
    np_segs    = np.repeat(to_numpy(segs), 3, 1) * 2 - 1
    np_images  = to_numpy(images)
    np_bimages = to_numpy(bimages)

    # generate testing results
    vis_samples = [None for i in range(n_samples + 3)]
    vis_samples[0] = np_images
    vis_samples[1] = np_bimages
    vis_samples[2] = np_segs
    
    for c in range(n_samples):

        segs_code = netEs(segs)
        bkgs_code = netEb(bimages)
        
        *_, f_images, z_list= netG(txt_data, txt_len, segs_code, bkgs_code)
        
        np_fakes = to_numpy(f_images)

        vis_samples[c+3] = np_fakes

    # save noise tensors
    with open(os.path.join(sample_folder, z_file), 'bw') as f:
        pickle.dump(z_list, f)

    # save images
    save_images(vis_samples, save=not sample_folder == '', save_path=os.path.join(
        sample_folder, png_file), dim_ordering='th')

    # save images individualy
    for i in range(batch_size):
        np_image  = np.expand_dims(np_images[i], 0)
        np_bimage = np.expand_dims(np_bimages[i], 0)
        np_seg    = np.expand_dims(np_segs[i], 0)
        np_fake   = np.expand_dims(np_fakes[i], 0)
        sample = [np_image, np_bimage, np_seg, np_fake]
        save_images(sample, save=not sample_folder == '', save_path=os.path.join(
            sample_folder, '%d.png' % i), dim_ordering='th')

    # save captions
    with open(os.path.join(sample_folder, txt_file), 'w') as f:
        for cap in captions:
            f.write(cap + '\n')

    print('Images and captions saved at %s' % sample_folder)