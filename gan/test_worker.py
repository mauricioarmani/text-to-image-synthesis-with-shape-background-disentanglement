import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
import random

proj_root = '.'
sys.path.insert(0, proj_root)
data_root = os.path.join(proj_root, 'data')
model_root = os.path.join(proj_root, 'models')
save_root  =  os.path.join(proj_root, 'results')

from gan.data_loader import BirdsDataset
from gan.data_loader_flowers import FlowersDataset
from gan.networks import Generator
from gan.networks import ImgEncoder

from gan.test import test_gan


if  __name__ == '__main__':

    seed = 354168
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    parser = argparse.ArgumentParser(description = 'Gans')    

    parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                        help='batch size.')
    parser.add_argument('--load_from_epoch', type=int, default= 0, 
                        help='load from epoch')
    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--dataset',    type=str,      default= None, 
                        help='which dataset to use [birds or flowers]') 
    parser.add_argument('--test_sample_num', type=int, default=None, 
                        help='The number of runs for each embeddings')
    parser.add_argument('--save_visual_results', action='store_true',
                        help='if save visual results in folders')
    parser.add_argument('--random_seg_noise', action='store_true',
                        help='if save visual results in folders')
    parser.add_argument('--align',    type=str,  choices=['shape', 'background', 'all', 'none'], 
                        help='Which concept to align during generation.')
    parser.add_argument('--shape_noise', action='store_true',
                        help='Run with random noise for shape.')
    parser.add_argument('--background_noise', action='store_true',
                        help='Run with random noise for background.')
    parser.add_argument('--manipulate',   action='store_true',
                        default=False, help='Framework for image manipulation.')

    args = parser.parse_args()
    
    # NNs
    netG  = Generator(tcode_dim=512, scode_dim=1024, emb_dim=128, hid_dim=128)
    netEs = ImgEncoder(num_chan=1, out_dim=1024)
    netEb = ImgEncoder(num_chan=3, out_dim=1024)

    netG.cuda()
    netEs.cuda()
    netEb.cuda()

    data_name = args.dataset
    datadir = os.path.join(data_root, data_name)

    print('> Loading test data ...')
    if args.dataset == 'birds':
        dataset = BirdsDataset(datadir, mode='test')
    elif args.dataset == 'flowers':
        dataset = FlowersDataset(datadir, mode='test')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # create results folder
    model_name = args.model_name  
    model_folder = os.path.join(model_root, model_name)

    # creade model folder
    model_marker = model_name + '_G_epoch_{}'.format(args.load_from_epoch)

    test_gan(dataloader, save_root, model_folder, model_marker, netG, netEs, netEb, args)
