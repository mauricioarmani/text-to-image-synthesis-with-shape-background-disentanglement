import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

proj_root = '.'
sys.path.insert(0, proj_root)
data_root = os.path.join(proj_root, 'data')
model_root = os.path.join(proj_root, 'models')

from gan.data_loader import BirdsDataset
from gan.networks import Generator
from gan.networks import Discriminator
from gan.networks import ImgEncoder
from segmentation.train import Unet
from gan.proj_utils.torch_utils import roll


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gans')

    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--unet_checkpoint', type=str, default='', 
                        help='Unet checkpoint')
    parser.add_argument('--load_from_epoch', type=int, default= 0, 
                        help='load from epoch')
    parser.add_argument('--align',    type=str,  choices=['shape', 'background', 'all', 'none'], 
                        help='Which concept to align during generation.')
    parser.add_argument('--shape_noise', action='store_true',
                        help='Run with random noise for shape.')
    parser.add_argument('--background_noise', action='store_true',
                        help='Run with random noise for background.')

    args = parser.parse_args()

    model_name = args.model_name

    # NNs
    netG  = Generator(tcode_dim=512, scode_dim=1024, emb_dim=128, hid_dim=128)
    netS  = Unet()
    netEs = ImgEncoder(num_chan=1, out_dim=1024)
    netEb = ImgEncoder(num_chan=3, out_dim=1024)

    netG  = netG.cuda()
    netS  = netS.cuda()
    netEs = netEs.cuda()
    netEb = netEb.cuda()

    data_name = model_name.split('_')[-1]
    datadir = os.path.join(data_root, data_name)
    model_folder = os.path.join(model_root, model_name)
    
    print('> Loading test data ...')
    dataset    = BirdsDataset(datadir, mode='test')
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)

    ''' load model '''
    assert args.load_from_epoch != '', 'args.load_from_epoch is empty'
    G_weightspath  = os.path.join(model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))
    Es_weightspath = os.path.join(model_folder, 'Es_epoch{}.pth'.format(args.load_from_epoch))
    Eb_weightspath = os.path.join(model_folder, 'Eb_epoch{}.pth'.format(args.load_from_epoch))
    print('reload weights from {}'.format(G_weightspath))
    print('reload weights from {}'.format(Es_weightspath))
    print('reload weights from {}'.format(Eb_weightspath))

    netG.load_state_dict(torch.load(G_weightspath))
    netS.load_state_dict(torch.load(args.unet_checkpoint))
    netEs.load_state_dict(torch.load(Es_weightspath))
    netEb.load_state_dict(torch.load(Eb_weightspath))

    netG.eval()
    netS.eval()
    netEs.eval()
    netEb.eval()

    total_inter = 0
    total_union = 0
    for data in tqdm(dataloader):
        test_images, _, segs, txt_data, txt_len, chosen_captions, saveIDs, classIDs = data

        segs = segs.cuda()
        test_images = test_images.cuda()
        txt_data = txt_data.cuda()

        test_bimages = test_images

        test_sample_num = 10
        for t in range(test_sample_num):

            # alignment
            if args.align == 'shape':
                test_bimages = roll(test_images, 2, dim=0) # for text and seg mismatched backgrounds
            elif args.align == 'background':
                segs = roll(segs, 1, dim=0) # for text mismatched segmentations
            elif args.align == 'all':
                pass
            elif args.align == 'none':
                test_bimages = roll(test_images, 2, dim=0) # for text and seg mismatched backgrounds
                segs = roll(segs, 1, dim=0) # for text mismatched segmentations

            ''' Encode Segmentation'''
            segs_code = netEs(segs)
            bkgs_code = netEb(test_bimages)

            test_outputs = {}
            _, _, _, f_images, _, _ = netG(txt_data, txt_len, segs_code, bkgs_code, 
                                            shape_noise=args.shape_noise, 
                                            background_noise=args.background_noise, vs=True)
            
            f_segs = netS(f_images)


            b_f_segs = (f_segs >= 0.5)
            b_segs   = (segs >= 0.5)

            inter = (b_segs & b_f_segs).sum()
            total_inter += inter
            total_union += b_segs.sum() + b_f_segs.sum() - inter

    IOU = total_inter.float()/total_union
    print('IOU = %f' % IOU.item())