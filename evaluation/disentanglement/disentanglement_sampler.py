import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader

proj_root = '.'
sys.path.insert(0, proj_root)
data_root = os.path.join(proj_root, 'data')
model_root = os.path.join(proj_root, 'models')
disent_root = os.path.join(proj_root, 'evaluation', 'disentanglement')

from gan.data_loader import BirdsDataset
from gan.networks import Generator
from gan.networks import ImgEncoder
from gan.proj_utils.torch_utils import roll

from gan.train import train_gan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gans')

    parser.add_argument('--model_name', type=str, default=None)
    parser.add_argument('--load_from_epoch', type=int, default= 0, 
                        help='load from epoch')
    parser.add_argument('--idx', type=int, default=0, 
                        help='idx')

    args = parser.parse_args()

    model_name = args.model_name
    idx = args.idx

    # NNs
    netG  = Generator(tcode_dim=512, scode_dim=1024, emb_dim=128, hid_dim=128)
    netEs = ImgEncoder(num_chan=1, out_dim=1024)
    netEb = ImgEncoder(num_chan=3, out_dim=1024)

    netG  = netG.cuda()
    netEs = netEs.cuda()
    netEb = netEb.cuda()

    data_name = model_name.split('_')[-1]
    datadir = os.path.join(data_root, data_name)
    model_folder = os.path.join(model_root, model_name)
    
    print('> Loading test data ...')
    dataset    = BirdsDataset(datadir, mode='test')
    batch_size = 20
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    ''' load model '''
    assert args.load_from_epoch != '', 'args.load_from_epoch is empty'
    G_weightspath  = os.path.join(model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))
    Es_weightspath = os.path.join(model_folder, 'Es_epoch{}.pth'.format(args.load_from_epoch))
    Eb_weightspath = os.path.join(model_folder, 'Eb_epoch{}.pth'.format(args.load_from_epoch))
    print('reload weights from {}'.format(G_weightspath))
    print('reload weights from {}'.format(Es_weightspath))
    print('reload weights from {}'.format(Eb_weightspath))

    netG.load_state_dict(torch.load(G_weightspath))
    netEs.load_state_dict(torch.load(Es_weightspath))
    netEb.load_state_dict(torch.load(Eb_weightspath))

    netG.eval()
    netEs.eval()
    netEb.eval()

    total_inter = 0
    total_union = 0

    images, _, segs, txt_code, txt_len, _, _, _ = iter(dataloader).next()

    segs = segs.cuda()
    images = images.cuda()

    txt_code  = txt_code.cuda()
    segs_code = netEs(segs)
    bkgs_code = netEb(images)

    _, _, _, f_images, z_list = netG(txt_code, txt_len, segs_code, bkgs_code)

    z_t, z_s, z_b = z_list

    samples_folder = os.path.join(disent_root, 'samples%d' % idx)
    if not os.path.exists(samples_folder):
        os.makedirs(samples_folder)

    for t, zt in enumerate(z_t):
        for s, zs in enumerate(z_s):
            for b, zb in enumerate(z_b):
                z = [zt.unsqueeze(0), zs.unsqueeze(0), zb.unsqueeze(0)]
                f_image, _ = netG(z_list=z)
                img_path = os.path.join(samples_folder, '%d_%d_%d.pt' % (t, s, b))
                torch.save(f_image, img_path)

    segs_path = os.path.join(samples_folder, 'segs.pt')
    torch.save(segs, segs_path)