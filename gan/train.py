import numpy as np
import os
import sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import time

from .proj_utils.torch_utils import set_lr, to_numpy, roll, to_binary
from gan.networks import ImgEncoder


def compute_g_loss(f_logit, f_logit_c, r_labels):
    criterion  = F.mse_loss
    r_g_loss   = criterion(f_logit, r_labels)
    f_g_loss_c = criterion(f_logit_c, r_labels)
    return r_g_loss + 10*f_g_loss_c

def compute_d_loss(r_logit, r_logit_c, w_logit_c, f_logit, r_labels, f_labels):
    criterion  = F.mse_loss
    r_d_loss   = criterion(r_logit,   r_labels)
    r_d_loss_c = criterion(r_logit_c, r_labels)
    w_d_loss_c = criterion(w_logit_c, f_labels)
    f_d_loss   = criterion(f_logit,   f_labels)
    return r_d_loss + 10*r_d_loss_c + 10*w_d_loss_c + f_d_loss

def get_kl_loss(mu, logvar):
    kld = mu.pow(2).add(logvar.mul(2).exp()).add(-1).mul(0.5).add(logvar.mul(-1))
    kl_loss = torch.mean(kld)
    return kl_loss

def shape_consistency_loss(f_seg, r_seg):
    consistency = F.mse_loss(f_seg, r_seg) # l1 bug?
    return consistency

def background_consistency_loss(f_bkgrds, bkgrds, f_segs, segs):
    crit_mask = ((f_segs < 1) & (segs < 1)).float().cuda()
    l1 = F.l1_loss(f_bkgrds, bkgrds, reduction='none')
    l1_mean = (crit_mask * l1).mean()
    return l1_mean

def idt_consistency_loss(f_images, r_images):
    consistency = F.l1_loss(f_images, r_images)
    return consistency

def obj_consistency_loss(img_images, txt_images, segs):
    l1 = F.l1_loss(img_images, txt_images, reduction='none')
    l1_mean = (segs * l1).mean()
    return l1_mean

def train_gan(dataloader, model_folder, netG, netD, netS, netEs, netEb, args):
    """
    Parameters:
    ----------
    dataloader: 
        data loader. refers to fuel.dataset
    model_root: 
        the folder to save the models weights
    netG:
        Generator
    netD:
        Descriminator
    netE:
        Segmentation Encoder
    netS:
        Segmentation Network
    """

    # if args.manipulate:
    #     netEi = ImgEncoder(num_chan=3, out_dim=args.scode_dim)
    #     netEi.cuda()

    d_lr = args.d_lr
    g_lr = args.g_lr
    tot_epoch = args.maxepoch

    ''' configure optimizers '''
    optimizerD = optim.Adam(netD.parameters(), lr=d_lr, betas=(0.5, 0.999))
    paramsG = list(netG.parameters())+list(netEs.parameters())+list(netEb.parameters())

    # if args.manipulate:
    #     paramsG += list(netEi.parameters())
    
    optimizerG = optim.Adam(paramsG, lr=g_lr, betas=(0.5, 0.999))

    ''' create tensorboard writer '''
    writer = SummaryWriter(model_folder)
    
    # --- load model from  checkpoint ---
    netS.load_state_dict(torch.load(args.unet_checkpoint))
    if args.reuse_weights:
        G_weightspath = os.path.join(
            model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))
        D_weightspath = os.path.join(
            model_folder, 'D_epoch{}.pth'.format(args.load_from_epoch))
        Es_weightspath = os.path.join(
            model_folder, 'Es_epoch{}.pth'.format(args.load_from_epoch))
        Eb_weightspath = os.path.join(
            model_folder, 'Eb_epoch{}.pth'.format(args.load_from_epoch))

        netG.load_state_dict(torch.load(G_weightspath))
        netD.load_state_dict(torch.load(D_weightspath))
        netEs.load_state_dict(torch.load(Es_weightspath))
        netEb.load_state_dict(torch.load(Eb_weightspath))

        # if args.manipulate:
        #     Ei_weightspath = os.path.join(
        #         model_folder, 'Ei_epoch{}.pth'.format(args.load_from_epoch))
        #     netEi.load_state_dict(torch.load(Ei_weightspath))

        start_epoch = args.load_from_epoch + 1
        d_lr /= 2 ** (start_epoch // args.epoch_decay)
        g_lr /= 2 ** (start_epoch // args.epoch_decay) 

    else:
        start_epoch = 1

    ''' create labels '''
    r_labels = torch.FloatTensor(args.batch_size, 1).fill_(1).cuda()
    f_labels = torch.FloatTensor(args.batch_size, 1).fill_(0).cuda()
    
    # --- Start training ---
    for epoch in range(start_epoch, tot_epoch + 1):
        start_timer = time.time()
        '''decay learning rate every epoch_decay epoches'''
        if epoch % args.epoch_decay == 0:
            d_lr = d_lr/2
            g_lr = g_lr/2

            set_lr(optimizerD, d_lr)
            set_lr(optimizerG, g_lr)

        netG.train()
        netD.train()
        netEs.train()
        netEb.train()
        # if args.manipulate:
        #     netEi.train()
        netS.eval()

        for i, data in enumerate(dataloader):
            images, w_images, segs, txt_data, txt_len, _ = data

            it = epoch*len(dataloader) + i

            # to cuda
            images   = images.cuda()
            w_images = w_images.cuda()
            segs = segs.cuda()
            txt_data = txt_data.cuda()

            ''' UPDATE D '''
            for p in netD.parameters(): p.requires_grad = True
            optimizerD.zero_grad()

            if args.manipulate:
                bimages = images # for text and seg mismatched backgrounds
                bsegs   = segs   # background segmentations
            else:
                bimages = roll(images, 2, dim=0) # for text and seg mismatched backgrounds
                bsegs   = roll(segs, 2, dim=0)   # background segmentations
                segs    = roll(segs, 1, dim=0)   # for text mismatched segmentations
    
            segs_code = netEs(segs)    # segmentation encoding
            bkgs_code = netEb(bimages) # background image encoding

            mean_var, smean_var, bmean_var, f_images, z_list = netG(txt_data, txt_len, segs_code, bkgs_code)

            # if args.manipulate:
            #     img_code = netEi(images)
            #     f_images_img, _ = netG.img_forward(img_code, z_list)


            f_images_cp = f_images.data.cuda()

            r_logit, r_logit_c = netD(images,   txt_data, txt_len)
            _      , w_logit_c = netD(w_images, txt_data, txt_len)
            f_logit, _         = netD(f_images_cp, txt_data, txt_len)

            d_adv_loss = compute_d_loss(r_logit, r_logit_c, 
                                        w_logit_c, f_logit, 
                                        r_labels, f_labels)

            d_loss = d_adv_loss
            d_loss.backward()
            optimizerD.step()
            optimizerD.zero_grad()


            ''' UPDATE G '''
            for p in netD.parameters(): p.requires_grad = False  # to avoid computation
            optimizerG.zero_grad()

            f_logit, f_logit_c = netD(f_images, txt_data, txt_len)

            g_adv_loss = compute_g_loss(f_logit, f_logit_c, r_labels)
            f_segs = netS(f_images) # segmentation from Unet
            seg_consist_loss = shape_consistency_loss(f_segs, segs)
            bkg_consist_loss = background_consistency_loss(f_images, bimages, f_segs, bsegs)

            # if args.manipulate:
                # obj_consist_loss = obj_consistency_loss(f_images_img, f_images, segs)

            kl_loss  = get_kl_loss(mean_var[0], mean_var[1])   # text
            skl_loss = get_kl_loss(smean_var[0], smean_var[1]) # segmentation
            bkl_loss = get_kl_loss(bmean_var[0], bmean_var[1]) # background

            if args.manipulate:
                idt_consist_loss = idt_consistency_loss(f_images, images)
            else:
                idt_consist_loss = 0.

            g_loss = g_adv_loss \
                    + args.KL_COE * kl_loss \
                    + args.KL_COE * skl_loss \
                    + args.KL_COE * bkl_loss \
                    + args.CONSIST_COE*seg_consist_loss \
                    + args.CONSIST_COE*bkg_consist_loss \
                    + args.CONSIST_COE*idt_consist_loss

            if args.manipulate:
                g_loss += args.CONSIST_COE*obj_consist_loss

            g_loss.backward()
            optimizerG.step()
            optimizerG.zero_grad()


            # --- visualize train samples----
            if it % args.verbose_per_iter == 0:
                writer.add_images('txt',         (images[:args.n_plots]+1)/2, it)
                writer.add_images('background', (bimages[:args.n_plots]+1)/2, it)
                writer.add_images('segmentation',   segs[:args.n_plots].repeat(1,3,1,1), it)
                writer.add_images('generated', (f_images[:args.n_plots]+1)/2, it)
                # if args.manipulate:
                #     writer.add_images('generated_img', (f_images_img[:args.n_plots]+1)/2, it)
                #     writer.add_scalar('obj_consist_loss', to_numpy(obj_consist_loss).mean(), it)
                writer.add_scalar('g_lr', g_lr, it)
                writer.add_scalar('d_lr', g_lr, it)
                writer.add_scalar('g_loss', to_numpy(g_loss).mean(), it)
                writer.add_scalar('d_loss', to_numpy(d_loss).mean(), it)
                writer.add_scalar('imkl_loss', to_numpy(kl_loss).mean(), it)
                writer.add_scalar('segkl_loss', to_numpy(skl_loss).mean(), it)
                writer.add_scalar('bkgkl_loss', to_numpy(bkl_loss).mean(), it)
                writer.add_scalar('seg_consist_loss', to_numpy(seg_consist_loss).mean(), it)
                writer.add_scalar('bkg_consist_loss', to_numpy(bkg_consist_loss).mean(), it)
                if args.manipulate:
                    writer.add_scalar('idt_consist_loss', to_numpy(idt_consist_loss).mean(), it)


        # --- save weights ---
        if epoch % args.save_freq == 0:

            netG  = netG.cpu()
            netD  = netD.cpu()
            netEs = netEs.cpu()
            netEb = netEb.cpu()

            torch.save(netD.state_dict(),  os.path.join(model_folder, 'D_epoch{}.pth'.format(epoch)))
            torch.save(netG.state_dict(),  os.path.join(model_folder, 'G_epoch{}.pth'.format(epoch)))
            torch.save(netEs.state_dict(), os.path.join(model_folder, 'Es_epoch{}.pth'.format(epoch)))
            torch.save(netEb.state_dict(), os.path.join(model_folder, 'Eb_epoch{}.pth'.format(epoch)))
            
            print('save weights at {}'.format(model_folder))
            netD  = netD.cuda()
            netG  = netG.cuda()
            netEs = netEs.cuda()
            netEb = netEb.cuda()

            # if args.manipulate:
            #     netEi = netEi.cpu()
            #     torch.save(netEi.state_dict(), os.path.join(model_folder, 'Ei_epoch{}.pth'.format(epoch)))
            #     netEi = netEi.cuda()

        end_timer = time.time() - start_timer
        print('epoch {}/{} finished [time = {}s] ...'.format(epoch, tot_epoch, end_timer))

    writer.close()