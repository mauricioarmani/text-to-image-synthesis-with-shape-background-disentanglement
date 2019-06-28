import numpy as np
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import h5py
import re
import scipy

from .proj_utils.local_utils import mkdirs, imresize_shape
from .proj_utils.torch_utils import to_torch, to_binary, to_numpy, roll


def test_gan(dataloader, save_root, model_folder, model_marker, netG, netEs, netEb, args):
    highest_res  = 64

    h5_filename = model_marker+'_align_%s' % args.align

    if args.shape_noise:
        h5_filename += '_shape_noise'
    if args.background_noise:
        h5_filename += '_background_noise'

    h5_filename += '.h5'

    save_h5      = os.path.join(save_root, h5_filename)
    org_h5path   = os.path.join(save_root, 'original.h5')

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

    num_examples = len(dataloader.dataset)
    total_number = num_examples * args.test_sample_num

    all_choosen_caption = []
    org_file_not_exists = not os.path.exists(org_h5path)

    if org_file_not_exists:
        org_h5   = h5py.File(org_h5path,'w')
        org_dset = org_h5.create_dataset('output_{}'.format(highest_res), 
                                                shape=(num_examples, highest_res, highest_res, 3), 
                                                dtype=np.uint8)
    else:
        org_dset = None
        org_emb_dset = None

    with h5py.File(save_h5,'w') as h5file:
        
        start_count = 0
        data_count = {}
        dset = {}
        gen_samples = []
        img_samples = []
        vis_samples = {}
        tmp_samples = {}
        init_flag = True

        netG.eval()
        netEs.eval()
        netEb.eval()

        for data in dataloader:
            test_images, _, segs, txt_data, txt_len, chosen_captions, saveIDs, classIDs = data

            segs = segs.cuda()
            test_images = test_images.cuda()
            txt_data = txt_data.cuda()

            test_bimages = test_images

            np_test_images = to_numpy(test_images)

            this_batch_size =  np_test_images.shape[0]

            all_choosen_caption.extend(chosen_captions)    
            if org_dset is not None:
                org_dset[start_count:start_count+this_batch_size] = ((np_test_images + 1) * 127.5 ).astype(np.uint8)
                org_emb_dset[start_count:start_count+this_batch_size] = test_embeddings_list[0]

            start_count += this_batch_size
            
            for t in range(args.test_sample_num):

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
                _, _, _, fake_images, _ = netG(txt_data, txt_len, segs_code, bkgs_code, 
                                                shape_noise=args.shape_noise, background_noise=args.background_noise)

                test_outputs['output_64'] = fake_images

                if  t == 0: 
                    if init_flag is True:
                        dset['saveIDs']   = h5file.create_dataset('saveIDs', shape=(total_number,), dtype=np.int64)
                        dset['classIDs']  = h5file.create_dataset('classIDs', shape=(total_number,), dtype=np.int64)

                        for k in test_outputs.keys():
                            vis_samples[k] = [None for i in range(args.test_sample_num + 1)] # +1 to fill real image
                            img_shape = test_outputs[k].size()[2::]
    
                            dset[k] = h5file.create_dataset(k, shape=(total_number,)+ img_shape + (3,), dtype=np.uint8)
                            data_count[k] = 0

                    init_flag = False    
                
                for typ, img_val in test_outputs.items():
                    cpu_data = img_val.cpu().data.numpy()
                    row, col = cpu_data.shape[2],cpu_data.shape[3] 
                    if t==0:
                        this_reshape = imresize_shape(np_test_images,  (row, col)) 
                        this_reshape = this_reshape * (2. / 255) - 1.
                        
                        # this_reshape = this_reshape.transpose(0, 3, 1, 2)
                        vis_samples[typ][0] = this_reshape

                    vis_samples[typ][t+1] = cpu_data
                    bs = cpu_data.shape[0]
                    
                    start = data_count[typ]
                    this_sample = ((cpu_data + 1) * 127.5 ).astype(np.uint8)
                    this_sample = this_sample.transpose(0, 2, 3, 1)

                    dset[typ][start: start + bs] = this_sample
                    dset['saveIDs'][start: start + bs] = saveIDs
                    dset['classIDs'][start: start + bs] = classIDs
                    data_count[typ] = start + bs
            
            print('saved files [sample {}/{}]: '.format(start_count, num_examples), data_count)  
            
        caption_array = np.array(all_choosen_caption, dtype=object)
        string_dt = h5py.special_dtype(vlen=str)
        h5file.create_dataset("captions", data=caption_array, dtype=string_dt)
        if org_dset is not None:
            org_h5.close()