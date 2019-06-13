import numpy as np
import os
import torch
from PIL import Image, ImageDraw, ImageFont
import h5py
import re
import scipy

from .proj_utils.local_utils import mkdirs, imresize_shape
from .proj_utils.torch_utils import to_torch, to_binary, to_numpy


def test_gan(dataloader, save_root, model_folder, model_marker, netG, netE, args):
    # test_sampler = dataset.next_batch_test
    highest_res  = 64
    
    save_folder  = os.path.join(save_root, model_marker) # to be defined in the later part
    save_h5      = os.path.join(save_root, model_marker+'.h5')
    org_h5path   = os.path.join(save_root, 'original.h5')

    ''' create model folder '''
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    ''' load model '''
    assert args.load_from_epoch != '', 'args.load_from_epoch is empty'
    G_weightspath = os.path.join(model_folder, 'G_epoch{}.pth'.format(args.load_from_epoch))
    E_weightspath = os.path.join(model_folder, 'E_epoch{}.pth'.format(args.load_from_epoch))
    print('reload weights from {}'.format(G_weightspath))
    print('reload weights from {}'.format(E_weightspath))

    netG.load_state_dict(torch.load(G_weightspath))
    netE.load_state_dict(torch.load(E_weightspath))

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
        netE.eval()

        for data in dataloader:
            test_images, _, segs, txt_data, txt_len, chosen_captions, saveIDs, classIDs = data
            test_images = to_numpy(test_images)

            this_batch_size =  test_images.shape[0]

            all_choosen_caption.extend(chosen_captions)    
            if org_dset is not None:
                org_dset[start_count:start_count+this_batch_size] = ((test_images + 1) * 127.5 ).astype(np.uint8)
                org_emb_dset[start_count:start_count+this_batch_size] = test_embeddings_list[0]

            start_count += this_batch_size
            
            for t in range(args.test_sample_num):
                
                # segs = to_torch(np_segs).cuda()
                segs = to_binary(segs).cuda()

                ''' Encode Segmentation'''
                segs_code = netE(segs)

                txt_data = txt_data.cuda()

                test_outputs = {}
                # fake_images, *_ = netG(txt_data, txt_len, segs_code, random_seg_noise=args.random_seg_noise)
                fake_images, *_ = netG(txt_data, txt_len, segs_code)
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
                        this_reshape = imresize_shape(test_images,  (row, col)) 
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
            
            if args.save_visual_results:
                save_super_images(vis_samples, chosen_captions, this_batch_size, save_folder, saveIDs, classIDs)

            print('saved files [sample {}/{}]: '.format(start_count, num_examples), data_count)  
            
        caption_array = np.array(all_choosen_caption, dtype=object)
        string_dt = h5py.special_dtype(vlen=str)
        h5file.create_dataset("captions", data=caption_array, dtype=string_dt)
        if org_dset is not None:
            org_h5.close() 

#-----------------------------------------------------------------------------------------------#
#  drawCaption and save_super_images is modified from https://github.com/hanzhanggit/StackGAN   #
#-----------------------------------------------------------------------------------------------#
def drawCaption(img, caption, level=['output 64', 'output 128', 'output 256']):
    img_txt = Image.fromarray(img)
    # get a font
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 40)
    # get a drawing context
    d = ImageDraw.Draw(img_txt)

    # draw text, half opacity
    for idx, this_level in enumerate(level):
        d.text((10, 256 + idx * 256), this_level, font=fnt, fill=(255, 255, 255, 255))

    idx = caption.find(' ', 60)
    if idx == -1:
        d.text((256, 10), caption, font=fnt, fill=(255, 255, 255, 255))
    else:
        cap1 = caption[:idx]
        cap2 = caption[idx+1:]
        d.text((256, 10), cap1, font=fnt, fill=(255, 255, 255, 255))
        d.text((256, 60), cap2, font=fnt, fill=(255, 255, 255, 255))

    return img_txt

def save_super_images(vis_samples, captions_batch, batch_size, save_folder, saveIDs, 
                                    classIDs, max_sample_num=8, save_single_img=True):
    save_folder_caption = os.path.join(save_folder, 'with_captions')
    save_folder_images  = os.path.join(save_folder, 'images')
    
    dst_shape = (0,0)
    all_row = []
    level = []
    for typ, img_list in vis_samples.items():
        this_shape = img_list[0].shape[2::] # bs, 3, row, col
        if this_shape[0] > dst_shape[0]:
            dst_shape = this_shape
        level.append(typ)

    valid_caption = []
    valid_IDS = []
    valid_classIDS = []
    for j in range(batch_size):
        if not re.search('[a-zA-Z]+', captions_batch[j][0]):
            print("Not valid caption? :",  captions_batch[j])
            continue
        else:  
            valid_caption.append(captions_batch[j])
            valid_IDS.append(saveIDs[j])
            valid_classIDS.append(classIDs[j])

    for typ, img_list in vis_samples.items(): 
        img_tensor = np.stack(img_list, 1) # N * T * 3 *row*col
        img_tensor = img_tensor.transpose(0,1,3,4,2)
        img_tensor = (img_tensor + 1.0) * 127.5
        img_tensor = img_tensor.astype(np.uint8)

        this_img_list = []

        batch_size  = img_tensor.shape[0]
        batch_all = []
        for bidx in range(batch_size):
            if save_single_img:
                this_folder_id = os.path.join(save_folder_images, '{}_{}'.format(valid_classIDS[bidx], valid_IDS[bidx]))
                mkdirs([this_folder_id])

            if not re.search('[a-zA-Z]+', captions_batch[j]):
                continue
            padding = np.zeros(dst_shape + (3,), dtype=np.uint8)
            this_row = [padding]
            # First row with up to 8 samples
            for tidx in range(img_tensor.shape[1]):
                this_img  = img_tensor[bidx][tidx]
                
                re_sample = imresize_shape(this_img, dst_shape)
                if tidx <= max_sample_num:
                    this_row.append(re_sample)  

                if save_single_img:
                    scipy.misc.imsave(os.path.join(this_folder_id, '{}_copy_{}.jpg'.format(typ, tidx)),  re_sample)
                
            this_row = np.concatenate(this_row, axis=1) # row, col*T, 3
            batch_all.append(this_row)
        batch_all = np.stack(batch_all, 0) # bs*row*colT*3 
        all_row.append(batch_all)

    all_row = np.stack(all_row, 0) # n_type * bs * shape    
    
    batch_size = len(valid_IDS) 
    
    mkdirs([save_folder_caption, save_folder_images])
    for idx in range(batch_size):
        this_select = all_row[:, idx] # ntype*row*col
        
        ntype, row, col, chn = this_select.shape
        superimage = np.reshape(this_select, (-1, col, chn) )  # big_row, col, 3

        top_padding = np.zeros((128, superimage.shape[1], 3))
        superimage =\
            np.concatenate([top_padding, superimage], axis=0)
            
        save_path = os.path.join(save_folder_caption, '{}_{}.png'.format(valid_classIDS[idx], valid_IDS[idx]) )    
        superimage = drawCaption(np.uint8(superimage), valid_caption[idx], level)
        scipy.misc.imsave(save_path, superimage)