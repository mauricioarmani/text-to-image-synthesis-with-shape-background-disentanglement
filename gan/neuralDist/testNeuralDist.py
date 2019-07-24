import numpy as np
import os, sys
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
from torch.autograd import Variable
from torch.nn import Parameter
import torchvision.transforms as transforms

from torch.nn.utils import clip_grad_norm
from ..proj_utils.plot_utils import *
from ..proj_utils.torch_utils import *
from ..proj_utils.local_utils import Indexflow, IndexH5

import scipy
import time, json
import random, h5py

TINY = 1e-8


def get_trans(img_encoder):
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])
    img_tensor_list = []
    
    trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=img_encoder.mean, std=img_encoder.std),
                ])
    return trans

def _process(inputs):
    this_img, trans =inputs
    this_crop = scipy.misc.imresize(this_img, (299, 299))
    this_img_tensor = trans(this_crop)
    return this_img_tensor

def pre_process(images, trans=None):
    
    images = (images + 1) /2 * 255
    images = images.transpose(0,2,3,1)
    bs = images.shape[0]
    img_tensor_list = []
    targets = [_process((images[idx], trans)) for idx in range(bs)]
    
    for idx in range(bs):
        this_img_tensor = targets[idx]
        img_tensor_list.append(this_img_tensor)

    img_tensor_all = torch.stack(img_tensor_list, 0)
    
    return img_tensor_all


def test_nd(h5_path, weight_root, img_encoder, vs_model, args, target_resolution=256):
    h5_folder, h5_name = os.path.split(h5_path)
    h5_name_noext = os.path.splitext(h5_name)[0]
    result_path  = os.path.join(h5_folder, h5_name_noext+"_epoch_{}_neu_dist.json".format(args.load_from_epoch))
    print("{} exists or not: ".format(h5_path), os.path.exists(h5_path))
    with h5py.File(h5_path,'r') as h5_data:
        all_embeddings = h5_data["embedding"]
        
        all_keys = ['output_'+str(target_resolution)]
        trans_func = get_trans(img_encoder)

        ''' load model '''
        weightspath = os.path.join(weight_root, 'W_epoch{}.pth'.format(args.load_from_epoch))
        weights_dict = torch.load(weightspath, map_location=lambda storage, loc: storage)
        print('reload weights from {}'.format(weightspath))
        vs_model.load_state_dict(weights_dict)# 12)

        vs_model.eval()
        img_encoder.eval()
        all_results = {}
        # for this_key in all_keys:
        for this_key in ['output_64']:
            this_images = h5_data[this_key]
            num_imgs = this_images.shape[0]
            all_distance = 0
            all_cost_list = []
            print("Now processing {}".format(this_key))
            n_processed = 0
            for thisInd in Indexflow(num_imgs, args.batch_size, random=False): #
                this_batch    = IndexH5(this_images, thisInd) 
                np_embeddings = IndexH5(all_embeddings, thisInd)

                img_299 = pre_process(this_batch, trans_func)

                with torch.no_grad():
                    embeddings = torch.from_numpy(np_embeddings.astype(np.float32)).cuda()
                    img_299    = img_299.cuda()

                # print(img_299[1])
                img_feat = img_encoder(img_299)
                # print(img_feat[0])
                # print(img_feat[1])
                img_feat = img_feat.squeeze(-1).squeeze(-1)

                with torch.no_grad():
                    img_feat = img_feat.clone()

                sent_emb, img_emb = vs_model(embeddings, img_feat)
                cost = torch.sum(img_emb*sent_emb, 1, keepdim=False)
                cost_val = cost.cpu().data.numpy()
                all_cost_list.append(cost_val)

                n_processed += args.batch_size
                if n_processed % (50*args.batch_size) == 0:
                    print('{}/{} processed'.format(n_processed, num_imgs))
                
            all_cost = np.concatenate(all_cost_list, 0)    
            cost_mean = float(np.mean(all_cost))
            cost_std  = float(np.std(all_cost))

            all_results[this_key] = {"mean":cost_mean, "std":cost_std}

        print(all_results)
        print('save VS-Similarity results at', result_path)   
        with open(result_path, 'w') as f:
            json.dump(all_results, f)
        