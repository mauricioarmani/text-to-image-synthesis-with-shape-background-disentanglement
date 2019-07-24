import numpy as np
import pickle
import random
import os
import scipy.misc as misc
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random

from .proj_utils.torch_utils import to_binary

class FlowersDataset(Dataset):
    def __init__(self, workdir, mode='train'):

        self.workdir = workdir
        self.mode = mode

        self.imsize = 64

        self.image_filename = '/76images.pickle'
        self.segs_filename = '/76segmentations.pickle'

        pickle_path = os.path.join(self.workdir, self.mode)

        # images
        with open(pickle_path + self.image_filename, 'rb') as f:
            images = pickle.load(f)
            self.images = np.array(images, dtype=np.uint8)

        # segmentations
        with open(pickle_path + self.segs_filename, 'rb') as f:
            segmentations = pickle.load(f, encoding='bytes')
            self.segmentations = np.array(segmentations, dtype=np.uint8)

        # names
        with open(pickle_path + '/filenames.pickle', 'rb') as f:
            self.filenames = pickle.load(f)

        self.size = len(self.images)
        self.saveIDs = np.arange(self.size)

        # class ids
        with open(pickle_path + '/class_info.pickle', 'rb') as f:
            class_id = pickle.load(f, encoding="bytes")
            self.class_id = np.array(class_id)

        # caption vectors
        self.caption_vecs = self.load_caption_vecs(self.workdir, self.class_id,
                self.filenames)
        word_vecs, len_descs = self.caption_vecs


    def load_caption_vecs(self, workdir, class_ids, filenames):
        word_vecs = torch.FloatTensor(self.size, 10, 50, 300)
        len_descs = torch.LongTensor(self.size, 10)
        for i, filename in enumerate(filenames):
            cls = 'class_'+str(class_ids[i]).zfill(5)
            data = torch.load(os.path.join(workdir, 'flowers_icml_vec', cls,
                filename + '.pth'))
            word_vecs[i] = data['word_vec']
            len_descs[i] = torch.LongTensor(data['len_desc'])
        return (word_vecs, len_descs)

    def read_captions(self, filename, class_id, randid):
        cls = 'class_'+str(class_id).zfill(5)
        cap_path = os.path.join(self.workdir, 'cvpr2016_flowers/text_c10', cls,
                filename+'.txt')
        with open(cap_path, "r") as f:
            captions = f.read().split('\n')
        return captions[randid]

    def transforms(self, image):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()
        ])
        return transform(image)

    def __getitem__(self, index):
        w_index = np.random.randint(0, self.size)

        image   = self.images[index]
        seg     = self.segmentations[index]
        w_image = self.images[w_index, :, :, :]

        if self.mode == 'test':
            seed = 0
        else:
            seed = np.random.randint(2147483647)

        random.seed(seed)
        image = self.transforms(image)
        random.seed(seed)
        seg = self.transforms(seg*255)
        w_image = self.transforms(w_image)

        image   = image * 2 - 1
        seg     = to_binary(seg)
        w_image = w_image * 2 - 1

        filename = self.filenames[index]

        if self.mode == 'test':
            randid = 0 # choose instance caption
        else:
            randid = np.random.randint(0, 10) # choose instance caption

        word_vec = self.caption_vecs[0][index][randid]
        len_desc = self.caption_vecs[1][index][randid]
        raw_captions = self.read_captions(filename, self.class_id[index], randid)

        data = [image, w_image, seg, word_vec, len_desc, raw_captions]

        if self.mode == 'test':
            data.append(self.saveIDs[index])
            data.append(self.class_id[index])

        return data

    def __len__(self):
        return self.size
