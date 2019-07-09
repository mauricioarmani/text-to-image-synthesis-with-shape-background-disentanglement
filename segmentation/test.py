import pickle
import os
import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.nn import Module
import argparse


class Dataset(data.Dataset):
    def __init__(self, imsize, datapath='../data', dataset='birds'):
        imgfile = '%s/test/76images.pickle' % dataset
        segfile = '%s/test/76segmentations.pickle' % dataset
        pickle_img = os.path.join(datapath, imgfile)
        pickle_seg = os.path.join(datapath, segfile)

        self.imsize = imsize

        with open(pickle_img,'rb') as f:
            self.images = np.asarray(pickle.load(f))

        with open(pickle_seg,'rb') as f:
            self.segs = np.asarray(pickle.load(f, encoding='bytes'))

    def transform(self, img, seg):
        img_chnn = img.shape[-1]
        seg_chnn = seg.shape[-1]
        transformed_img = np.zeros([self.imsize, self.imsize, img_chnn])
        transformed_seg = np.zeros([self.imsize, self.imsize, seg_chnn])

        ori_size = img.shape[1]

        h1 = int(np.floor((ori_size - self.imsize) * 0.5))
        w1 = int(np.floor((ori_size - self.imsize) * 0.5))
        cropped_img = img[w1: w1 + self.imsize, h1: h1 + self.imsize, :]
        cropped_seg = seg[w1: w1 + self.imsize, h1: h1 + self.imsize, :]

        transformed_img = cropped_img
        transformed_seg = cropped_seg

        return transformed_img, transformed_seg

    def __getitem__(self, index):
        img, seg = self.transform(self.images[index], self.segs[index])
        tensor_img = torch.cuda.FloatTensor(img/255.)
        tensor_seg = torch.cuda.FloatTensor(seg/255.)
        return tensor_img, tensor_seg

    def __len__(self):
        return self.images.shape[0]


class DownConv(Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(DownConv, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_bn = nn.GroupNorm(32, out_feat)
        self.conv1_drop = nn.Dropout2d(drop_rate)

        self.conv2 = nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv2_bn = nn.GroupNorm(32, out_feat)
        self.conv2_drop = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        return x


class UpConv(Module):
    def __init__(self, in_feat, out_feat, drop_rate=0.4, bn_momentum=0.1):
        super(UpConv, self).__init__()
        self.downconv = DownConv(in_feat, out_feat, drop_rate, bn_momentum)

    def forward(self, x, y):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x


class Unet(Module):
    """A reference U-Net model.

    .. seealso::
        Ronneberger, O., et al (2015). U-Net: Convolutional
        Networks for Biomedical Image Segmentation
        ArXiv link: https://arxiv.org/abs/1505.04597
    """
    def __init__(self, drop_rate=0.4, bn_momentum=0.1):
        super(Unet, self).__init__()

        #Downsampling path
        self.conv1 = DownConv(3, 64, drop_rate, bn_momentum)
        self.mp1 = nn.MaxPool2d(2)

        self.conv2 = DownConv(64, 128, drop_rate, bn_momentum)
        self.mp2 = nn.MaxPool2d(2)

        self.conv3 = DownConv(128, 256, drop_rate, bn_momentum)
        self.mp3 = nn.MaxPool2d(2)

        # Bottom
        self.conv4 = DownConv(256, 256, drop_rate, bn_momentum)

        # Upsampling path
        self.up1 = UpConv(512, 256, drop_rate, bn_momentum)
        self.up2 = UpConv(384, 128, drop_rate, bn_momentum)
        self.up3 = UpConv(192, 64, drop_rate, bn_momentum)

        self.conv9 = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.mp1(x1)

        x3 = self.conv2(x2)
        x4 = self.mp2(x3)

        x5 = self.conv3(x4)
        x6 = self.mp3(x5)

        # Bottom
        x7 = self.conv4(x6)

        # Up-sampling
        x8 = self.up1(x7, x5)
        x9 = self.up2(x8, x3)
        x10 = self.up3(x9, x1)

        x11 = self.conv9(x10)
        preds = self.sigmoid(x11)

        return preds


if __name__ == '__main__':

    device = torch.device("cuda:0")

    test_dataset = Dataset(imsize=64)
    dataset_size = len(test_dataset)

    testloader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    model = Unet().to(device)
    model.load_state_dict(torch.load('checkpoints/checkpoint590.pt'))
    model.eval()

    total_inter = 0
    total_union = 0
    for img, seg in testloader:

        p_seg = model(img.permute(0,3,1,2)).permute(0,2,3,1)
        
        b_p_seg = (p_seg >= 0.5)
        b_seg = (seg >= 0.5)

        inter = (b_seg & b_p_seg).sum()
        total_inter += inter
        total_union += b_seg.sum() + b_p_seg.sum() - inter

    IOU = total_inter.float()/total_union
    print('IOU = %f' % IOU.item())