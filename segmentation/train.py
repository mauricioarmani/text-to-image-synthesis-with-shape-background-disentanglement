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
    def __init__(self, imsize, datapath='../../data', dataset='birds'):
        print('Loading data...')
        imgfile = '%s/train/76images.pickle' % dataset
        segfile = '%s/train/76segmentations.pickle' % dataset
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

        h1 = int( np.floor((ori_size - self.imsize) * np.random.random()) )
        w1 = int( np.floor((ori_size - self.imsize) * np.random.random()) )
        cropped_img = img[w1: w1 + self.imsize, h1: h1 + self.imsize, :]
        cropped_seg = seg[w1: w1 + self.imsize, h1: h1 + self.imsize, :]

        if np.random.random() > 0.5:
            transformed_img = np.fliplr(cropped_img)
            transformed_seg = np.fliplr(cropped_seg)
        else:
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


def loss_function(pred, target):
    criteria = nn.MSELoss()
    return criteria(pred, target)


def dice_loss(input, target):
    """Dice loss.
    :param input: The input (predicted)
    :param target:  The target (ground truth)
    :returns: the Dice score between 0 and 1.
    """
    eps = 0.0001

    iflat = input.view(-1)
    tflat = target.view(-1)

    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()

    dice = (2.0 * intersection + eps) / (union + eps)

    return - dice


def train(model, device, trainloader, optimizer, epoch, writer):
    model.train()

    for itr, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)

        data = data.permute(0,3,1,2)
        target = target.permute(0,3,1,2)

        optimizer.zero_grad()
        
        pred = model(data)

        loss = dice_loss(pred, target)
        
        loss.backward()
        optimizer.step()

        writer.add_scalar('train/loss', loss.data.item(), epoch*len(trainloader) + itr)


def valid(model, device, validloader, epoch, writer):
    model.eval()
    valid_loss = 0
    with torch.no_grad():
        for data, target in validloader:
            data, target = data.to(device), target.to(device)

            data = data.permute(0,3,1,2)
            target = target.permute(0,3,1,2)

            pred = model(data)

            valid_loss += dice_loss(pred, target)
            
            optimizer.step()

    writer.add_images('val/Image', data, epoch)
    writer.add_images('val/Pred', pred.repeat(1,3,1,1), epoch)
    writer.add_scalar('val/loss', valid_loss.data.item(), epoch)
    return valid_loss


if __name__ == '__main__':
    from tensorboardX import SummaryWriter

    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse_weights',   action='store_true',
                        default=False, help='continue from last checkout point')
    parser.add_argument('--model_name', type=str,      default=None)

    args = parser.parse_args()

    device = torch.device("cuda:0")

    batch_size = 32
    n_epochs = 600
    imsize = 64

    dataset = Dataset(imsize=imsize)
    dataset_size = len(dataset)

    train_size = int(0.8 * dataset_size)
    valid_size = int(0.1 * dataset_size)
    test_size  = dataset_size - valid_size - train_size

    train_dataset, valid_dataset, test_dataset = data.random_split(dataset, lengths=[train_size, valid_size, test_size])

    trainloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validloader = data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    model = Unet().to(device)

    start_epoch = 0

    if args.reuse_weights == True:
        best_model = torch.load(args.model_name)
        model.load_state_dict(best_model)
        start_epoch = int(args.model_name[-6:-3])

    optimizer = Adam(model.parameters(), lr=1e-4)

    writer = SummaryWriter()

    curr_loss = np.inf
    for epoch in range(start_epoch, n_epochs):
        print('Epoch %d' % epoch)
        train(model, device, trainloader, optimizer, epoch, writer)
        valid_loss = valid(model, device, validloader, epoch, writer)

        if valid_loss < curr_loss:
            curr_loss = valid_loss
            checkpoint_path = "checkpoints/checkpoint%.3d.pt" % epoch
            torch.save(model.state_dict(), checkpoint_path)
            print("Saved checkpoint at %s" % checkpoint_path)