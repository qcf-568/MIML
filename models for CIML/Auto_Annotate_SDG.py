#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: liuyaqi
"""
import os
import cv2
import random
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import numpy as np
import time
import logging
import argparse
from PIL import Image
from tqdm import tqdm
import albumentations as A
import torch.distributed as dist
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import safm_convb as safm
parser = argparse.ArgumentParser()
parser.add_argument('--nm', type=str, default='ori')
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--pth', type=str, default='SAFM.pth')
parser.add_argument('--thres', type=float, default=0.5)
parser.add_argument('--numw', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--input_scale', type=int, default=512)
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
args = parser.parse_args()

class CVPR24EvalDataset(Dataset):
    def __init__(self, roots, img_dir, sz=512, fan=False):
        self.fan = fan
        self.roots = os.path.join(roots, img_dir)
        '''
        Dir strucure in self.roots:
        |
        self.roots
            |
            |---dir1
            |     |----0.jpg (SDG authentic image)
            |     |----1.jpg (SDG manipulated image)
            |
            |---dir2
            |     |----0.jpg (SDG authentic image)
            |     |----1.jpg (SDG manipulated image)
            |
        .........
        '''
        self.indexs = [os.path.join(self.roots, x) for x in os.listdir(self.roots)]
        self.indexs.sort()
        self.lens = len(self.indexs)
        self.tsr = ToTensorV2()
        self.lbl = torch.FloatTensor([1])
        self.rsz = torchvision.transforms.Compose([torchvision.transforms.Resize((sz,sz))])
        self.toctsr =torchvision.transforms.Compose([torchvision.transforms.Resize((sz, sz)), torchvision.transforms.Normalize(mean=((0.485, 0.455, 0.406)), std=((0.229, 0.224, 0.225)))])

    def __len__(self):
        return self.lens

    def __getitem__(self, idx):
        this_r = self.indexs[idx]
        img1 = self.toctsr(self.tsr(image=cv2.cvtColor(cv2.imread(os.path.join(this_r, '0.jpg')), cv2.COLOR_BGR2RGB))['image'].float()/255.0)
        img2 = self.toctsr(self.tsr(image=cv2.cvtColor(cv2.imread(os.path.join(this_r, '1.jpg')), cv2.COLOR_BGR2RGB))['image'].float()/255.0)
        return (img1, img2, this_r.split('/')[-1])

test_data = CVPR24EvalDataset('./', 'SDG')
test_loader = DataLoader(dataset=test_data, batch_size=1, num_workers=4)

model = safm.SAFM(2, 512)
model = model.cuda()
model = nn.DataParallel(model)
loader = torch.load(args.pth, map_location='cpu')
model.load_state_dict(loader)
model.eval()


if not os.path.exists('SDG_preds'):
    os.makedirs('SDG_preds')


with torch.no_grad():
    ious = []
    ps = []
    rs = []
    fs = []
    for (im1, im2, fnm) in tqdm(test_loader):
        im1 = im1.cuda()
        im2 = im2.cuda()
        _, pred, _, _ = model(im1, im2)
        _, pred2, _, _ = model(im1, torch.flip(im2, [2]))
        pred2 = torch.flip(pred2, [2])

        _, pred3, _, _ = model(im1, torch.flip(im2, [3]))
        pred3 = torch.flip(pred3, [3])

        preds = F.softmax((pred+pred2+pred3) ,dim=1)[:,1:2].squeeze().cpu().numpy()
        s1 = (preds>(1/16)).sum()
        s2 = (preds>(15/16)).sum()
        if (s2/(s1+1e-6)>0.5):
            cv2.imwrite('SDG_preds/'+fnm[0]+'.png', (preds*255).astype(np.uint8))


                
                
        
        
        
