import os
import cv2
import math
import torch
import numpy as np
import torch.nn as nn
import logging
from tqdm import tqdm
import torch.optim as optim
import torch.distributed as dist
import random
import pickle
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', type=str)
parser.add_argument('--model_name', type=str, default='cls')
parser.add_argument('--att', type=str, default='None')
parser.add_argument('--num', type=str, default='1')
parser.add_argument('--n_class', type=int, default=2)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--es', type=int, default=0)
parser.add_argument('--ep', type=int, default=10)
parser.add_argument('--xk', type=int, default=0)
parser.add_argument('--numw', type=int, default=16)
parser.add_argument('--load', type=int, default=0)
parser.add_argument('--pilt', type=int, default=0)
parser.add_argument('--base', type=int, default=1)
parser.add_argument('--lr_base', type=float, default=3e-4)
parser.add_argument('--cp', type=float, default=1.0)
parser.add_argument('--mode', type=str, default='0123')
parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--adds', type=str, default='123')
parser.add_argument('--lossw', type=str, default='1,2,3,4')
args = parser.parse_args()

from tqdm import tqdm

class CVPR24EVALDataset(Dataset):
    def __init__(self, roots):
        self.indexs = [(os.path.join(roots, d,'0.jpg'), os.path.join(roots, d,'1.jpg')) for d in os.listdir(roots)]
        self.roots = roots
        self.indexs.sort()
        self.lens = len(self.indexs)
        self.rsztsr = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((512,512)),torchvision.transforms.Normalize(mean=((0.485, 0.455, 0.406)), std=((0.229, 0.224, 0.225)))])

    def __len__(self):
        return self.lens

    def __getitem__(self, idx):
        try:
            img1 = cv2.cvtColor(cv2.imread(self.indexs[idx][0]),cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(cv2.imread(self.indexs[idx][1]),cv2.COLOR_BGR2RGB)
            img1 = self.rsztsr(img1)
            img2 = self.rsztsr(img2)
            imgs = torch.cat((img1, img2), 0)
            return (imgs, self.indexs[idx][0], self.indexs[idx][1], False)
        except:
            print('error')
            return (None, None, None, True)

device = torch.device("cuda")

roots1 = './' 
test_data = CVPR24EVALDataset(roots1)

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter("[%(asctime)s][%(filename)s][%(levelname)s] %(message)s")
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger

class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def second2time(second):
    if second < 60:
        return str('{}'.format(round(second, 4)))
    elif second < 60*60:
        m = second//60
        s = second % 60
        return str('{}:{}'.format(int(m), round(s, 1)))
    elif second < 60*60*60:
        h = second//(60*60)
        m = second % (60*60)//60
        s = second % (60*60) % 60
        return str('{}:{}:{}'.format(int(h), int(m), int(s)))

def inial_logger(file):
    logger = logging.getLogger('log')
    logger.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(file)
    file_handler.setLevel(level=logging.INFO)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from mmseg.utils import get_root_logger
from convnext import ConvNeXt

model=ConvNeXt(in_chans=6, depths=[3, 3, 27, 3],  dims=[256, 512, 1024, 2048],  drop_path_rate=0.8, layer_scale_init_value=1.0, num_classes=8).to(device)

model = nn.DataParallel(model)
loaders = torch.load('convxl.pth',map_location='cpu')['state_dict']
model.load_state_dict(loaders)
model = model.cuda()
model.eval()

all_dict = {}
SPG = []
SDG = []
NotAlignedSPG = []

with torch.no_grad():
    for idx in tqdm(range(len(test_data))):
        (imgs,auth,temp,flags) = test_data.__getitem__(idx)
        if flags:
            continue
        pred = model(imgs.unsqueeze(0))
        b,c = pred.shape
        pred = F.softmax(pred.reshape(b,c//2,2),dim=-1).cpu().numpy()
        all_dict[temp]=(auth, pred)
        if ((pred[0,0,1]>0.5) and (pred[0,1,1]>0.5)): # SPG
            SPG.append((auth, temp))
        if ((pred[0,0,0]>0.5) and (pred[0,1,0]>0.5)): # SDG
            SDG.append((auth, temp))
        if ((pred[0,0,1]>0.5) and (pred[0,1,0]>0.5)): # NotAlignedSPG
            NotAlignedSPG.append((auth, temp))

with open('convxl_cls.pk','wb') as f:
    pickle.dump(all_dict, f)



