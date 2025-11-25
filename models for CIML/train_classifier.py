import os
import cv2
import math
import torch#用户ID：7fb702cd-1293-4470-a3b2-4ba88c3b3d4a
import numpy as np
import torch.nn as nn
import logging
import torch.optim as optim
import torch.distributed as dist
import random
import pickle
from PIL import Image
from tqdm import tqdm
from torch.autograd import Variable
from torch.cuda.amp import autocast
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler#need pytorch>1.6
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../../')
parser.add_argument('--train_name', type=str, default='CHDOC_JPEG0')
parser.add_argument('--train_imgs_dir', type=str, default='/home/jingroup/storage/chenfan/CVPR/datas/jpegc1_use')
parser.add_argument('--val_imgs_dir', type=str, default='/home/jingroup/storage/chenfan/CVPR/datas/jpegc1_use')
parser.add_argument('--train_labels_dir', type=str, default='/home/jingroup/storage/chenfan/CH_DOC/label')
parser.add_argument('--val_labels_dir', type=str, default='/home/jingroup/storage/chenfan/CH_DOC/val_label')
parser.add_argument('--model_name', type=str, default='catneto')
parser.add_argument('--att', type=str, default='None')
parser.add_argument('--num', type=str, default='1')
parser.add_argument('--n_class', type=int, default=2)
parser.add_argument('--bs', type=int, default=4)
parser.add_argument('--es', type=int, default=0)
parser.add_argument('--ep', type=int, default=1)
parser.add_argument('--xk', type=int, default=0)
parser.add_argument('--numw', type=int, default=16)
parser.add_argument('--load', type=int, default=0)
parser.add_argument('--pilt', type=int, default=0)
parser.add_argument('--base', type=int, default=1)
parser.add_argument('--lr_base', type=float, default=3e-4)
parser.add_argument('--cp', type=float, default=1.0)
parser.add_argument('--mode', type=str, default='0123')
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
parser.add_argument('--adds', type=str, default='123')
parser.add_argument('--lossw', type=str, default='1,2,3,4')
args = parser.parse_args()

SEED=1234
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
args = parser.parse_args()
use_pilt = (args.pilt==1)

import numpy as np
from scipy.special import binom
import matplotlib.pyplot as plt

bernstein = lambda n, k, t: binom(n, k) * t ** k * (1. - t) ** (n - k)

def bezier(points, num=200):
    N = len(points)
    t = np.linspace(0, 1, num=num)
    curve = np.zeros((num, 2))
    for i in range(N):
        curve += np.outer(bernstein(N - 1, i, t), points[i])
    return curve

class Segment():
    def __init__(self, p1, p2, angle1, angle2, **kw):
        self.p1 = p1
        self.p2 = p2
        self.angle1 = angle1
        self.angle2 = angle2
        self.numpoints = kw.get("numpoints", 100)
        r = kw.get("r", 0.3)
        d = np.sqrt(np.sum((self.p2 - self.p1) ** 2))
        self.r = r * d
        self.p = np.zeros((4, 2))
        self.p[0, :] = self.p1[:]
        self.p[3, :] = self.p2[:]
        self.calc_intermediate_points(self.r)

    def calc_intermediate_points(self, r):
        self.p[1, :] = self.p1 + np.array([self.r * np.cos(self.angle1),
                                           self.r * np.sin(self.angle1)])
        self.p[2, :] = self.p2 + np.array([self.r * np.cos(self.angle2 + np.pi),
                                           self.r * np.sin(self.angle2 + np.pi)])
        self.curve = bezier(self.p, self.numpoints)

def get_curve(points, **kw):
    segments = []
    for i in range(len(points) - 1):
        seg = Segment(points[i, :2], points[i + 1, :2], points[i, 2], points[i + 1, 2], **kw)
        segments.append(seg)
    curve = np.concatenate([s.curve for s in segments])
    return segments, curve

def ccw_sort(p):
    d = p - np.mean(p, axis=0)
    s = np.arctan2(d[:, 0], d[:, 1])
    return p[np.argsort(s), :]

def get_bezier_curve(a, rad=0.2, edgy=0.):
    p = np.arctan(edgy) / np.pi + .5
    a = ccw_sort(a)
    a = np.append(a, np.atleast_2d(a[0, :]), axis=0)
    d = np.diff(a, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    f = lambda ang: (ang >= 0) * ang + (ang < 0) * (ang + 2 * np.pi)
    ang = f(ang)
    ang1 = ang
    ang2 = np.roll(ang, 1)
    ang = p * ang1 + (1 - p) * ang2 + (np.abs(ang2 - ang1) > np.pi) * np.pi
    ang = np.append(ang, [ang[0]])
    a = np.append(a, np.atleast_2d(ang).T, axis=1)
    s, c = get_curve(a, r=rad, method="var")
    x, y = c.T
    return x, y, a

def get_random_points(n=5, scale=0.8, mindst=None, rec=0):
    """ create n random points in the unit square, which are *mindst*
    apart, then scale them."""
    mindst = mindst or .7 / n
    a = np.random.rand(n, 2)
    d = np.sqrt(np.sum(np.diff(ccw_sort(a), axis=0), axis=1) ** 2)
    if np.all(d >= mindst) or rec >= 200:
        return a * scale
    else:
        return get_random_points(n=n, scale=scale, mindst=mindst, rec=rec + 1)

class CVPR24CLSDataset(Dataset):
    def __init__(self, roots, img_dir, times=1):
        # if not os.path.exists('pks'):
        #     os.mkdir('pks')
        pkp = 'pks/'+img_dir+'_cls.pk'
        self.roots = os.path.join(roots, img_dir)
        if os.path.exists(pkp):
            with open(pkp,'rb') as f:
                self.indexs = pickle.load(f)
        else:
            self.indexs = [os.path.join(self.roots, j) for j in os.listdir(self.roots)]
            self.indexs = [[os.path.join(j, x) for x in os.listdir(j) if x.endswith('.jpg')] for j in self.indexs]
            with open(pkp,'wb') as f:
                 pickle.dump(self.indexs, f)
        self.indexs = (self.indexs*times)
        self.lens = len(self.indexs)
        self.transforms = A.Compose([A.ColorJitter(), A.RGBShift(), A.ImageCompression(quality_lower=50, quality_upper=99, p=0.2), A.OneOf([A.Compose([A.Resize(1280,1280), A.RandomCrop(1024, 1024)]), A.Resize(1024,1024)],p=1.0), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])
        self.toctsr = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=((0.485, 0.455, 0.406)), std=((0.229, 0.224, 0.225)))])
        self.rsztsr = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((512,512)),torchvision.transforms.Normalize(mean=((0.485, 0.455, 0.406)), std=((0.229, 0.224, 0.225)))])
        self.flptsr = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.RandomHorizontalFlip(p=1.0),torchvision.transforms.Resize((512,512)),torchvision.transforms.Normalize(mean=((0.485, 0.455, 0.406)), std=((0.229, 0.224, 0.225)))])

    def __len__(self):
        return self.lens

    def cmsp(self, img1, img2, im2, typea=True):
        # msp2 = im2[:-4]+'.png'
        # if ((random.uniform(0,1)<0.5) and (not im2.endswith('0.jpg')) and os.path.exists(msp2)):
        #     msk2 = (cv2.imread(msp2,0)>127).astype(np.float32)
        # else:
        for cnts in range(random.randint(1,3)):
            h = random.randint(64,640)
            w = random.randint(64,640)
            msk2 = np.zeros((h, w), dtype=np.float32)
            a = get_random_points(n=random.randint(3,8), scale=1)
            x, y, _ = get_bezier_curve(a, rad=0.2, edgy=0.01)
            x, y = x * w, y * h
            contours = np.array([(x[i], y[i]) for i in range(len(x))], dtype=np.int32)
            cv2.fillPoly(msk2, [contours], 1)
            if random.uniform(0,1)<0.8:
                msk2 = cv2.GaussianBlur(msk2, (random.randint(1,5)*2+1, random.randint(1,5)*2+1), random.uniform(1,3))
            start_x1 = np.random.randint(0,1024-h)
            start_y1 = np.random.randint(0,1024-w)
            end_x1 = (start_x1 + h)
            end_y1 = (start_y1 + w)
            start_x2 = np.random.randint(0,1024-h)
            start_y2 = np.random.randint(0,1024-w)
            end_x2 = (start_x2 + h)
            end_y2 = (start_y2 + w)
            msk2 = msk2[...,None]
            img1[start_x1:end_x1, start_y1:end_y1] = ((img1[start_x1:end_x1, start_y1:end_y1] * (1 - msk2)) + img2[start_x2:end_x2, start_y2:end_y2] * msk2)
        return img1

    def __getitem__(self, idx):
      try:
        im1 = random.choice(self.indexs[idx])
        if random.uniform(0,1)<0.5:
            idx2 = np.random.randint(0,self.lens)
            while (idx2==idx):
                idx2 = np.random.randint(0,self.lens)
            cls_label = 0
            # flp_label = -1
            sam_label = 0
            reg_label = (0.0,0.0,1.0,1.0)
            im2 = random.choice(self.indexs[idx2])
            img2 = self.transforms(image=cv2.cvtColor(cv2.imread(im2),cv2.COLOR_BGR2RGB))['image']
        else:
            cls_label = 1
        img1 = self.transforms(image=cv2.cvtColor(cv2.imread(im1),cv2.COLOR_BGR2RGB))['image']
        if (cls_label==0):
            if random.uniform(0,1)<0.9:
                if random.uniform(0,1)<0.5:
                    img1 = self.cmsp(img1, img2, im2)
                else:
                    img2 = self.cmsp(img2, img1, im1)
            # cv2.imwrite('demos/%d.jpg'%idx,img1)
            # cv2.imwrite('demos/%d_.jpg'%idx,img2)
            img1 = self.rsztsr(img1)
            img2 = self.rsztsr(img2)
            # print('*'*10)
        else:
            idx3 = np.random.randint(0,self.lens)
            im3 = random.choice(self.indexs[idx3])
            img3 = self.transforms(image=cv2.cvtColor(cv2.imread(im3),cv2.COLOR_BGR2RGB))['image']
            # print(img1.shape, img3.shape)
            img2 = img1.copy()
            if random.uniform(0,1)<0.9:
                img2 = self.cmsp(img2, img3, im3)
            if random.uniform(0,1)<0.8:
                sam_label = 0
                if random.uniform(0,1)<0.75:
                    start_x = random.randint(16, 128)
                    start_y = random.randint(16, 128)
                    end_x = random.randint(892, 1008)
                    end_y = random.randint(892, 1008)
                else:
                    start_x = random.randint(16, 256)
                    start_y = random.randint(16, 256)
                    height = random.randint(512, 1008-start_x)
                    width = random.randint(512, 1008-start_y)
                    end_x = (start_x + height)
                    end_y = (start_y + width)
                reg_label = (start_x/512.0, start_y/512.0, end_x/512.0-1, end_y/512.0-1)
                img2 = img2[start_x:end_x, start_y:end_y]
            else:
                sam_label = 1
                reg_label = (0.0,0.0,1.0,1.0)
            # cv2.imwrite('demos/%d.jpg'%idx,img1)
            # cv2.imwrite('demos/%d_.jpg'%idx,img2)
            # print(idx,cls_label,sam_label,reg_label,img1.shape,img2.shape)
            img1 = self.rsztsr(img1)
            img2 = self.rsztsr(img2)
            # print('img2',img2.shape)
        if random.uniform(0,1)<0.5:
            imgs = torch.cat((img1, img2), 0)
        else:
            imgs = torch.cat((img2, img1), 0)
        return (imgs, torch.LongTensor([cls_label]), torch.LongTensor([sam_label]), torch.FloatTensor(reg_label))
      except:
        print('error')
        return self.__getitem__(random.randint(0,self.lens-1))

class CVPR24ORIDataset(Dataset):
    def __init__(self, roots, img_dir, maxs=35000):
        # if not os.path.exists('pks'):
        #     os.mkdir('pks')
        pkp = 'pks/'+img_dir+'_cls.pk'
        self.roots = os.path.join(roots, img_dir)
        if os.path.exists(pkp):
            with open(pkp,'rb') as f:
                self.indexs = pickle.load(f)
        else:
            self.indexs = [os.path.join(self.roots, j) for j in os.listdir(self.roots)]
            # self.indexs = [[os.path.join(j, x) for x in os.listdir(j) if x.endswith('.jpg')] for j in self.indexs]
            with open(pkp,'wb') as f:
                 pickle.dump(self.indexs, f)
        self.indexs = (self.indexs[:maxs])
        self.lens = len(self.indexs)
        self.transforms = A.Compose([A.ColorJitter(), A.RGBShift(), A.ImageCompression(quality_lower=50, quality_upper=99, p=0.2), A.OneOf([A.Compose([A.Resize(1280,1280), A.RandomCrop(1024, 1024)]), A.Resize(1024,1024)],p=1.0), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5)])
        self.toctsr = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=((0.485, 0.455, 0.406)), std=((0.229, 0.224, 0.225)))])
        self.rsztsr = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Resize((512,512)),torchvision.transforms.Normalize(mean=((0.485, 0.455, 0.406)), std=((0.229, 0.224, 0.225)))])
        self.flptsr = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.RandomHorizontalFlip(p=1.0),torchvision.transforms.Resize((512,512)),torchvision.transforms.Normalize(mean=((0.485, 0.455, 0.406)), std=((0.229, 0.224, 0.225)))])

    def __len__(self):
        return self.lens

    def cmsp(self, img1, img2, im2, typea=True):
        # msp2 = im2[:-4]+'.png'
        # if ((random.uniform(0,1)<0.5) and (not im2.endswith('0.jpg')) and os.path.exists(msp2)):
        #     msk2 = (cv2.imread(msp2,0)>127).astype(np.float32)
        # else:
        for cnts in range(random.randint(1,3)):
            h = random.randint(64,640)
            w = random.randint(64,640)
            msk2 = np.zeros((h, w), dtype=np.float32)
            a = get_random_points(n=random.randint(3,8), scale=1)
            x, y, _ = get_bezier_curve(a, rad=0.2, edgy=0.01)
            x, y = x * w, y * h
            contours = np.array([(x[i], y[i]) for i in range(len(x))], dtype=np.int32)
            cv2.fillPoly(msk2, [contours], 1)
            if random.uniform(0,1)<0.8:
                msk2 = cv2.GaussianBlur(msk2, (random.randint(1,5)*2+1, random.randint(1,5)*2+1), random.uniform(1,3))
            start_x1 = np.random.randint(0,1024-h)
            start_y1 = np.random.randint(0,1024-w)
            end_x1 = (start_x1 + h)
            end_y1 = (start_y1 + w)
            start_x2 = np.random.randint(0,1024-h)
            start_y2 = np.random.randint(0,1024-w)
            end_x2 = (start_x2 + h)
            end_y2 = (start_y2 + w)
            msk2 = msk2[...,None]
            img1[start_x1:end_x1, start_y1:end_y1] = ((img1[start_x1:end_x1, start_y1:end_y1] * (1 - msk2)) + img2[start_x2:end_x2, start_y2:end_y2] * msk2)
        return img1

    def __getitem__(self, idx):
      try:
        im1 = self.indexs[idx]
        if random.uniform(0,1)<0.5:
            idx2 = np.random.randint(0,self.lens)
            while (idx2==idx):
                idx2 = np.random.randint(0,self.lens)
            cls_label = 0
            # flp_label = -1
            sam_label = 0
            reg_label = (0.0,0.0,0.0,0.0)
            im2 = self.indexs[idx2]
            img2 = self.transforms(image=cv2.cvtColor(cv2.imread(im2),cv2.COLOR_BGR2RGB))['image']
        else:
            cls_label = 1
        img1 = self.transforms(image=cv2.cvtColor(cv2.imread(im1),cv2.COLOR_BGR2RGB))['image']
        if (cls_label==0):
            if random.uniform(0,1)<0.9:
                if random.uniform(0,1)<0.5:
                    img1 = self.cmsp(img1, img2, im2)
                else:
                    img2 = self.cmsp(img2, img1, im1)
            # cv2.imwrite('demos/%d.jpg'%idx,img1)
            # cv2.imwrite('demos/%d_.jpg'%idx,img2)
            img1 = self.rsztsr(img1)
            img2 = self.rsztsr(img2)
            # print('*'*10)
        else:
            idx3 = np.random.randint(0,self.lens)
            im3 = self.indexs[idx3]
            img3 = self.transforms(image=cv2.cvtColor(cv2.imread(im3),cv2.COLOR_BGR2RGB))['image']
            img2 = img1.copy()
            if random.uniform(0,1)<0.9:
                img2 = self.cmsp(img2, img3, im3)
            if random.uniform(0,1)<0.8:
                sam_label = 0
                if random.uniform(0,1)<0.75:
                    start_x = random.randint(16, 128)
                    start_y = random.randint(16, 128)
                    end_x = random.randint(892, 1008)
                    end_y = random.randint(892, 1008)
                else:
                    start_x = random.randint(16, 256)
                    start_y = random.randint(16, 256)
                    height = random.randint(512, 1008-start_x)
                    width = random.randint(512, 1008-start_y)
                    end_x = (start_x + height)
                    end_y = (start_y + width)
                reg_label = (start_x/512.0, start_y/512.0, end_x/512.0-1, end_y/512.0-1)
                img2 = img2[start_x:end_x, start_y:end_y]
            else:
                sam_label = 1
                reg_label = (0.0,0.0,1.0,1.0)
            # cv2.imwrite('demos/%d.jpg'%idx,img1)
            # cv2.imwrite('demos/%d_.jpg'%idx,img2)
            # print(idx,cls_label,sam_label,reg_label,img1.shape,img2.shape)
            img1 = self.rsztsr(img1)
            img2 = self.rsztsr(img2)
        if random.uniform(0,1)<0.5:
            imgs = torch.cat((img1, img2), 0)
        else:
            imgs = torch.cat((img2, img1), 0)
        return (imgs, torch.LongTensor([cls_label]), torch.LongTensor([sam_label]), torch.FloatTensor(reg_label))
      except:
        print('error')
        return self.__getitem__(random.randint(0,self.lens-1))

ngpu = torch.cuda.device_count()
ngpub = ngpu * args.base
if ngpu > 1:
    gpus = True
    device = torch.device("cuda",args.local_rank)
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
else:
    gpus = False
    device = torch.device("cuda")

roots1 = '/media/16T1/chenfan/cvpr_data/' 
roots2 = '/media/16T1/chenfan/tampCOCO/'

train_data = torch.utils.data.ConcatDataset([CVPR24CLSDataset(roots1,'casia11',times=50), CVPR24CLSDataset(roots1,'imd11',times=100), CVPR24ORIDataset(roots2,'train2017'), CVPR24ORIDataset(roots2,'train2017')])
test_data1 = CVPR24CLSDataset(roots1, 'imd11')
test_data2 = CVPR24CLSDataset(roots1, 'imd22')

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

class IOUMetric:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        mean_iu = np.nanmean(iu)
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu, fwavacc

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from mmseg.utils import get_root_logger
from nat import NAT

model=NAT(in_chans=6, depths=[3, 4, 18, 5], num_heads=[6, 12, 24, 48], embed_dim=192, mlp_ratio=2,  drop_path_rate=0.2, kernel_size=7, dilations=[[1, 8, 1],[1, 4, 1, 4],[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],[1, 1, 1, 1, 1],], num_classes=8).to(device)
loaders = torch.load('ade_pths/dinat_l.pth')#['state_dict']
loaders['head.weight'] = loaders['head.weight'][:8]
loaders['head.bias'] = loaders['head.bias'][:8]
loaders['patch_embed.proj.0.weight'] = torch.cat((loaders['patch_embed.proj.0.weight'], loaders['patch_embed.proj.0.weight']),1)
model.load_state_dict(loaders)#,strict=False)

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1,2,3,1,1,bias=False)
        weights_x = torch.tensor([[1., 0., -1.],[2., 0., -2.],[1., 0., -1.]])
        weights_y = torch.tensor([[1., 2., 1.],[0., 0., 0.],[-1., -2., -1.]])
        self.weights = nn.Parameter(torch.cat((weights_x.view(1, 1, 3, 3),weights_y.view(1, 1, 3, 3))),requires_grad=False)
        self.conv.weights = self.weights
    def forward(self, x):
        output = self.conv(x)
        return torch.sum(torch.abs(output),dim=1,keepdim=False).unsqueeze(1)

if gpus:
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)

model_name = args.model_name
save_ckpt_dir = os.path.join('./outputs/', model_name, 'ckpt')
save_log_dir = os.path.join('./outputs/', model_name)
try:
  if not os.path.exists(save_ckpt_dir):
    os.makedirs(save_ckpt_dir)
except:
  pass
try:
  if not os.path.exists(save_log_dir):
    os.makedirs(save_log_dir)
except:
  pass
import gc
# ngpub = max(ngpub,2)
# 参数设置
param = {}
param['batch_size'] = args.bs       # 批大小
param['epochs'] = args.ep       # 训练轮数，请和scheduler的策略对应，不然复现不出效果，对于t0=3,t_mut=2的scheduler来讲，44的时候会达到最优
param['disp_inter'] = 1       # 显示间隔(epoch)
param['save_inter'] = 4       # 保存间隔(epoch)
param['iter_inter'] = 64     # 显示迭代间隔(batch)
param['min_inter'] = 10
param['model_name'] = model_name          # 模型名称
param['save_log_dir'] = save_log_dir      # 日志保存路径
param['save_ckpt_dir'] = save_ckpt_dir    # 权重保存路径
param['T0']=int(24/ngpub)  #cosine warmup的参数
# param['save_epoch']={2:[5,13,29,61],3:[8,20,44,92]}
param['load_ckpt_dir'] = None
import time

def collate_batch(batch_list):
    assert type(batch_list) == list, f"Error"
    batch_size = len(batch_list)
    data = torch.cat([item[0] for item in batch_list]).reshape(batch_size, -1)
    labels = torch.cat([item[1] for item in batch_list]).reshape(batch_size, -1)
    return data, labels

def train_net_qyl(param, model, train_data, test_data1, test_data2, plot=False,device='cuda'):
    # 初始化参数
    global gpus
    model_name      = param['model_name']
    epochs          = param['epochs']
    batch_size      = param['batch_size']
    iter_inter      = param['iter_inter']
    save_log_dir    = param['save_log_dir']
    save_ckpt_dir   = param['save_ckpt_dir']
    load_ckpt_dir   = param['load_ckpt_dir']
    T0=param['T0']
    scaler = GradScaler()
    world_size = dist.get_world_size()
    if gpus:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
        # val_sampler1 = torch.utils.data.distributed.DistributedSampler(test_data1,shuffle=False)
        # val_sampler2 = torch.utils.data.distributed.DistributedSampler(test_data2,shuffle=False)
    lr_base = args.lr_base 
    train_data_size = train_data.__len__()#+train_data2.__len__()
    if gpus:
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=args.numw, shuffle=False, sampler=train_sampler)
        valid_loader1 = DataLoader(dataset=test_data1, batch_size=batch_size, num_workers=args.numw, shuffle=False)
        valid_loader2 = DataLoader(dataset=test_data2, batch_size=batch_size, num_workers=args.numw, shuffle=False)
    else:
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, num_workers=args.numw, shuffle=False)
        valid_loader1 = DataLoader(dataset=test_data1, batch_size=batch_size, num_workers=args.numw, shuffle=False)
        valid_loader2 = DataLoader(dataset=test_data2, batch_size=batch_size, num_workers=args.numw, shuffle=False) 
    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)
    l1_loss = nn.SmoothL1Loss(beta=1/4096.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4 ,weight_decay=5e-2)
    iter_per_epoch = len(train_loader)
    totalstep = epochs*iter_per_epoch
    warmupr = 1/epochs
    warmstep = 1024//ngpu
    lr_min = 1e-6
    lr_min /= lr_base
    lr_dict = {i:((((1+math.cos((i-warmstep)*math.pi/(totalstep-warmstep)))/2)+lr_min) if (i > warmstep) else (i/warmstep+lr_min)) for i in range(totalstep)}
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_dict[epoch])
    logger = get_logger(os.path.join(save_log_dir, time.strftime("%m-%d", time.localtime()) +'_'+model_name+ '.log'))
    train_loss_total_epochs, valid_loss_total_epochs, epoch_lr = [], [], []
    best_iou = 0
    best_epoch=0
    epoch_start = 0
    '''
    if ((args.load!=0) and (args.es!=0)):
        ckpt = torch.load(os.path.join(save_ckpt_dir, 'checkpoint-best.pth'),map_location='cpu')
        epoch_start = (ckpt['epoch']+1)
        assert epoch_start==args.es,'{}!={}'.format(epoch_start,args.es)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
    '''
    logger.info('Total Epoch:{} Training num:{}  Validation num:{}'.format(epochs, train_data_size, 0))
    for epoch in range(epoch_start, epochs):
        train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        model.train()
        iter_i = (epoch*iter_per_epoch)
        train_epoch_loss = AverageMeter()
        train_iter_loss = AverageMeter()
        for batch_idx,batch_samples in enumerate(tqdm(train_loader)):
            data, target1, target2, target3 = batch_samples
            data, target1, target2, target3 = Variable(data.to(device)), Variable(target1.squeeze(1).to(device)), Variable(target2.squeeze(1).to(device)), Variable(target3.to(device))
            if True:#with autocast(): #need pytorch>1.6
                pred = model(data)
                loss_cls = ce_loss(pred[:,0:2], target1)
                loss_sam = ce_loss(pred[:,2:4], target2)
                loss_box = l1_loss(pred[:,4:], target3)
                loss = (loss_cls+loss_sam+loss_box)
                # scaler.scale(loss+bl).backward()
                # scaler.step(optimizer)
                # scaler.update()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            scheduler.step(iter_i+batch_idx) 
            image_loss = loss.item()
            train_epoch_loss.update(image_loss)
            train_iter_loss.update(image_loss)
            if ((batch_idx%16384==0) and (batch_idx!=0)):
                state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
                filename = os.path.join(save_ckpt_dir, 'checkpoint_%d_%d.pth'%(epoch, batch_idx))
                torch.save(state, filename)
            if batch_idx % iter_inter == 0:
                spend_time = time.time() - epoch_start
                logger.info('[train] epoch:{} iter:{}/{} {:.2f}% lr:{:.6f} loss:{:.6f} ETA:{}min'.format(
                    epoch, batch_idx, iter_per_epoch, batch_idx/iter_per_epoch*100,
                    optimizer.param_groups[-1]['lr'],
                    train_iter_loss.avg,spend_time / (batch_idx+1) * iter_per_epoch // 60 - spend_time // 60))
                train_iter_loss.reset()
                print(loss_cls.item(), loss_sam.item(), loss_box.item())
            if ((batch_idx!=0) and (batch_idx % 1024 == 0)):
                pass
                '''
                model.eval()
                valid_epoch_loss = AverageMeter()
                valid_iter_loss = AverageMeter()
                with torch.no_grad():
                  with model.no_sync():
                    iou=IOUMetric(2)
                    for batch_idx, batch_samples in enumerate(tqdm(valid_loader1)):
                        data, target = batch_samples
                        data, target = Variable(data.to(device)), Variable(target.to(device))
                        if True:#with autocast(): #need pytorch>1.6
                            pred = model(data)
                            loss = ce_loss(pred, target)
                        pred= torch.argmax(pred,axis=1)
                        iou.add_batch(pred.cpu().data.numpy(),target.cpu().data.numpy())
                        image_loss = loss.item()
                        valid_epoch_loss.update(image_loss)
                        valid_iter_loss.update(image_loss)
                    val_loss=valid_iter_loss.avg
                    acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
                    logger.info('[val imd] epoch:{} iou:{}'.format(epoch,iu))
                '''
                '''
                model.eval()
                valid_epoch_loss = AverageMeter()
                valid_iter_loss = AverageMeter()
                with torch.no_grad():
                  with model.no_sync():
                    iou=IOUMetric(2)
                    for batch_idx, batch_samples in enumerate(tqdm(valid_loader2)):
                        data, target = batch_samples
                        data, target = Variable(data.to(device)), Variable(target.to(device))
                        if True:#with autocast(): #need pytorch>1.6
                            pred = model(data)
                            loss = ce_loss(pred, target)
                        pred= torch.argmax(pred,axis=1)
                        iou.add_batch(pred.cpu().data.numpy(),target.cpu().data.numpy())
                        image_loss = loss.item()
                        valid_epoch_loss.update(image_loss)
                        valid_iter_loss.update(image_loss)
                    val_loss=valid_iter_loss.avg
                    acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
                    logger.info('[val imd] epoch:{} iou:{}'.format(epoch,iu))
                '''
                model.train()

        # 验证阶段
        model.eval()
        valid_epoch_loss = AverageMeter()
        valid_iter_loss = AverageMeter()
        # if args.local_rank==0:
        #     iou=IOUMetric(2)
        with torch.no_grad():
          with model.no_sync():
            iou=IOUMetric(2)
            for batch_idx, batch_samples in enumerate(tqdm(valid_loader1)):
                data, target = batch_samples
                data, target = Variable(data.to(device)), Variable(target.to(device))
                if True:#with autocast(): #need pytorch>1.6
                    pred = model(data)
                    loss = ce_loss(pred, target)
                # pred=pred.cpu()#.data.numpy()
                pred= torch.argmax(pred,axis=1)
                # print(pred.shape)
                iou.add_batch(pred.cpu().data.numpy(),target.cpu().data.numpy())
                image_loss = loss.item()
                valid_epoch_loss.update(image_loss)
                valid_iter_loss.update(image_loss)
            val_loss=valid_iter_loss.avg
            acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
            logger.info('[val imd] epoch:{} iou:{}'.format(epoch,iu))
        '''
        model.eval()
        valid_epoch_loss = AverageMeter()
        valid_iter_loss = AverageMeter()
        # if args.local_rank==0:
        #     iou=IOUMetric(2)
        with torch.no_grad():
          with model.no_sync():
            iou=IOUMetric(2)
            for batch_idx, batch_samples in enumerate(tqdm(valid_loader2)):
                data, target = batch_samples
                data, target = Variable(data.to(device)), Variable(target.to(device))
                if True:#with autocast(): #need pytorch>1.6
                    pred = model(data)
                    loss = ce_loss(pred, target)
                # pred=pred.cpu()#.data.numpy()
                pred= torch.argmax(pred,axis=1)
                # print(pred.shape)
                iou.add_batch(pred.cpu().data.numpy(),target.cpu().data.numpy())
                image_loss = loss.item()
                valid_epoch_loss.update(image_loss)
                valid_iter_loss.update(image_loss)
            val_loss=valid_iter_loss.avg
            acc, acc_cls, iu, mean_iu, fwavacc=iou.evaluate()
            logger.info('[val imd] epoch:{} iou:{}'.format(epoch,iu))
        '''

        # 保存loss、lr
        train_loss_total_epochs.append(train_epoch_loss.avg)
        valid_loss_total_epochs.append(valid_epoch_loss.avg)
        epoch_lr.append(optimizer.param_groups[0]['lr'])
        # 保存模型
        # if epoch in save_epoch[T0]:
        #     torch.save(model.state_dict(),'{}/cosine_epoch{}.pth'.format(save_ckpt_dir,epoch))
        # state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
        # filename = os.path.join(save_ckpt_dir, 'checkpoint-latest.pth')
        # torch.save(state, filename)  # pytorch1.6会压缩模型，低版本无法加载
        # 保存最优模型
        if iu[1] > best_iou:  # train_loss_per_epoch valid_loss_per_epoch
            state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
            filename = os.path.join(save_ckpt_dir, 'checkpoint-best.pth')
            torch.save(state, filename)
            best_iou = iu[1]
            # best_mode = copy.deepcopy(model)
            logger.info('[save] Best Model saved at epoch:{} ============================='.format(epoch))
        #scheduler.step()

train_net_qyl(param, model, train_data, test_data1, test_data2, device=device)


