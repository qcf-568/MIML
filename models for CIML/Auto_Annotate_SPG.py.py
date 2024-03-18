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
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='../../')
parser.add_argument('--train_name', type=str, default='CHDOC_JPEG0')
parser.add_argument('--model_name', type=str, default='exp')
parser.add_argument('--att', type=str, default='None')
parser.add_argument('--num', type=str, default='1')
parser.add_argument('--n_class', type=int, default=2)
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--es', type=int, default=0)
parser.add_argument('--ep', type=int, default=1)
parser.add_argument('--xk', type=int, default=0)
parser.add_argument('--numw', type=int, default=8)
parser.add_argument('--load', type=int, default=0)
parser.add_argument('--pilt', type=int, default=0)
parser.add_argument('--base', type=int, default=1)
parser.add_argument('--lr_base', type=float, default=3e-4)
parser.add_argument('--cp', type=float, default=1.0)
parser.add_argument('--mode', type=str, default='0123')
parser.add_argument('--adds', type=str, default='123')
parser.add_argument('--loss-', type=str, default='1,2,3,4')
args = parser.parse_args()

def getdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


class CVPR24REDataset(Dataset):
    def __init__(self, roots, img_dir, times=3, repeats=1):
        self.roots = os.path.join(roots, img_dir)
        self.indexs = [os.path.join(self.roots, x) for x in os.listdir(self.roots)]
        self.lens = len(self.indexs)
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
        self.rsz = A.Compose([A.Resize(1024,1024)])
        self.transforms = A.Compose([ToTensorV2()])
        self.toctsr =torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=((0.485, 0.455, 0.406)*times), std=((0.229, 0.224, 0.225)*times))])

    def __len__(self):
        return self.lens

    def __getitem__(self, idx):
        this_r = self.indexs[idx]
        print(this_r)
        this_r = (os.path.join(this_r, '1.jpg'), os.path.join(this_r, '0.jpg'))
        img1 = cv2.cvtColor(cv2.imread(this_r[1]), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread(this_r[0]), cv2.COLOR_BGR2RGB)
        h,w = img2.shape[:2]
        mask = np.zeros((h,w),dtype=np.uint8)
        img1 = self.rsz(image=img1)['image']
        rsts = self.rsz(image=img2, mask=mask)
        img2 = rsts['image']
        mask = rsts['mask']
        imgs = np.concatenate((img1,img2),2)
        rsts = self.transforms(image=imgs,mask=mask)
        imgs = rsts['image']
        imgs = (torch.cat((imgs,torch.abs(imgs[:3]-imgs[3:])), 0).float()/255.0)
        imgs = self.toctsr(imgs)
        mask = rsts['mask'].long()
        return (imgs, mask, this_r[0].split('/')[-2], h, w)

ngpu = torch.cuda.device_count()
ngpub = ngpu * args.base
if False:
    gpus = True
    device = torch.device("cuda",args.local_rank)
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')
else:
    gpus = False
    device = torch.device("cuda")

roots1 = './' 

test_data1 = CVPR24REDataset('your_data_dir/', 'SPG')
test_data2 = CVPR24REDataset('your_data_dir/', 'SPG')

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

from dass import DASS

model=DASS(in_chans=9).to(device)

model = nn.DataParallel(model)
loader = torch.load('DASS.pth',map_location='cpu')['state_dict']
model.load_state_dict(loader)

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
param['load_ckpt_dir'] = None
import time

def collate_batch(batch_list):
    assert type(batch_list) == list, f"Error"
    batch_size = len(batch_list)
    data = torch.cat([item[0] for item in batch_list]).reshape(batch_size, -1)
    labels = torch.cat([item[1] for item in batch_list]).reshape(batch_size, -1)
    return data, labels

def train_net_qyl(param, model, test_data1, test_data2, plot=False,device='cuda'):
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
    lr_base = args.lr_base 
    if gpus:
        # valid_loader1 = DataLoader(dataset=test_data1, batch_size=batch_size, num_workers=args.numw, shuffle=False)
        valid_loader2 = DataLoader(dataset=test_data2, batch_size=batch_size, num_workers=args.numw, shuffle=False)
    else:
        # valid_loader1 = DataLoader(dataset=test_data1, batch_size=batch_size, num_workers=args.numw, shuffle=False)
        valid_loader2 = DataLoader(dataset=test_data2, batch_size=batch_size, num_workers=args.numw, shuffle=False) 
    optimizer = optim.AdamW(model.parameters(), lr=1e-4 ,weight_decay=5e-2)
    if True:
        model.eval()
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(tqdm(valid_loader2)):
                data, target, fnms, h, w = batch_samples
                h = h.item()
                w = w.item()
                data, target = Variable(data.to(device)), Variable(target.to(device))
                if True:
                    d2 = torch.flip(data,dims=[2])
                    d3 = torch.flip(data,dims=[3])
                    data = torch.cat((data,d2,d3),0)
                    pred = model(data)
                    pred[1:2] = torch.flip(pred[1:2], dims=[2])
                    pred[2:3] = torch.flip(pred[2:3], dims=[3])
                    pred = pred.mean(0,keepdim=True)
                pred= (F.softmax(pred,dim=1)[:,1:2].cpu().numpy()*255).astype(np.uint8)
                for (p, fnm) in zip(pred, fnms):
                    ds = 'SPG_preds/'
                    getdir(ds)
                    p = cv2.resize(p.squeeze(),(w,h))
                    cv2.imwrite(ds+'/'+fnm+'.png', p)

train_net_qyl(param, model, test_data1, test_data2, device=device)


