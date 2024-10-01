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
parser.add_argument('--bs', type=int, default=1)
parser.add_argument('--es', type=int, default=0)
parser.add_argument('--ep', type=int, default=128)
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

class CVPR24Dataset(Dataset):
    def __init__(self, roots, img_dir, times=3):
        with open('pks/'+img_dir+'.pk','rb') as f:
            self.indexs = pickle.load(f)
        self.lens = len(self.indexs)
        self.roots = os.path.join(roots, img_dir)
        self.rsz = A.Compose([A.RandomScale(scale_limit=0.5,interpolation=1,p=0.5), A.RandomScale(scale_limit=0.5,interpolation=0,p=0.5), A.ImageCompression(quality_lower=50, quality_upper=99, p=0.5), A.Resize(1024,1024)])
        self.transforms = A.Compose([A.RandomCrop(width=768, height=768), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), ToTensorV2()])
        self.toctsr =torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=((0.485, 0.455, 0.406)*times), std=((0.229, 0.224, 0.225)*times))])

    def __len__(self):
        return self.lens

    def __getitem__(self, idx):
        im1, im2 = self.indexs[idx]
        # [{'filename': '0/0.jpg', 'ann': {'seg_map': '0/0.png'}}, {'filename': '0/1.jpg', 'ann': {'seg_map': '0/1.png'}}]
        img1 = cv2.imread(os.path.join(self.roots, im1['filename']))
        img2 = cv2.imread(os.path.join(self.roots, im2['filename']))
        mask = (cv2.imread(os.path.join(self.roots, im2['ann']['seg_map']), 0)>127).astype(np.uint8)
        img1 = self.rsz(image=img1)['image']
        rsts = self.rsz(image=img2, mask=mask)
        img2 = rsts['image']
        mask = rsts['mask']
        diff = np.abs(img1.astype(np.float32)-img2.astype(np.float32)).astype(np.uint8)
        imgs = np.concatenate((img1,img2,diff),2)
        rsts = self.transforms(image=imgs,mask=mask)
        imgs = self.toctsr(rsts['image'].float())
        mask = rsts['mask'].long()
        return (imgs, mask)

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

train_data = torch.utils.data.ConcatDataset([CVPR24Dataset(roots1,'c22'), CVPR24Dataset(roots1,'imc22')])
test_data1 = CVPR24Dataset(roots1, 'c22t')
test_data2 = CVPR24Dataset(roots1, 'imc22t')

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
from convbuper import ConvBUPer

model=ConvBUPer().to(device)
loaders = torch.load('ade_pths/convb_ade.pth')['state_dict']
loaders['backbone.downsample_layers.0.0.weight']=torch.cat((loaders['backbone.downsample_layers.0.0.weight'], loaders['backbone.downsample_layers.0.0.weight'], loaders['backbone.downsample_layers.0.0.weight']),1)
for hd in ('decode_head', 'auxiliary_head'):
    for nm in ('weight', 'bias'):
        loaders[f'{hd}.conv_seg.{nm}']=loaders[f'{hd}.conv_seg.{nm}'][:2]
model.load_state_dict(loaders)

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
    ce_loss = nn.CrossEntropyLoss()
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
            data, target = batch_samples
            data, target = Variable(data.to(device)), Variable(target.to(device))
            if True:#with autocast(): #need pytorch>1.6
                pred,pred2 = model(data)
                loss = (ce_loss(pred, target)+ce_loss(pred2, target)/2.0)
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
            logger.info('[val casia] epoch:{} iou:{}'.format(epoch,iu))

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


