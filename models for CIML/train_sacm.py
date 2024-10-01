#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@ori author: liuyaqi
@modify: QuChenfan
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
import utils
import time
import logging
import argparse
from PIL import Image
from tqdm import tqdm
import albumentations as A
import torch.distributed as dist
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import dmac_convb as dmac_vgg
parser = argparse.ArgumentParser()
parser.add_argument('--nm', type=str, default='ori')
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--data_path', type=str, default='/media/dplearning2/chenfan/cvpr_data/')
parser.add_argument('--numw', type=int, default=16)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--input_scale', type=int, default=512)
parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
args = parser.parse_args()

SEED=1234
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def snapshot(model, prefix, epoch, iter):
    print('taking snapshot ...')
    torch.save(model.state_dict(), prefix + str(epoch) + '_' + str(iter) + '.pth')

def trainhist_snapshot(train_hist, prefix, epoch, iter):
    filename = prefix + str(epoch) + '_' + str(iter) + '_loss.log'
    file = open(filename,'w')
    file.write(str(train_hist))
    file.close()
   
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

class CVPR24Dataset(Dataset):
    def __init__(self, roots, img_dir, sz=512):
        self.roots = os.path.join(roots, img_dir)
        csvs = [(self.roots+'/DMAC-COCO/train2014/labelfiles/'+x) for x in os.listdir(self.roots+'/DMAC-COCO/train2014/labelfiles/') if (x.endswith('.csv') and (not ('neg' in x)) and ('fore' in x))]
        assert len(csvs)==3
        self.indexs = []
        for c in csvs:
            with open(c) as f:
                fl = f.read().splitlines()[1:]
                self.indexs.extend(fl)
        # print(self.indexs[:10])
        self.mskb = np.zeros((sz, sz), dtype=np.uint8)
        self.lens = len(self.indexs)
        self.pt = A.Compose([A.ColorJitter(p=0.3), A.RGBShift(p=0.3), A.GaussianBlur(p=0.1), A.ImageCompression(quality_lower=50, quality_upper=99, p=0.5), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), A.Resize(sz, sz)])
        self.pt2 = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(mean=((0.485, 0.455, 0.406)), std=((0.229, 0.224, 0.225)))])
        self.tsr = A.Compose([A.Resize(sz,sz), ToTensorV2()])
        self.lbl = torch.FloatTensor([1])

    def __len__(self):
        return self.lens

    def __getitem__(self, idx):
      if True:
        im1, im2, lbl, gt1, gt2 = self.indexs[idx].split(',')
        im1 = os.path.join(self.roots, 'DMAC-COCO/train2014', im1)
        im2 = os.path.join(self.roots, 'DMAC-COCO/train2014', im2)
        if not (os.path.exists(im1) and os.path.exists(im2)):
            return self.__getitem__(np.random.randint(0, self.lens))
        # print(os.path.exists(os.path.join(self.roots, 'DMAC-COCO/train2014', im1)), os.path.exists(os.path.join(self.roots, 'DMAC-COCO/train2014', im2)))
        img1 = cv2.imread(os.path.join(self.roots, 'DMAC-COCO/train2014', im1))
        img2 = cv2.imread(os.path.join(self.roots, 'DMAC-COCO/train2014', im2))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        if ((gt1=='null') or (gt2=='null')):
            msk1 = self.mskb
            msk2 = self.mskb
        else:
            msk1 = (cv2.imread(os.path.join(self.roots, 'DMAC-COCO/train2014', gt1), 0)>127).astype(np.uint8)
            msk2 = (cv2.imread(os.path.join(self.roots, 'DMAC-COCO/train2014', gt2), 0)>127).astype(np.uint8)
        rsts = self.pt(image=img1, mask=msk1)
        img1 = rsts['image']
        msk1 = rsts['mask']
        rsts = self.pt(image=img2, mask=msk2)
        img2 = rsts['image']
        msk2 = rsts['mask']
        img1 = self.pt2(img1)
        img2 = self.pt2(img2)
        msk1 = self.tsr(image=msk1)['image']
        msk2 = self.tsr(image=msk2)['image']
        return (img1, img2, self.lbl, msk1, msk2)
      else:
        print('error')
        return self.__getitem__(np.random.randint(0,self.lens))

class CVPR24RealDataset(Dataset):
    def __init__(self, roots, img_dir, repeats=1, fan=False):
        self.fan = fan
        self.roots = os.path.join(roots, img_dir)
        self.indexs = [os.path.join(self.roots, x) for x in os.listdir(self.roots)]
        self.indexs = (self.indexs * repeats)
        self.lens = len(self.indexs)
        self.roots = os.path.join(roots, img_dir)
        self.lbl = torch.FloatTensor([0])
        self.rsz = A.Compose([A.RGBShift(p=0.3), A.ImageCompression(quality_lower=50, quality_upper=99, p=0.5), A.Resize(640, 640)])
        self.transforms = A.Compose([A.RandomCrop(width=512, height=512), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5), ToTensorV2()])
        self.toctsr =torchvision.transforms.Compose([torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])

    def __len__(self):
        return self.lens

    def __getitem__(self, idx):
      if True:
        this_r = self.indexs[idx]
        img1 = self.rsz(image=cv2.cvtColor(cv2.imread(os.path.join(this_r, '0.jpg')), cv2.COLOR_BGR2RGB))['image']
        img2 = self.rsz(image=cv2.cvtColor(cv2.imread(os.path.join(this_r, '1.jpg')), cv2.COLOR_BGR2RGB))['image']
        if self.fan:
            mask = (cv2.resize(cv2.imread(os.path.join(this_r, '1.png'), 0), (640, 640))<127).astype(np.uint8)
        else:
            mask = (cv2.resize(cv2.imread(os.path.join(this_r, '1.png'), 0), (640, 640))>127).astype(np.uint8)
        # imgs = np.concatenate((img1,img2),2)
        rsts = self.transforms(image=img1)
        img1 = self.toctsr(rsts['image'].float()/255.0)
        rsts = self.transforms(image=img2, mask=mask)
        img2 = self.toctsr(rsts['image'].float()/255.0)
        # imgs = self.toctsr(rsts['image'].float())
        mask = (F.interpolate(rsts['mask'][None,None].float(), size=(512, 512), mode='bilinear')>0.5).squeeze().long()
        return (img1, img2, self.lbl, mask, mask)
      else:
        print('error')
        return self.__getitem__(np.random.randint(0,self.lens))

class CVPR24EvalDataset(Dataset):
    def __init__(self, roots, img_dir, sz=512, fan=False):
        self.fan = fan
        self.roots = os.path.join(roots, img_dir)
        self.indexs = [os.path.join(self.roots, x) for x in os.listdir(self.roots)]
        self.lens = len(self.indexs)
        self.roots = os.path.join(roots, img_dir)
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
        if self.fan:
            mask = (cv2.imread(os.path.join(this_r, '1.png'), 0)<127).astype(np.uint8)
        else:
            mask = (cv2.imread(os.path.join(this_r, '1.png'), 0)>127).astype(np.uint8)
        mask = self.rsz(self.tsr(image=((mask>0.5).astype(np.uint8)))['image'].float()).long()
        return (img1, img2, mask)#self.lbl, mask, mask)

class CELearning(object):
    def __init__(self, args):
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.input_scale = args.input_scale
        self.start_epoch_idx = 0 # args.start_epoch_idx
        self.start_iter_idx = 0 # args.start_iter_idx
        self.snapshot_prefix_loc = 'pths/'
        if not os.path.exists(self.snapshot_prefix_loc):
            try:
                os.mkdir(self.snapshot_prefix_loc)
            except:
                pass
        self.logger = get_logger(os.path.join(self.snapshot_prefix_loc, time.strftime("%m-%d", time.localtime()) + args.nm + '.log'))
        self.data_path = args.data_path
        self.dataset = torch.utils.data.ConcatDataset([CVPR24Dataset(self.data_path,'dataprepare%d'%i) for i in range(1,10)] + [CVPR24RealDataset('/media/dplearning2/chenfan/cvpr_data/', 'b22', repeats=50), CVPR24RealDataset('/media/dplearning2/chenfan/cvpr_data/', 'imb22', repeats=100)])
        self.roots = '/media/dplearning2/chenfan/cvpr_data/'
        self.test_data1 = CVPR24EvalDataset(self.roots, 'imc22t', fan=True)
        self.test_data2 = CVPR24EvalDataset(self.roots, 'imb22t')
        self.test_loader1 = DataLoader(dataset=self.test_data1, batch_size=args.batch_size, num_workers=4)
        self.test_loader2 = DataLoader(dataset=self.test_data2, batch_size=args.batch_size, num_workers=4)
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset,shuffle=True)
        self.train_loader = DataLoader(dataset=self.dataset, batch_size=args.batch_size, num_workers=args.numw, sampler=self.train_sampler)
        self.loc = dmac_vgg.DMAC_VGG(2, self.input_scale)
        
        # loc_saved_state_dict = torch.load('DMAC_vgg_pretrained_init.pth', map_location='cpu')
        # self.loc.load_state_dict(loc_saved_state_dict)    
        
        self.loc = self.loc.cuda()
        self.loc = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.loc)
        self.loc = torch.nn.parallel.DistributedDataParallel(self.loc, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        loc_saved_state_dict = torch.load('pths/convnext_ade.pth', map_location='cpu')
        self.loc.load_state_dict(loc_saved_state_dict,strict=False) 

        self.loc_optimizer = optim.AdamW(self.loc.parameters(),lr=1e-4)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.ce_criterion = nn.NLLLoss2d()
        self.ce_criterion_ns = nn.NLLLoss2d(reduce=False)
        
        # print('---------- Networks architecture -------------')
        # print_network(self.loc)
        # print('-----------------------------------------------')
    
    
        
    def train(self):
        best_ious = 0
        self.train_hist = {}
        self.train_hist['loc_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []
        
        self.loc.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.start_epoch_idx, self.epoch):
            epoch_start_time = time.time()
            if epoch == self.start_epoch_idx:
                start_iter_idx = self.start_iter_idx
            else:
                start_iter_idx = 0
            for iter,chunk in enumerate(tqdm(self.train_loader)):
                images1, images2, labels, gt1, gt2 = chunk
                # images1, images2, labels, gt1, gt2 = utils.get_data_from_chunk(self.data_path,chunk,self.input_scale)
                # print(images1.shape, images2.shape, labels.shape, gt1.shape, gt2.shape, gt1.max(), images1.max())
                images1 = images1.to(device)#*255
                images2 = images2.to(device)#*255
                labels = labels.to(device)
                # gt masks variable
                gt1_ = torch.squeeze(gt1,dim=1).long().to(device)
                gt2_ = torch.squeeze(gt2,dim=1).long().to(device)
                
                # localization
                output1, output2, o1, o2 = self.loc(images1, images2)
                
                #localization update
                if True:#(iter+1) % self.loc_update_stride == 0:
                    self.loc_optimizer.zero_grad()
                    #localization net update
                    log_o1 = self.logsoftmax(output1)
                    log_o2 = self.logsoftmax(output2)
                    loc_loss_1 = self.ce_criterion_ns(log_o1,gt1_)
                    loc_loss_1 = (labels.unsqueeze(2).expand_as(loc_loss_1) * loc_loss_1).mean()
                    loc_loss_2 = self.ce_criterion(log_o2,gt2_)

                    lo1 = self.logsoftmax(o1)
                    lo2 = self.logsoftmax(o2)
                    loss_1 = self.ce_criterion_ns(lo1,gt1_)
                    loss_1 = (labels.unsqueeze(2).expand_as(loss_1) * loss_1).mean()
                    loss_2 = self.ce_criterion(lo2,gt2_)

                    loc_loss = (loc_loss_1 + loc_loss_2 + loss_1*0.4 + loss_2*0.4)
                    self.train_hist['loc_loss'].append(loc_loss.data)
                    loc_loss.backward()
                    self.loc_optimizer.step()
                    outm1 = output1.max(dim=1).indices
                    outm2 = output2.max(dim=1).indices
                    if (((iter+1) % 64 == 0) and (args.local_rank==0)):
                        with torch.no_grad():
                            i1 = (outm1 * gt1_).sum()
                            u1 = (outm1.sum()+gt1_.sum()-i1)
                            i2 = (outm2 * gt2_).sum()
                            u2 = (outm2.sum()+gt2_.sum()-i2)
                        print((i1/(u1+1e-6)).cpu().data,(i2/(u2+1e-6)).cpu().data)
                        print('********************************************************************************')
                        print('iter = ',iter, '  epoch = ', epoch, 'completed, loc_loss = ', loc_loss.data.cpu().numpy())
                        print('iter = ',iter, '  epoch = ', epoch, 'completed, loc_loss_1 = ', loc_loss_1.data.cpu().numpy())
                        print('iter = ',iter, '  epoch = ', epoch, 'completed, loc_loss_2 = ', loc_loss_2.data.cpu().numpy())
                if (((iter + 1) % 2048 == 0) and (args.local_rank==0)):
                    self.loc.eval()
                    with torch.no_grad():
                      with self.loc.no_sync():
                        ious = []
                        for (im1, im2, gt) in tqdm(self.test_loader1):
                            im1 = im1.to(device)
                            im2 = im2.to(device)
                            gt = gt.to(device)
                            _, pred, _, _ = self.loc(im1, im2)
                            pred = pred.max(1,keepdim=True).indices
                            i2 = (pred * gt).sum()
                            u2 = (pred.sum()+gt.sum()-i2)
                            ious.append((i2/(u2+1e-6)).cpu().numpy())
                        ious = np.array(ious).mean()
                        self.logger.info('Epoch:{} Iter:{} casia IoU:{}'.format(epoch, iter, ious))

                        ious = []
                        for (im1, im2, gt) in tqdm(self.test_loader2):
                            im1 = im1.to(device)
                            im2 = im2.to(device)
                            gt = gt.to(device)
                            _, pred, _, _ = self.loc(im1, im2)
                            pred = pred.max(1,keepdim=True).indices
                            i2 = (pred * gt).sum()
                            u2 = (pred.sum()+gt.sum()-i2)
                    ious.append((i2/(u2+1e-6)).cpu().numpy())
                    ious = np.array(ious).mean()
                    self.logger.info('Epoch:{} Iter:{} imd IoU:{}'.format(epoch, iter, ious))
                    snapshot(self.loc, self.snapshot_prefix_loc, epoch, iter)       
                    trainhist_snapshot(self.train_hist['loc_loss'],self.snapshot_prefix_loc, epoch, iter)
                    if ious>best_ious:
                        best_ious = ious
                        snapshot(self.loc, self.snapshot_prefix_loc, 0, 0)
                        print('='*20+'saved best'+'='*20)
                    print('='*20+'saved'+'='*20)
                    self.train_hist['loc_loss'] = []
                    self.loc.train()
            if args.local_rank==0:    
                self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)

        if args.local_rank==0:
            self.train_hist['total_time'].append(time.time() - start_time)
            print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']), self.epoch, self.train_hist['total_time'][0]))
            print("Training finish!... save training results")
        return 

device = torch.device("cuda",args.local_rank)
torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='nccl')
tc = CELearning(args)
tc.train()
                
                
        
        
        
