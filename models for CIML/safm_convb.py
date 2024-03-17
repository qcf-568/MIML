"""
@author: liuyaqi
"""
import torch
import torch.nn as nn
from torch.nn import functional as F
from mmseg.models.decode_heads import DepthwiseSeparableASPPHead
import math
import numpy as np


affine_par = True

def outS(i):
    i = int(i)
    i = int(np.floor((i+1)/2.0))
    i = int(np.floor((i+1)/2.0))
    i = int(np.floor((i+1)/2.0))
    return i

class Convblock(nn.Module):
    def __init__(self, in_channels, out_channels, padding_, dilation_, batch_norm=False):
        super(Convblock,self).__init__()
        self.bnflag = batch_norm
        self.convb = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding_, dilation=dilation_)
        if self.bnflag:
            self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.convb(x)
        if self.bnflag:
            x = self.bn(x)
        x = self.relu(x)
        return x

def make_layers(cfg, in_channels = 3, batch_norm=False):
    layers = []
    
    for v in cfg:
        if v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'M1':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'D512':
            cb = Convblock(in_channels, 512, padding_=2, dilation_ = 2)
            layers += [cb]
            in_channels = 512
        else:
            cb = Convblock(in_channels, v, padding_=1, dilation_ = 1)
            layers += [cb]
            in_channels = v
    return nn.Sequential(*layers)


def get_mapping_indices(n_rows, n_cols ):
    new_indices = []
    for r_x in range( n_rows ):
        for c_x in range( n_cols ):
            for r_b in range( n_rows ):
                r_a = ( r_b + r_x ) % n_rows
                for c_b in range( n_cols ):
                    c_a = ( c_b + c_x ) % n_cols
                    idx_a = r_a * n_cols + c_a
                    idx_b = r_b * n_cols + c_b
                    idx = idx_a * ( n_rows * n_cols ) + idx_b
                    new_indices.append( idx )
    return new_indices

class Correlation_Module(nn.Module):
    
    def __init__(self):
        super(Correlation_Module, self).__init__()

    
    def forward(self, x1, x2, new_indices):
        
        [bs1, c1, h1, w1] = x1.size()
        pixel_num = h1 * w1
        
        x1 = torch.div(x1.view(bs1, c1, pixel_num),c1)
        x2 = x2.view(bs1, c1, pixel_num).permute(0,2,1)
        
        x1_x2 = torch.bmm(x2,x1)
        
        x1_x2 = torch.index_select(x1_x2.view(bs1, -1),1,new_indices)
        x1_x2 = x1_x2.view(bs1,pixel_num,h1,w1)
        
        return x1_x2
        
class Poolopt_on_Corrmat(nn.Module):
    
    def __init__(self, select_indices):
        super(Poolopt_on_Corrmat, self).__init__()
        self.fc = nn.Conv2d(4096, 14, 1, 1, 0)
        self.select_indices = select_indices
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, corr):
        max_corr = torch.max(corr,1,keepdim=True)
        avg_corr = torch.mean(corr,1,keepdim=True)
        sort_corr = self.fc(corr)
        corr = torch.cat((max_corr[0],avg_corr, sort_corr),1)
        return corr
    
        
def corr_fun(x1, x2, Corr, poolopt_on_corrmat, new_indices):
    
    corr12 = Corr(x1,x2,new_indices)
    corr21 = Corr(x2,x1,new_indices)
    
    corr11 = Corr(x1,x1,new_indices)
    corr22 = Corr(x2,x2,new_indices)
    
    corr12 = poolopt_on_corrmat(corr12)
    corr21 = poolopt_on_corrmat(corr21)
    
    corr11 = poolopt_on_corrmat(corr11)
    corr22 = poolopt_on_corrmat(corr22)
    
    corr1 = torch.cat((corr12,corr11),1)
    corr2 = torch.cat((corr21,corr22),1)
    
    return corr1,corr2

from convbuper import ConvBUPer

class SAFM_model(nn.Module):
    def __init__(self, block, NoLabels, h, w):
        super(SAFM_model, self).__init__()
        self.models = ConvBUPer()
        
        new_indices = get_mapping_indices(h, w)        
        self.new_indices = torch.tensor(new_indices,dtype=torch.long).cuda()
        self.Corr = Correlation_Module()
        
        sort_indices = [0,1,2,3,4,5,7,8,9,10,11,12,13,14]
        sort_indices = torch.tensor(sort_indices,dtype=torch.long).cuda()
        self.poolopt_on_corrmat = Poolopt_on_Corrmat(sort_indices)
        self.deeplab = DepthwiseSeparableASPPHead(in_channels=2048,in_index=1,channels=512,dilations=(1, 12, 24, 36),c1_in_channels=512,c1_channels=48,dropout_ratio=0.1,num_classes=2,norm_cfg=dict(type='SyncBN', requires_grad=True),align_corners=False)
        self.seg_out = nn.Conv2d(512,2,1,1,0)
        self.deeplab.conv_seg = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2.0),nn.Conv2d(512,128,3,1,1),nn.BatchNorm2d(128),nn.ReLU(),nn.UpsamplingBilinear2d(scale_factor=2.0),nn.Conv2d(128,32,3,1,1),nn.BatchNorm2d(32),nn.ReLU(),nn.UpsamplingBilinear2d(scale_factor=2.0),nn.Conv2d(32,2,1,1,0))

    def forward(self,x1,x2):
        x1_3, x1_4, x1_5, x1_6 = self.models(x1)
        x2_3, x2_4, x2_5, x2_6 = self.models(x2)
        c1_3, c2_3 = corr_fun(x1_3, x2_3, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1_3a, c2_3a = corr_fun(x1_3, x2_4, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1_3b, c2_3b = corr_fun(x1_3, x2_5, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1_3c, c2_3c = corr_fun(x1_3, x2_6, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1_4, c2_4 = corr_fun(x1_4, x2_4, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1_4a, c2_4a = corr_fun(x1_4, x2_3, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1_4b, c2_4b = corr_fun(x1_4, x2_5, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1_4c, c2_4c = corr_fun(x1_4, x2_6, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1_5, c2_5 = corr_fun(x1_5, x2_5, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1_5a, c2_5a = corr_fun(x1_5, x2_3, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1_5b, c2_5b = corr_fun(x1_5, x2_4, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1_5c, c2_5c = corr_fun(x1_5, x2_6, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1_6, c2_6 = corr_fun(x1_6, x2_6, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1_6a, c2_6a = corr_fun(x1_6, x2_3, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1_6b, c2_6b = corr_fun(x1_6, x2_4, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1_6c, c2_6c = corr_fun(x1_6, x2_5, self.Corr, self.poolopt_on_corrmat, self.new_indices)
        c1 = torch.cat((c1_3,c1_4,c1_5,c1_6,c1_3a,c1_4a,c1_5a,c1_6a,c1_3b,c1_4b,c1_5b,c1_6b,c1_3c,c1_4c,c1_5c,c1_6c),1)
        c2 = torch.cat((c2_3,c2_4,c2_5,c2_6,c2_3a,c2_4a,c2_5a,c2_6a,c2_3b,c2_4b,c2_5b,c2_6b,c2_3c,c2_4c,c2_5c,c2_6c),1)
        p1 = self.seg_out(c1)
        p2 = self.seg_out(c2)
        c1 = (torch.cat((x1_3, x1_4, x1_5, c1), 1)+F.softmax(p1, dim=1)[:,1:2])
        c2 = (torch.cat((x2_3, x2_4, x2_5, c2), 1)+F.softmax(p2, dim=1)[:,1:2])
        x1 = self.deeplab([x1_3, c1])
        x2 = self.deeplab([x2_3, c2])
        return x1,x2,F.interpolate(p1,scale_factor=8.0,mode='bilinear'),F.interpolate(p2,scale_factor=8.0,mode='bilinear')
    
    def _make_pred_layer(self, block, dilation_series, padding_series, inputscale, NoLabels):
        return block(dilation_series,padding_series,inputscale,NoLabels)


def SAFM(NoLabels, dim):
    w = outS(dim)
    h = outS(dim)
    model = SAFM_model(make_layers, NoLabels, h, w)
    return model
