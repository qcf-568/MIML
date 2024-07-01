# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn import functional as F
from .sep_aspp_head2 import DepthwiseSeparableASPPHead2
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM
import scipy.stats as st

def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)

class HeadAttn(nn.Module):
    def __init__(self, conv_cfg, norm_cfg, act_cfg):
        super(HeadAttn, self).__init__()
        self.kg = nn.Sequential(nn.UpsamplingBilinear2d(size=(64,64)),
            ConvModule(1, 32, 5, padding=2, stride=2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg), # 32
            ConvModule(32, 64, 5, padding=2, stride=2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg), # 16
            ConvModule(64, 128, 5, padding=2, stride=2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg), # 8
            ConvModule(128, 256, 5, padding=2, stride=2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg), # 4
            ConvModule(256, 512, 3, padding=1, stride=2, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg), # 2
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.1),
            nn.Linear(512, 961),
        )

    def forward(self, attn):
        b = attn.size(0)
        kg = self.kg(attn).reshape(b,1,31,31)
        attn2 = torch.cat([F.conv2d(attn[i:i+1], kg[i:i+1], padding=15) for i in range(b)])
        attn2 = min_max_norm(attn2)
        return attn2.max(attn)


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction, conv_cfg, norm_cfg, act_cfg):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.c11 = ConvModule(
            2560,
            2048,
            1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

    def forward(self, x):
        return self.c11(x * self.cSE(x))


@HEADS.register_module()
class TestRFB1HA1CLab(BaseDecodeHead):
    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(TestRFB1HA1CLab, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.avg = nn.AdaptiveAvgPool2d(1)
        reduction = 8
        self.SE1 = nn.Sequential(nn.Conv2d(1024, 1024 // reduction, 1),nn.ReLU(inplace=True),nn.Conv2d(1024 // reduction, 2, 1),nn.Sigmoid(),)
        self.SE2 = nn.Sequential(nn.Conv2d(1536, 1536 // reduction, 1),nn.ReLU(inplace=True),nn.Conv2d(1536 // reduction, 3, 1),nn.Sigmoid(),)
        self.SE3 = nn.Sequential(nn.Conv2d(2048, 2048 // reduction, 1),nn.ReLU(inplace=True),nn.Conv2d(2048 // reduction, 4, 1),nn.Sigmoid(),)
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:
            l_conv = ConvModule(
                in_channels,
                self.channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                self.channels,
                self.channels,
                3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.MSDEC = DepthwiseSeparableASPPHead2(in_channels=2048,in_index=3,channels=512,dilations=(1, 12, 24, 36),c1_in_channels=256,c1_channels=48,dropout_ratio=0.1,num_classes=2,norm_cfg=dict(type='SyncBN', requires_grad=True),align_corners=False)
        self.convert = nn.Conv2d(512,256,1,1,0)
        self.ha = HeadAttn(self.conv_cfg, self.norm_cfg, self.act_cfg)
        self.CE = SCSEModule(2560, 8, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)
        self.ds = nn.UpsamplingBilinear2d(scale_factor=0.5)
        self.maxp = nn.AdaptiveMaxPool2d(1)
        self.cls_head = nn.Sequential(
                ConvModule(3072, 512, 1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False),
                nn.MaxPool2d(2,2),
                ConvModule(512, 256, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False),
                nn.MaxPool2d(2,2),
                ConvModule(256, 256, 3, padding=1, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg, inplace=False),
                nn.AdaptiveMaxPool2d(1),
                nn.Dropout(p=0.2),
                nn.Conv2d(256, 2, 1, 1, 0)
        )

    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            if i==3:
                wt = self.SE1(torch.cat((self.avg(laterals[2]), self.avg(laterals[3])), 1))
                laterals[2] = (wt[:,0:1] * laterals[2]) + (wt[:,1:2] * resize(laterals[3],size=prev_shape,mode='bilinear',align_corners=self.align_corners))
            elif i==2:
                wt = self.SE2(torch.cat((self.avg(laterals[1]), self.avg(laterals[2]), self.avg(laterals[3])), 1))
                laterals[1] = (wt[:,0:1] * laterals[1]) + (wt[:,1:2] * resize(laterals[2],size=prev_shape,mode='bilinear',align_corners=self.align_corners))+ (wt[:,2:3] * resize(laterals[3],size=prev_shape,mode='bilinear',align_corners=self.align_corners))
            elif i==1:
                wt = self.SE3(torch.cat((self.avg(laterals[0]), self.avg(laterals[1]), self.avg(laterals[2]), self.avg(laterals[3])), 1))
                laterals[0] = (wt[:,0:1] * laterals[0]) + (wt[:,1:2] * resize(laterals[1],size=prev_shape,mode='bilinear',align_corners=self.align_corners))+ (wt[:,2:3] * resize(laterals[2],size=prev_shape,mode='bilinear',align_corners=self.align_corners))+ (wt[:,3:4] * resize(laterals[3],size=prev_shape,mode='bilinear',align_corners=self.align_corners))

        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]

        fpn_outs.append(laterals[-1])
        cls_aux = [self.cls_seg(fpn_outs[0])]
        feat0 = self.convert(fpn_outs[0])
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(fpn_outs[i], size=fpn_outs[1].shape[2:], mode='bilinear', align_corners=self.align_corners)
        fpn_outs[0] = resize(fpn_outs[0], size=fpn_outs[1].shape[2:], mode='bilinear', align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        for cnt in range(3):
            pos_map = F.interpolate(F.softmax(cls_aux[-1], dim=1)[:,1:2], scale_factor=0.5, mode='bilinear')
            fpn_outs = (fpn_outs * self.ha(pos_map))
            lab_outs, fpn_adds = self.MSDEC([feat0, fpn_outs], trans=False)
            if cnt!=2:
                cls_aux.append(lab_outs)
                fpn_outs = self.CE(torch.cat((fpn_outs, fpn_adds), 1))
        if self.training:
            b,c,h,w = fpn_adds.shape
            cpred = self.cls_head(torch.stack((fpn_adds, fpn_outs[:,:512], fpn_outs[:,512:1024], fpn_outs[:,1024:1536], fpn_outs[:,1536:2048], self.ds(F.softmax(lab_outs,dim=1)[:,1:2]).expand_as(fpn_adds)),2).reshape(b,c*6,h,w).detach())
            return (lab_outs, cls_aux, cpred)# feats
        else:
            b,c,h,w = fpn_adds.shape
            cpred = self.cls_head(torch.stack((fpn_adds, fpn_outs[:,:512], fpn_outs[:,512:1024], fpn_outs[:,1024:1536], fpn_outs[:,1536:2048], self.ds(F.softmax(lab_outs,dim=1)[:,1:2]).expand_as(fpn_adds)),2).reshape(b,c*6,h,w).detach())
            return lab_outs, F.softmax(cpred,dim=1)[:,1:2].squeeze()

