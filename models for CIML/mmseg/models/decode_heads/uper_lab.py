# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from .sep_aspp_head import DepthwiseSeparableASPPHead 
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from .psp_head import PPM


@HEADS.register_module()
class UPerLab(BaseDecodeHead):

    def __init__(self, pool_scales=(1, 2, 3, 6), **kwargs):
        super(UPerLab, self).__init__(
            input_transform='multiple_select', **kwargs)
        # PSP Module
        self.deeplab = DepthwiseSeparableASPPHead(in_channels=2048,in_index=3,channels=512,dilations=(1, 12, 24, 36),c1_in_channels=256,c1_channels=48,dropout_ratio=0.1,num_classes=2,norm_cfg=dict(type='SyncBN', requires_grad=True),align_corners=False)
        self.convert = nn.Conv2d(512,256,1,1,0)
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
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
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

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + resize(
                laterals[i],
                size=prev_shape,
                mode='bilinear',
                align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])
        if self.training:
            cls_aux = self.cls_seg(fpn_outs[0])
        feat0 = self.convert(fpn_outs[0])
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[1].shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        fpn_outs[0] = resize(fpn_outs[0], size=fpn_outs[1].shape[2:], mode='bilinear', align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        if self.training:
            return (self.deeplab([feat0, fpn_outs], trans=False), cls_aux)# feats
        else:
            return self.deeplab([feat0, fpn_outs], trans=False)

    def forward_train(self, inputs, img_metas, gt_semantic_seg, train_cfg):
        seg_logits, aux_logits = self(inputs)
        losses = self.losses(seg_logits, gt_semantic_seg)
        losses_aux = self.losses(aux_logits, gt_semantic_seg, addstr='_uper')
        losses.update(losses_aux)
        return losses
