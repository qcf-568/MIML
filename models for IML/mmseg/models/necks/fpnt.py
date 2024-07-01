import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, xavier_init
from mmseg.ops import resize
from mmseg.models.backbones.swin import SwinBlockSequence
from timm.models.layers import trunc_normal_
from einops import rearrange

from mmseg.models.builder import NECKS

class PPM(nn.Module):
    """Pooling Pyramid Module used in PSPNet.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, conv_cfg, norm_cfg, act_cfg):
        super(PPM, self).__init__()
        pool_scales = (1, 2, 3, 6)
        self.align_corners = True#align_corners
        self.in_channels = 1024#in_channels
        self.channels = 256#channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.mls = nn.ModuleList()
        for pool_scale in pool_scales:
            self.mls.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        self.in_channels,
                        self.channels,
                        1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg,
                    )))
        self.conv = ConvModule(2048, 1024, 3, conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg, act_cfg=self.act_cfg)

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self.mls:
            ppm_out = ppm(x)
            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode='bilinear',
                align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        ppm_outs.append(x)
        return self.conv(torch.cat(ppm_outs,1))
        # return ppm_outs

# WBT: window based transformer
class SwinModule(nn.Module):
    def __init__(self,
                 dim, num_heads, window_size=7, depth=1,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.
                 ):
        super(SwinModule, self).__init__()
        self.swin_blocks = SwinBlockSequence(
                 embed_dims=dim,
                 depth=depth,
                 num_heads=num_heads,
                 window_size=window_size,
                 feedforward_channels= int(mlp_ratio * dim),
                 qkv_bias=qkv_bias,
                 qk_scale=qk_scale,
                 drop_rate=drop,
                 attn_drop_rate=attn_drop,
                 drop_path_rate=drop_path,
                 downsample=None,
                 with_cp=False)

    def forward(self, x):
        H, W = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x, _, _, _ = self.swin_blocks(x, (H, W))
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        return x


#          ┌──────────────────┐
# ┌────────┼───────┐┌─────────┼──────┐
# │      ┌┐├┐      ││       ┌┐▼┐     │
# │      ││││      ││       │││├─────┼──▶
# │      ││││      ││       ││││     │
# │      └┘▲┘      ││       └┘├┘     │
# │        │ ──────┼┼────────▶│      │
# │    ┌┐┌┐├┐┌┐    ││     ┌┐┌┐▼┐┌┐   │
# │    ││││││││    ││     │││││││├───┼──▶ ┏━━━━━○─────────│──────────┐
# │    ││││││││    ││     │┏━━━━━━━━┓━━━━━┛     │      ┌──▼─┐        │
# │    └┘└┘▲┘└┘    ││     └┃└┘├┘└┘  ┃│          │      │ Up │        │
# │        │ ──────┼┼──────┃─▶│     ┃│          │      └──┬─┘        │
# │  ┌┐┌┐┌┐├┐┌┐┌┐  ││   ┌┐┌┃┌┐▼┐┌┐┌┐┃│          │┌────┐  ┌▼┐   ┌────┐│
# │  ││││││││││││  ││   │││┃│││││││├┃┼──▶       ││1x1 ├─▶│+│──▶│WTB ──▶
# │  ││││││││││││  ││   │││┗━━━━━━━━┛━━━━━┓     │└────┘  └│┘   └────┘│
# │  └┘└┘└┘▲┘└┘└┘  ││   └┘└┘└┘├┘└┘└┘ │    ┗━━━━━○─────────│──────────┘
# │        │ ──────┼┼────────▶│      │                    ▼
# │┌┐┌┐┌┐┌┐├┐┌┐┌┐┌┐││┌┐┌┐┌┐┌┐┌▼┌┐┌┐┌┐│
# ││││││││││││││││││││││││││││││││││││
# ││││││││││││││││││││││││││││││││││├┼───▶
# │└┘└┘└┘└┘└┘└┘└┘└┘││└┘└┘└┘└┘└┘└┘└┘└┘│
# └───┬────────┬───┘└───┬────────┬───┘
#     │Backbone│        │  FPNT  │
#     └────────┘        └────────┘
@NECKS.register_module()
class FPNT(nn.Module):
    """Feature Pyramid Network Transformer.
        Adapted from mmseg/models/necks/fpn.py
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 #
                 depth_swin=1,
                 num_heads=8,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 ):
        super(FPNT, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.ppm = PPM(conv_cfg, norm_cfg, act_cfg)

        self.lateral_convs = nn.ModuleList()
        self.fpn_swins = nn.ModuleList()

        for i in range(self.num_ins):
            l_conv = ConvModule(in_channels[i], out_channels, 1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None, act_cfg=act_cfg, inplace=False)
            fpn_swin = SwinModule(dim=out_channels, depth=depth_swin, num_heads=num_heads, window_size=window_size,
                                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate)

            self.lateral_convs.append(l_conv)
            self.fpn_swins.append(fpn_swin)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        inputs = list(inputs)
        inputs[-1] = self.ppm(inputs[-1])
        inputs = tuple(inputs)
        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += F.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        outs = [
            self.fpn_swins[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        return tuple(outs)
