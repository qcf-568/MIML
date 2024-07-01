import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import force_fp32
from mmseg.ops import resize
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.losses import accuracy

from timm.models.vision_transformer import Attention#, Mlp
from timm.models.layers import DropPath
from einops import rearrange


class Mlp(nn.Module):
    """ Multilayer perceptron from timm library."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossAttention(nn.Module):
    '''
    Taken from timm library Attention module
    with slight modifications to do Cross-Attention.
    '''
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
    def forward(self,q_in, kv_in):
        B, N, C = kv_in.shape
        _, L, _ = q_in.shape
        # Create key and value tokens
        kv = self.kv(kv_in).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # Create query tokens
        q = self.to_q(q_in)
        q = q.reshape(B, L, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

#--- Decoder Block ---
#         │
#         ├────────┐
# ┌───────┴───────┐│
# │      MLP      ││
# └───────┬───────┘│
#         ├────────┘
#         ├────────┐
# ┌───────┴───────┐│
# │Self-Attention ││
# └───────┬───────┘│
#         ├────────┘
#         ├────────┐
# ┌───────┴───────┐│
# │Cross-Attention││
# └───────┬───────┘│
#         ├────────┘
#         │
class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads,mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=0.)
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads,qkv_bias=qkv_bias, qk_scale=qk_scale,
                                            attn_drop=attn_drop, proj_drop=0.)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, act_layer=act_layer, drop=drop)

    def forward_crossattn(self, queries, features):
        ##--------- Cross-Attention Block
        out = queries + self.drop_path1(self.cross_attn(self.norm1(queries), features))
        return out

    def forward_attn(self, q):
        ##--------- Self-Attention Block
        q = q + self.drop_path2(self.attn(self.norm2(q)))
        return q

    def forward_mlp(self, q):
        ##--------- MLP Block
        cls_features = q + self.drop_path3(self.mlp(self.norm3(q)))
        return cls_features

    def forward(self, queries, features):
        out = self.forward_crossattn(queries, features)
        out = self.forward_attn(out)
        out = self.forward_mlp(out)
        return out

class TransformerLearner(nn.Module):
    def __init__(self, dim, num_heads, num_queries, branch_depth, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super(TransformerLearner, self).__init__()
        dpr = [x.item() for x in torch.linspace(0, drop_path, branch_depth)]
        self.layers = nn.ModuleList([
            DecoderBlock(dim, num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                 attn_drop=attn_drop, drop_path=dpr[i], act_layer=act_layer, norm_layer=norm_layer)
            for i in range(branch_depth)
        ])
        # Learnable class embeddings
        self.queries = nn.Parameter(torch.zeros(1, num_queries, dim), requires_grad=True)
        # Norm features and class embeddings for training stability
        self.norm_features = nn.LayerNorm(dim)
        self.norm_embs = nn.LayerNorm(dim)

    def forward(self, features):
        B, _, H, W = features.shape
        # Tokenize 2D feature maps
        features = rearrange(features, 'b c h w -> b (h w) c')
        features = self.norm_features(features)

        # Expand to batch size
        cls_embs = self.queries.expand(B, -1, -1)

        # Decoder
        for layer in self.layers:
            cls_embs = layer(cls_embs, features)

        # Norm class embeddings for stability
        cls_embs = self.norm_embs(cls_embs)
        # Prediction
        pred = (features @ cls_embs.transpose(-2, -1))
        # Reshape into 2D maps
        pred = rearrange(pred, 'b (h w) c -> b c h w', h=H, w=W)

        return pred


#          ┌──────────────────┐
# ┌────────┼───────┐┌─────────┼──────┐
# │      ┌┐├┐      ││       ┌┐▼┐     │  ┌─────────┐
# │      ││││      ││       │││├─────┼──▶ Learner ────┐
# │      ││││      ││       ││││     │  └─────────┘   │
# │      └┘▲┘      ││       └┘├┘     │                │
# │        │ ──────┼┼────────▶│      │                │
# │    ┌┐┌┐├┐┌┐    ││     ┌┐┌┐▼┐┌┐   │  ┌─────────┐   │
# │    ││││││││    ││     │││││││├───┼──▶ Learner ──┐ │
# │    ││││││││    ││     ││││││││   │  └─────────┘ │ │
# │    └┘└┘▲┘└┘    ││     └┘└┘├┘└┘   │             ┌▼─▼┐ ┌─────┐
# │        │ ──────┼┼────────▶│      │             │ + ├─▶Pred │
# │  ┌┐┌┐┌┐├┐┌┐┌┐  ││   ┌┐┌┐┌┐▼┐┌┐┌┐ │             └▲─▲┘ └─────┘
# │  ││││││││││││  ││   ││││││││││││ │  ┌─────────┐ │ │
# │  ││││││││││││  ││   │││││││││││├─┼──▶ Learner ──┘ │
# │  └┘└┘└┘▲┘└┘└┘  ││   └┘└┘└┘├┘└┘└┘ │  └─────────┘   │
# │        │ ──────┼┼────────▶│      │                │
# │┌┐┌┐┌┐┌┐├┐┌┐┌┐┌┐││┌┐┌┐┌┐┌┐┌▼┌┐┌┐┌┐│                │
# ││││││││││││││││││││││││││││││││││││  ┌─────────┐   │
# ││││││││││││││││││││││││││││││││││├┼──┼▶Learner ────┘
# │└┘└┘└┘└┘└┘└┘└┘└┘││└┘└┘└┘└┘└┘└┘└┘└┘│  └─────────┘
# └───┬────────┬───┘└───┬────────┬───┘
#     │Backbone│        │  FPNT  │
#     └────────┘        └────────┘
# ╔════════════════════════════════════════════════════════════╗
# ║        ____ ____ _  _ ____ ____ ____ _  _ ____ ____        ║
# ║        [__  |___ |\ | |___ |  | |__/ |\/| |___ |__/        ║
# ║        ___] |___ | \| |    |__| |  \ |  | |___ |  \        ║
# ║                                                            ║
# ╚════════════════════════════════════════════════════════════╝
@HEADS.register_module()
class SenFormer(BaseDecodeHead):

    def __init__(self, feature_strides, num_heads, branch_depth, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eps=1.e-15, **kwargs):
        super(SenFormer, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.eps = eps

        self.learners = nn.ModuleList()
        for i in range(len(feature_strides)):
            self.learners.append(
                TransformerLearner(dim=self.in_channels[i],
                                 num_heads=num_heads,
                                 num_queries=self.num_classes,
                                 branch_depth=branch_depth,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path,
                                 act_layer=act_layer,
                                 norm_layer=norm_layer)
            )

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        # -------
        prob_outputs = [] # save probabilty maps for ensemble prediction
        logit_outputs = [] # save logits for learners' supervision
        for i in range(0, len(self.feature_strides)):
            # learner prediction
            learner_pred = self.learners[i](x[i])
            learner_pred = resize(learner_pred, size=x[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            # -------
            logit_outputs.append(learner_pred)
            prob_outputs.append(F.softmax(learner_pred, dim=1))

        # Ensemble prediction
        ensemble_pred = torch.stack(prob_outputs, dim=0).sum(dim=0)

        return logit_outputs, ensemble_pred

    def forward_test(self, inputs, img_metas, test_cfg):
        _, ensemble_pred = self.forward(inputs)
        return ensemble_pred

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        logit_outputs, ensemble_pred = seg_logit # unpack

        # Upscale outputs to the ground truth size
        ## Ensemble predicition
        ensemble_pred = torch.log(torch.clamp(ensemble_pred, min=self.eps))
        ensemble_pred =  resize(input=ensemble_pred, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        ## Learners predicitions
        seg_log_logit = [resize(input=logit_outputs[i], size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
                         for i in range(len(logit_outputs))]

        seg_label = seg_label.squeeze(1)
        # Losses
        loss = dict()
        ## Loss for the ensemble prediction
        loss_classic = F.cross_entropy(ensemble_pred, seg_label, ignore_index=self.ignore_index)
        ## Loss for each learner
        losses_seg = [F.cross_entropy(seg_log_logit[i], seg_label, ignore_index=self.ignore_index) for i in
                      range(len(seg_log_logit))]
        loss_extra = torch.stack(losses_seg, dim=0).sum()
        loss_extra = loss_extra / len(seg_log_logit)

        loss['loss_classic'] = loss_classic
        loss['loss_seg'] = loss_extra
        loss['acc_seg'] = accuracy(ensemble_pred, seg_label)
        return loss


@HEADS.register_module()
class SenFormerNWS(BaseDecodeHead):
    '''
    SenFormer with No Weight Sharing (NWS)
    '''
    def __init__(self, feature_strides, num_heads, branch_depth, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, eps=1.e-15, **kwargs):
        super(SenFormer, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides
        self.eps = eps

        self.learners = nn.ModuleList()
        for i in range(len(feature_strides)):
            self.learners.append(
                TransformerLearner(dim=self.in_channels[i],
                                 num_heads=num_heads,
                                 num_queries=self.num_classes,
                                 branch_depth=branch_depth,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path,
                                 act_layer=act_layer,
                                 norm_layer=norm_layer)
            )

    def forward(self, inputs):
        x = self._transform_inputs(inputs)
        # -------
        prob_outputs = [] # save probabilty maps for ensemble prediction
        logit_outputs = [] # save logits for learners' supervision
        for i in range(0, len(self.feature_strides)):
            # learner prediction
            learner_pred = self.learners[i](x[i])
            learner_pred = resize(learner_pred, size=x[0].shape[2:], mode='bilinear', align_corners=self.align_corners)
            # -------
            logit_outputs.append(learner_pred)
            prob_outputs.append(F.softmax(learner_pred, dim=1))

        # Ensemble prediction
        ensemble_pred = torch.stack(prob_outputs, dim=0).sum(dim=0)

        return logit_outputs, ensemble_pred

    def forward_test(self, inputs, img_metas, test_cfg):
        _, ensemble_pred = self.forward(inputs)
        return ensemble_pred

    @force_fp32(apply_to=('seg_logit',))
    def losses(self, seg_logit, seg_label):
        """Compute segmentation loss."""
        logit_outputs, ensemble_pred = seg_logit # unpack

        # Upscale outputs to the ground truth size
        ## Ensemble predicition
        ensemble_pred = torch.log(torch.clamp(ensemble_pred, min=self.eps))
        ensemble_pred =  resize(input=ensemble_pred, size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
        ## Learners predicitions
        seg_log_logit = [resize(input=logit_outputs[i], size=seg_label.shape[2:], mode='bilinear', align_corners=self.align_corners)
                         for i in range(len(logit_outputs))]

        seg_label = seg_label.squeeze(1)
        # Losses
        loss = dict()
        ## Loss for the ensemble prediction
        loss_classic = F.cross_entropy(ensemble_pred, seg_label, ignore_index=self.ignore_index)
        ## Loss for each learner
        losses_seg = [F.cross_entropy(seg_log_logit[i], seg_label, ignore_index=self.ignore_index) for i in
                      range(len(seg_log_logit))]
        loss_extra = torch.stack(losses_seg, dim=0).sum()
        loss_extra = loss_extra / len(seg_log_logit)

        loss['loss_classic'] = loss_classic
        loss['loss_seg'] = loss_extra
        loss['acc_seg'] = accuracy(ensemble_pred, seg_label)
        return loss
