from mmcv.cnn import ConvModule

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from .dpg_head import DPGHead
from .rcm import RCM, RCA

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class PyramidPoolAgg(nn.Module):
    def __init__(self, stride=2):
        super().__init__()
        self.stride = stride

    def forward(self, inputs):
        B, C, H, W = inputs[-1].shape
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1
        return torch.cat([nn.functional.adaptive_avg_pool2d(inp, (H, W)) for inp in inputs], dim=1)

class FuseBlockMulti(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int = 1,
        norm_cfg=dict(type='BN', requires_grad=True),
        activations = None,
    ) -> None:
        super(FuseBlockMulti, self).__init__()
        self.stride = stride
        self.norm_cfg = norm_cfg
        assert stride in [1, 2]

        if activations is None:
            activations = nn.ReLU

        self.fuse1 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.fuse2 = ConvModule(inp, oup, kernel_size=1, norm_cfg=self.norm_cfg, act_cfg=None)
        self.act = h_sigmoid()

    def forward(self, x_l, x_h):
        B, C, H, W = x_l.shape
        inp = self.fuse1(x_l)
        sig_act = self.fuse2(x_h)
        sig_act = F.interpolate(self.act(sig_act), size=(H, W), mode='bilinear', align_corners=False)
        out = inp * sig_act
        return out

class NextLayer(nn.Module):
    def __init__(self, block_num, embedding_dim, dw_size, module=RCM, mlp_ratio=2, token_mixer=RCA, square_kernel_size=3):#
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(module(embedding_dim, token_mixer=token_mixer, dw_size=dw_size, mlp_ratio=mlp_ratio, square_kernel_size=square_kernel_size))

    def forward(self, x):
        # token * N 
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return x

class SpatialGatherModule(nn.Module):
    """Aggregate the context features according to the initial predicted
    probability distribution.

    Employ the soft-weighted method to aggregate the context.
    """

    def __init__(self, scale):
        super(SpatialGatherModule, self).__init__()
        self.scale = scale

    def forward(self, feats, probs):
        """Forward function."""
        batch_size, num_classes, height, width = probs.size()
        channels = feats.size(1)
        probs = probs.view(batch_size, num_classes, -1)
        feats = feats.view(batch_size, channels, -1)
        # [batch_size, height*width, num_classes]
        feats = feats.permute(0, 2, 1)
        # [batch_size, channels, height*width]
        probs = F.softmax(self.scale * probs, dim=2)
        # [batch_size, channels, num_classes]
        ocr_context = torch.matmul(probs, feats)
        ocr_context = ocr_context.permute(0, 2, 1).contiguous().unsqueeze(3)
        return ocr_context

@HEADS.register_module()
class CGRSeg(BaseDecodeHead):
    def __init__(self, is_dw=False, next_repeat=4, mr=2, dw_size=7, neck_size=3, square_kernel_size=1, module='RCA', ratio=1, **kwargs):
        super(CGRSeg, self).__init__(input_transform='multiple_select', **kwargs)
        embedding_dim = self.channels

        self.linear_fuse = nn.Conv2d(
            in_channels=embedding_dim,
            out_channels=embedding_dim,
            kernel_size=1,
            stride=1,
            groups=embedding_dim if is_dw else 1,
            norm_cfg=self.norm_cfg, 
            act_cfg=self.act_cfg
        )
        self.ppa=PyramidPoolAgg(stride=2)
        norm_cfg = dict(type='SyncBN', requires_grad=True)
        act_layer=nn.ReLU6
        module_dict={
            'RCA':RCA,
        }
        self.trans=NextLayer(next_repeat, sum(self.in_channels), dw_size=neck_size, mlp_ratio=mr, token_mixer=module_dict[module], square_kernel_size=square_kernel_size)
        self.SIM = nn.ModuleList() 
        self.meta = nn.ModuleList() 
        for i in range(len(self.in_channels)):
            self.SIM.append(FuseBlockMulti(self.in_channels[i], self.channels, norm_cfg=norm_cfg, activations=act_layer))
            self.meta.append(RCM(self.in_channels[i],token_mixer=module_dict[module], dw_size=dw_size, mlp_ratio=mr, square_kernel_size=square_kernel_size, ratio=ratio))
        self.conv=nn.ModuleList()
        for i in range(len(self.in_channels)-1):
            self.conv.append(nn.Conv2d(self.channels, self.in_channels[i], 1))

        self.spatial_gather_module=SpatialGatherModule(1)
        self.lgc=DPGHead(embedding_dim, embedding_dim, pool='att', fusions=['channel_mul'])

    def forward(self, inputs):
        xx = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        out=self.ppa(xx)
        out = self.trans(out)
        f_cat = out.split(self.in_channels, dim=1)
        results = []
        for i in range(len(self.in_channels)-1,-1,-1):
            if i==len(self.in_channels)-1:
                local_tokens = xx[i]
            else:
                local_tokens = xx[i]+ self.conv[i](F.interpolate(results[-1], size=xx[i].shape[2:], mode='bilinear', align_corners=False))
            global_semantics = f_cat[i]
            local_tokens=self.meta[i](local_tokens)
            flag = self.SIM[i](local_tokens, global_semantics)
            results.append(flag)
        x = results[-1]
        _c = self.linear_fuse(x)
        prev_output = self.cls_seg(_c)

        context = self.spatial_gather_module(x, prev_output) #8*128*150*1
        object_context = self.lgc(x, context)+x #8*128*8*8
        output = self.cls_seg(object_context)

        return output

