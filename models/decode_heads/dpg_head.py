import torch
from torch import nn
from mmcv.cnn import kaiming_init, constant_init


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
        m[-1].inited = True
    else:
        constant_init(m, val=0)
        m.inited = True


class DPGHead(nn.Module):
    def __init__(self, in_ch, mid_ch, pool, fusions):
        super(DPGHead, self).__init__()
        assert pool in ['avg', 'att']
        assert all([f in ['channel_add', 'channel_mul'] for f in fusions])
        assert len(fusions) > 0, 'at least one fusion should be used'
        self.inplanes = in_ch
        self.planes = mid_ch
        self.pool = pool
        self.fusions = fusions
        if 'att' in pool:
            self.conv_mask = nn.Conv2d(self.inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if 'channel_add' in fusions:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None
        if 'channel_mul' in fusions:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None
        self.reset_parameters()

    def reset_parameters(self):
        if self.pool == 'att':
            kaiming_init(self.conv_mask, mode='fan_in')
            self.conv_mask.inited = True

        if self.channel_add_conv is not None:
            last_zero_init(self.channel_add_conv)
        if self.channel_mul_conv is not None:
            last_zero_init(self.channel_mul_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pool == 'att':
            #[N, D, C, 1]
            input_x = x
            input_x = input_x.view(batch, channel, height*width) # [N, D, C]
            input_x = input_x.unsqueeze(1) # [N, 1, D, C]

            context_mask = self.conv_mask(x) # [N, 1, C, 1]
            context_mask = context_mask.view(batch, 1, height*width) # [N, 1, C]
            context_mask = self.softmax(context_mask) # [N, 1, C]
            context_mask = context_mask.unsqueeze(3) # [N, 1, C, 1]
            context = torch.matmul(input_x, context_mask)# [N, 1, D, 1]
            context = context.view(batch, channel, 1, 1) # [N, D, 1, 1]
        else:
            # [N, C, 1, 1]
            context = self.avg_pool(x)

        return context

    def forward(self, x, y):
        # [N, C, 1, 1]
        context = self.spatial_pool(y)

        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))# [N, D, 1, 1]
            out = x * channel_mul_term # [N, D, H, W]
        else:
            out = x
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)# [N, D, 1, 1]
            out = out + channel_add_term

        return out
