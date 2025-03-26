# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Block modules."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ADown",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "Silence",
    # "GhostBottleneckV2",
    # "GhostModuleV2",
    # "GhostNetV2",
    # "GhostModule",
    #
    # "GhostNet",
    # "C2f_Ghost",

)





class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


# class GhostModule(nn.Module):
#     def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
#         super(GhostModule, self).__init__()
#         self.oup = oup
#         init_channels = math.ceil(oup / ratio)
#         new_channels = init_channels * (ratio - 1)
#
#         self.primary_conv = nn.Sequential(
#             nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
#             nn.BatchNorm2d(init_channels),
#             nn.ReLU(inplace=True) if relu else nn.Sequential(),
#         )
#
#         self.cheap_operation = nn.Sequential(
#             nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
#             nn.BatchNorm2d(new_channels),
#             nn.ReLU(inplace=True) if relu else nn.Sequential(),
#         )
#
#     def forward(self, x):
#         x1 = self.primary_conv(x)
#         x2 = self.cheap_operation(x1)
#         out = torch.cat([x1, x2], dim=1)
#         return out[:, :self.oup, :, :]


# class GhostBottleneck(nn.Module):
#     """ Ghost bottleneck w/ optional SE"""
#
#     def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
#                  stride=1, act_layer=nn.ReLU, se_ratio=0.):
#         super(GhostBottleneck, self).__init__()
#         has_se = se_ratio is not None and se_ratio > 0.
#         self.stride = stride
#
#         # Point-wise expansion
#         self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)
#
#         # Depth-wise convolution
#         if self.stride > 1:
#             self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
#                                      padding=(dw_kernel_size - 1) // 2,
#                                      groups=mid_chs, bias=False)
#             self.bn_dw = nn.BatchNorm2d(mid_chs)
#
#         # Squeeze-and-excitation
#         if has_se:
#             self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
#         else:
#             self.se = None
#
#         # Point-wise linear projection
#         self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)
#
#         # shortcut
#         if (in_chs == out_chs and self.stride == 1):
#             self.shortcut = nn.Sequential()
#         else:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
#                           padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
#                 nn.BatchNorm2d(in_chs),
#                 nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_chs),
#             )
#
#     def forward(self, x):
#         residual = x
#
#         # 1st ghost bottleneck
#         x = self.ghost1(x)
#
#         # Depth-wise convolution
#         if self.stride > 1:
#             x = self.conv_dw(x)
#             x = self.bn_dw(x)
#
#         # Squeeze-and-excitation
#         if self.se is not None:
#             x = self.se(x)
#
#         # 2nd ghost bottleneck
#         x = self.ghost2(x)
#
#         x += self.shortcut(residual)
#         return x


# class GhostNet(nn.Module):
#     def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2):
#         super(GhostNet, self).__init__()
#         # setting of inverted residual blocks
#         self.cfgs = cfgs
#         self.dropout = dropout
#
#         # building first layer
#         output_channel = _make_divisible(16 * width, 4)
#         self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(output_channel)
#         self.act1 = nn.ReLU(inplace=True)
#         input_channel = output_channel
#
#         # building inverted residual blocks
#         stages = []
#         block = GhostBottleneck
#         for cfg in self.cfgs:
#             layers = []
#             for k, exp_size, c, se_ratio, s in cfg:
#                 output_channel = _make_divisible(c * width, 4)
#                 hidden_channel = _make_divisible(exp_size * width, 4)
#                 layers.append(block(input_channel, hidden_channel, output_channel, k, s,
#                                     se_ratio=se_ratio))
#                 input_channel = output_channel
#             stages.append(nn.Sequential(*layers))
#
#         output_channel = _make_divisible(exp_size * width, 4)
#         stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
#         input_channel = output_channel
#
#         self.blocks = nn.Sequential(*stages)
#
#         # building last several layers
#         output_channel = 1280
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
#         self.act2 = nn.ReLU(inplace=True)
#         self.classifier = nn.Linear(output_channel, num_classes)
#
#     def forward(self, x):
#         x = self.conv_stem(x)
#         x = self.bn1(x)
#         x = self.act1(x)
#         x = self.blocks(x)
#         x = self.global_pool(x)
#         x = self.conv_head(x)
#         x = self.act2(x)
#         x = x.view(x.size(0), -1)
#         if self.dropout > 0.:
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.classifier(x)
#         return x


# def ghostnet(**kwargs):
#     """
#     Constructs a GhostNet model
#     """
#     cfgs = [
#         # k, t, c, SE, s
#         # stage1
#         [[3, 16, 16, 0, 1]],
#         # stage2
#         [[3, 48, 24, 0, 2]],
#         [[3, 72, 24, 0, 1]],
#         # stage3
#         [[5, 72, 40, 0.25, 2]],
#         [[5, 120, 40, 0.25, 1]],
#         # stage4
#         [[3, 240, 80, 0, 2]],
#         [[3, 200, 80, 0, 1],
#          [3, 184, 80, 0, 1],
#          [3, 184, 80, 0, 1],
#          [3, 480, 112, 0.25, 1],
#          [3, 672, 112, 0.25, 1]
#          ],
#         # stage5
#         [[5, 672, 160, 0.25, 2]],
#         [[5, 960, 160, 0, 1],
#          [5, 960, 160, 0.25, 1],
#          [5, 960, 160, 0, 1],
#          [5, 960, 160, 0.25, 1]
#          ]
#     ]
#     return GhostNet(cfgs, **kwargs)


# def _make_divisible(v, divisor, min_value=None):
#     """
#     This function is taken from the original tf repo.
#     It ensures that all layers have a channel number that is divisible by 8
#     It can be seen here:
#     https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
#     """
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v


# def hard_sigmoid(x, inplace: bool = False):
#     if inplace:
#         return x.add_(3.).clamp_(0., 6.).div_(6.)
#     else:
#         return F.relu6(x + 3.) / 6.


# class SqueezeExcite(nn.Module):
#     def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
#                  act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
#         super(SqueezeExcite, self).__init__()
#         self.gate_fn = gate_fn
#         reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
#         self.act1 = act_layer(inplace=True)
#         self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)
#
#     def forward(self, x):
#         x_se = self.avg_pool(x)
#         x_se = self.conv_reduce(x_se)
#         x_se = self.act1(x_se)
#         x_se = self.conv_expand(x_se)
#         x = x * self.gate_fn(x_se)
#         return x


# class ConvBnAct(nn.Module):
#     def __init__(self, in_chs, out_chs, kernel_size,
#                  stride=1, act_layer=nn.ReLU):
#         super(ConvBnAct, self).__init__()
#         self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_chs)
#         self.act1 = act_layer(inplace=True)
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn1(x)
#         x = self.act1(x)
#         return x
#
#
# class GhostModuleV2(nn.Module):
#     def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True, mode=None, args=None):
#         super(GhostModuleV2, self).__init__()
#         self.mode = mode
#         self.gate_fn = nn.Sigmoid()
#
#         if self.mode in ['original']:
#             self.oup = oup
#             init_channels = math.ceil(oup / ratio)
#             new_channels = init_channels * (ratio - 1)
#             self.primary_conv = nn.Sequential(
#                 nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
#                 nn.BatchNorm2d(init_channels),
#                 nn.ReLU(inplace=True) if relu else nn.Sequential(),
#             )
#             self.cheap_operation = nn.Sequential(
#                 nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
#                 nn.BatchNorm2d(new_channels),
#                 nn.ReLU(inplace=True) if relu else nn.Sequential(),
#             )
#         elif self.mode in ['attn']:
#             self.oup = oup
#             init_channels = math.ceil(oup / ratio)
#             new_channels = init_channels * (ratio - 1)
#             self.primary_conv = nn.Sequential(
#                 nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
#                 nn.BatchNorm2d(init_channels),
#                 nn.ReLU(inplace=True) if relu else nn.Sequential(),
#             )
#             self.cheap_operation = nn.Sequential(
#                 nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
#                 nn.BatchNorm2d(new_channels),
#                 nn.ReLU(inplace=True) if relu else nn.Sequential(),
#             )
#             self.short_conv = nn.Sequential(
#                 nn.Conv2d(inp, oup, kernel_size, stride, kernel_size // 2, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.Conv2d(oup, oup, kernel_size=(1, 5), stride=1, padding=(0, 2), groups=oup, bias=False),
#                 nn.BatchNorm2d(oup),
#                 nn.Conv2d(oup, oup, kernel_size=(5, 1), stride=1, padding=(2, 0), groups=oup, bias=False),
#                 nn.BatchNorm2d(oup),
#             )
#
#     def forward(self, x):
#         if self.mode in ['original']:
#             x1 = self.primary_conv(x)
#             x2 = self.cheap_operation(x1)
#             out = torch.cat([x1, x2], dim=1)
#             return out[:, :self.oup, :, :]
#         elif self.mode in ['attn']:
#             res = self.short_conv(F.avg_pool2d(x, kernel_size=2, stride=2))
#             x1 = self.primary_conv(x)
#             x2 = self.cheap_operation(x1)
#             out = torch.cat([x1, x2], dim=1)
#             return out[:, :self.oup, :, :] * F.interpolate(self.gate_fn(res), size=(out.shape[-2], out.shape[-1]),
#                                                            mode='nearest')
#
#
# class GhostBottleneckV2(nn.Module):
#
#     def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
#                  stride=1, act_layer=nn.ReLU, se_ratio=0., layer_id=None, args=None):
#         super(GhostBottleneckV2, self).__init__()
#         has_se = se_ratio is not None and se_ratio > 0.
#         self.stride = stride
#
#         # Point-wise expansion
#         if layer_id <= 1:
#             self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True, mode='original', args=args)
#         else:
#             self.ghost1 = GhostModuleV2(in_chs, mid_chs, relu=True, mode='attn', args=args)
#
#             # Depth-wise convolution
#         if self.stride > 1:
#             self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
#                                      padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
#             self.bn_dw = nn.BatchNorm2d(mid_chs)
#
#         # Squeeze-and-excitation
#         if has_se:
#             self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
#         else:
#             self.se = None
#
#         self.ghost2 = GhostModuleV2(mid_chs, out_chs, relu=False, mode='original', args=args)
#
#         # shortcut
#         if (in_chs == out_chs and self.stride == 1):
#             self.shortcut = nn.Sequential()
#         else:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
#                           padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
#                 nn.BatchNorm2d(in_chs),
#                 nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
#                 nn.BatchNorm2d(out_chs),
#             )
#
#     def forward(self, x):
#         residual = x
#         x = self.ghost1(x)
#         if self.stride > 1:
#             x = self.conv_dw(x)
#             x = self.bn_dw(x)
#         if self.se is not None:
#             x = self.se(x)
#         x = self.ghost2(x)
#         x += self.shortcut(residual)
#         return x
#
#
# class GhostNetV2(nn.Module):
#     def __init__(self, cfgs, num_classes=1000, width=1.0, dropout=0.2, block=GhostBottleneckV2, args=None):
#         super(GhostNetV2, self).__init__()
#         self.cfgs = cfgs
#         self.dropout = dropout
#
#         # building first layer
#         output_channel = _make_divisible(16 * width, 4)
#         self.conv_stem = nn.Conv2d(3, output_channel, 3, 2, 1, bias=False)
#         self.bn1 = nn.BatchNorm2d(output_channel)
#         self.act1 = nn.ReLU(inplace=True)
#         input_channel = output_channel
#
#         # building inverted residual blocks
#         stages = []
#         # block = block
#         layer_id = 0
#         for cfg in self.cfgs:
#             layers = []
#             for k, exp_size, c, se_ratio, s in cfg:
#                 output_channel = _make_divisible(c * width, 4)
#                 hidden_channel = _make_divisible(exp_size * width, 4)
#                 if block == GhostBottleneckV2:
#                     layers.append(block(input_channel, hidden_channel, output_channel, k, s,
#                                         se_ratio=se_ratio, layer_id=layer_id, args=args))
#                 input_channel = output_channel
#                 layer_id += 1
#             stages.append(nn.Sequential(*layers))
#
#         output_channel = _make_divisible(exp_size * width, 4)
#         stages.append(nn.Sequential(ConvBnAct(input_channel, output_channel, 1)))
#         input_channel = output_channel
#
#         self.blocks = nn.Sequential(*stages)
#
#         # building last several layers
#         output_channel = 1280
#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.conv_head = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias=True)
#         self.act2 = nn.ReLU(inplace=True)
#         self.classifier = nn.Linear(output_channel, num_classes)
#
#     def forward(self, x):
#         x = self.conv_stem(x)
#         x = self.bn1(x)
#         x = self.act1(x)
#         x = self.blocks(x)
#         x = self.global_pool(x)
#         x = self.conv_head(x)
#         x = self.act2(x)
#         x = x.view(x.size(0), -1)
#         if self.dropout > 0.:
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.classifier(x)
#         return x



# def ghostnetv2(**kwargs):
#     cfgs = [
#         # k, t, c, SE, s
#         [[3, 16, 16, 0, 1]],
#         [[3, 48, 24, 0, 2]],
#         [[3, 72, 24, 0, 1]],
#         [[5, 72, 40, 0.25, 2]],
#         [[5, 120, 40, 0.25, 1]],
#         [[3, 240, 80, 0, 2]],
#         [[3, 200, 80, 0, 1],
#          [3, 184, 80, 0, 1],
#          [3, 184, 80, 0, 1],
#          [3, 480, 112, 0.25, 1],
#          [3, 672, 112, 0.25, 1]
#          ],
#         [[5, 672, 160, 0.25, 2]],
#         [[5, 960, 160, 0, 1],
#          [5, 960, 160, 0.25, 1],
#          [5, 960, 160, 0, 1],
#          [5, 960, 160, 0.25, 1]
#          ]
#     ]
#     return GhostNetV2(cfgs, num_classes=kwargs['num_classes'],
#                       width=kwargs['width'],
#                       dropout=kwargs['dropout'],
#                       args=kwargs['args'])

class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    # def __init__(self, c1, c2, k=5):
    def __init__(self, c1, c2, k=3):#根据SPPF池化核修改
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck with 2 convolutions module with arguments ch_in, ch_out, number, shortcut,
        groups, expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        # self.m = nn.ModuleList(GhostBottleneck(self.c, self.c, shortcut, g) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# class C2f_Ghost(nn.Module):
#     """ Faster Implementation of CSP Bottleneck with 2 convolutions using GhostBottleneck."""
#
#     def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
#         """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
#         expansion.
#         """
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#
#         # 使用 GhostBottleneck 替代原来的 Bottleneck
#         self.m = nn.ModuleList(
#             GhostBottleneck(self.c, self.c, s=1) for _ in range(n)  # n是GhostBottleneck的数量
#         )
#
#     def forward(self, x):
#         """Forward pass through C2f layer using GhostBottleneck."""
#         y = list(self.cv1(x).chunk(2, 1))  # 将特征图分成两个部分
#         y.extend(m(y[-1]) for m in self.m)  # 使用GhostBottleneck处理
#         return self.cv2(torch.cat(y, 1))  # 合并后输出
#
#     def forward_split(self, x):
#         """Forward pass using split() instead of chunk()."""
#         y = list(self.cv1(x).split((self.c, self.c), 1))  # 另一种分割方法
#         y.extend(m(y[-1]) for m in self.m)  # 使用GhostBottleneck处理
#         return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Contrastive Head for YOLO-World compute the region-text scores according to the similarity between image and text
    features.
    """

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcut option, groups and expansion
        ratio.
        """
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Rep CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class Silence(nn.Module):
    """Silence."""

    def __init__(self):
        """Initializes the Silence module."""
        super(Silence, self).__init__()

    def forward(self, x):
        """Forward pass through Silence layer."""
        return x


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super(CBLinear, self).__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        outs = self.conv(x).split(self.c2s, dim=1)
        return outs


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super(CBFuse, self).__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        out = torch.sum(torch.stack(res + xs[-1:]), dim=0)
        return out
