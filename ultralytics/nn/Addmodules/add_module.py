import math
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import drop_path, SqueezeExcite
from timm.models.layers import CondConv2d, hard_sigmoid, DropPath

__all__ = (
    "C2f_GhostModule_DynamicConv",
    "DynamicConv",
    "GhostBottleneck",
    "BiFPN_Add2",
    "BiFPN_Add3",
    "BiFPN_Concat2",
    "BiFPN_Concat3",
    "BiFPN_Add2Conv",
    "BiFPN_Add3Conv",
    "SeModule",
    "Block",
    "MobileNetV3_Small",
    "InvertedResidual",
    "InvertedResidual_Relu",
    "InvertedResidual_HardWish",
    "Conv_Hardwish",
    "SPPF_LSKA",
)

from ultralytics.nn.modules import LSKA

_SE_LAYER = partial(SqueezeExcite, gate_fn=hard_sigmoid, divisor=4)


# BiFPN
# 两个特征图add操作
class BiFPN_Add2(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add2, self).__init__()
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型parameter
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)#一個形狀為(2,)的可学习参数
        self.epsilon = 0.0001#是一个小常数，用于避免在权重归一化时发生除以零的错误。
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)#卷积，k=1，s=1无填充，用于调整融合后的特征图通道数
        self.silu = nn.SiLU()#激活函数，相比于ReLU具有更平滑的梯度特性

    def forward(self, x):#输入 x 是一个列表，包含两个张量 x[0] 和 x[1]，分别是两个需要融合的特征图。
        w = self.w#获取可学习权重。
        weight = w / (torch.sum(w, dim=0) + self.epsilon)#对权重进行归一化，使其和为 1，避免过于偏向某一个输入。
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1]))#对两个输入特征图进行加权求和。silu对加权后的结果应用激活函数怒，conv对激活后的结果通过 1x1 卷积，调整特征图的通道数。


class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)


class BiFPN_Add2Conv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # 定义一个可学习参数；创建一个全是1的张量，张量包含两个元素，张量的维度为1；自动求导
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.cv = Conv(c1, c2, 1, 1)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        # 对可学习参数w进行归一化操作，将w中的值缩放，使它们表示的权重在[0, 1]范围内，并且它们的总和等于1
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        # 类比w0 x0 w1 x1是四个2*2矩阵，w0 * x0 + w1 * x1 仍然是一个2*2矩阵，只是里面的值发生了变化
        # 因此对应这里的xout的 N C H W 均未发生变化，只是内部的具体值发生变化，张量xout的维度不变
        xout = self.silu(weight[0] * x[0] + weight[1] * x[1])
        return self.cv(xout)


'''
        # 该部分添加至yolo.py
        # 这里的c2不会更新至args
        # 如果用这行命令，则在yaml文件中给定c1和c2参数时，必须满足c1=c2
        elif m in [BiFPN_Add2Conv, BiFPN_Add3Conv]:
            c2 = max(ch[x] for x in f)
        # 如果用这行命令，c2可以随意给定
        elif m in [BiFPN_Add2Conv, BiFPN_Add3Conv]:
            c2 = args[1]
'''


# 三个特征图add操作,然后进行1*1卷积
class BiFPN_Add3Conv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.cv = Conv(c1, c2, 1, 1)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        xout = self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2])
        return self.cv(xout)



# 三个特征图add操作
class BiFPN_Add3(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add3, self).__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        # Fast normalized fusion
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))


class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1], weight[2] * x[2]]
        return torch.cat(x, self.d)


class DynamicConv(nn.Module):
    """ Dynamic Conv layer
    """

    def __init__(self, in_features, out_features, kernel_size=1, stride=1, padding='', dilation=1,
                 groups=1, bias=False, num_experts=4):
        super().__init__()
        self.routing = nn.Linear(in_features, num_experts)
        self.cond_conv = CondConv2d(in_features, out_features, kernel_size, stride, padding, dilation,
                                    groups, bias, num_experts)

    def forward(self, x):
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        routing_weights = torch.sigmoid(self.routing(pooled_inputs))
        x = self.cond_conv(x, routing_weights)
        return x


class ConvBnAct(nn.Module):
    """ Conv + Norm Layer + Activation w/ optional skip connection
    """

    def __init__(
            self, in_chs, out_chs, kernel_size, stride=1, dilation=1, pad_type='',
            skip=False, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d, drop_path_rate=0., num_experts=4):
        super(ConvBnAct, self).__init__()
        self.has_residual = skip and stride == 1 and in_chs == out_chs
        self.drop_path_rate = drop_path_rate
        # self.conv = create_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.conv = DynamicConv(in_chs, out_chs, kernel_size, stride, dilation=dilation, padding=pad_type,
                                num_experts=num_experts)
        self.bn1 = norm_layer(out_chs)
        self.act1 = act_layer()

    def feature_info(self, location):
        if location == 'expansion':  # output of conv after act, same as block coutput
            info = dict(module='act1', hook_type='forward', num_chs=self.conv.out_channels)
        else:  # location == 'bottleneck', block output
            info = dict(module='', hook_type='', num_chs=self.conv.out_channels)
        return info

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.has_residual:
            if self.drop_path_rate > 0.:
                x = drop_path(x, self.drop_path_rate, self.training)
            x += shortcut
        return x


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, act_layer=nn.ReLU, num_experts=4):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            DynamicConv(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False, num_experts=num_experts),
            nn.BatchNorm2d(init_channels),
            act_layer() if act_layer is not None else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            DynamicConv(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False,
                        num_experts=num_experts),
            nn.BatchNorm2d(new_channels),
            act_layer() if act_layer is not None else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1, act_layer=nn.ReLU, se_ratio=0., drop_path=0., num_experts=4):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, act_layer=act_layer, num_experts=num_experts)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(
                mid_chs, mid_chs, dw_kernel_size, stride=stride,
                padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)
        else:
            self.conv_dw = None
            self.bn_dw = None

        # Squeeze-and-excitation
        self.se = _SE_LAYER(mid_chs, se_ratio=se_ratio,
                            act_layer=act_layer if act_layer is not nn.GELU else nn.ReLU) if has_se else None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, act_layer=None, num_experts=num_experts)

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                DynamicConv(
                    in_chs, in_chs, dw_kernel_size, stride=stride,
                    padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False, num_experts=num_experts),
                nn.BatchNorm2d(in_chs),
                DynamicConv(in_chs, out_chs, 1, stride=1, padding=0, bias=False, num_experts=num_experts),
                nn.BatchNorm2d(out_chs),
            )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.conv_dw is not None:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x = self.shortcut(shortcut) + self.drop_path(x)
        return x


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class C2f_GhostModule_DynamicConv(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(GhostModule(self.c, self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class SPPF_LSKA(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.lska = LSKA(c_ * 4, k_size=11)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(self.lska(torch.cat((x, y1, y2, self.m(y2)), 1)))

class Block(nn.Module):
    '''expand + depthwise + pointwise'''

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride

        # 扩展卷积
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        # 深度卷积
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)

        # SE 模块
        self.se = SeModule(expand_size) if se else nn.Identity()

        # 逐点卷积
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        # 跳跃连接
        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        # 扩展卷积 -> 深度卷积 -> SE 模块 -> 逐点卷积
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        # 跳跃连接
        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)


# InvertedResidual 模块实现
class InvertedResidual(nn.Module):
    def __init__(self, kernel_size, input_ch, expand_size, output_ch, activation, stride, use_se):
        super().__init__()
        # 1. 扩展层 (1x1 Conv)
        self.expand_conv = nn.Conv2d(input_ch, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = activation(inplace=True)

        # 2. 深度可分离卷积 (DW Conv)
        self.dw_conv = nn.Conv2d(
            expand_size, expand_size, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False
        )
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = activation(inplace=True)

        # 3. SE 模块 (可选)
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expand_size, expand_size // 4, 1),
                activation(inplace=True),
                nn.Conv2d(expand_size // 4, expand_size, 1),
                nn.Sigmoid()
            )

        # 4. 输出层 (1x1 Conv)
        self.project_conv = nn.Conv2d(expand_size, output_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(output_ch)

        # 5. 快捷连接
        self.shortcut = (stride == 1) and (input_ch == output_ch)

    def forward(self, x):
        identity = x
        x = self.act1(self.bn1(self.expand_conv(x)))
        x = self.act2(self.bn2(self.dw_conv(x)))
        if self.use_se:
            x = x * self.se(x)
        x = self.bn3(self.project_conv(x))
        return x + identity if self.shortcut else x
class InvertedResidual_HardWish(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size,stride, use_se):
        super().__init__()
        print("Hard_input_ch", input_ch)
        # 1. 扩展层 (1x1 Conv)
        expand_size = output_ch * 6
        self.expand_conv = nn.Conv2d(input_ch, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = nn.Hardswish(inplace=True)

        # 2. 深度可分离卷积 (DW Conv)
        self.dw_conv = nn.Conv2d(
            expand_size, expand_size, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False
        )
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = nn.Hardswish(inplace=True)

        # 3. SE 模块 (可选)
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expand_size, expand_size // 4, 1),
                nn.Hardswish(inplace=True),
                nn.Conv2d(expand_size // 4, expand_size, 1),
                nn.Sigmoid()
            )

        # 4. 输出层 (1x1 Conv)
        self.project_conv = nn.Conv2d(expand_size, output_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(output_ch)

        # 5. 快捷连接
        self.shortcut = (stride == 1) and (input_ch == output_ch)

        print("Hard_output_ch", output_ch)
    def forward(self, x):
        identity = x
        x = self.act1(self.bn1(self.expand_conv(x)))
        x = self.act2(self.bn2(self.dw_conv(x)))
        if self.use_se:
            x = x * self.se(x)
        x = self.bn3(self.project_conv(x))
        return x + identity if self.shortcut else x

class Conv_Hardwish(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.Hardswish()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        print("Conv input_channel:", c1,"Conv output_channel:", c2)
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()


    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))

class InvertedResidual_Relu(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size,stride, use_se):
        super().__init__()
        print("input_ch", input_ch)
        expand_size = output_ch * 6
        # 1. 扩展层 (1x1 Conv)
        self.expand_conv = nn.Conv2d(input_ch, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = nn.ReLU(inplace=True)

        # 2. 深度可分离卷积 (DW Conv)
        self.dw_conv = nn.Conv2d(
            expand_size, expand_size, kernel_size=kernel_size,
            stride=stride, padding=kernel_size // 2, groups=expand_size, bias=False
        )
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = nn.ReLU(inplace=True)

        # 3. SE 模块 (可选)
        self.use_se = use_se
        if use_se:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(expand_size, expand_size // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(expand_size // 4, expand_size, 1),
                nn.Sigmoid()
            )

        # 4. 输出层 (1x1 Conv)
        self.project_conv = nn.Conv2d(expand_size, output_ch, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(output_ch)

        # 5. 快捷连接
        self.shortcut = (stride == 1) and (input_ch == output_ch)
        print("Relu_output_ch", output_ch)

    def forward(self, x):
        identity = x
        x = self.act1(self.bn1(self.expand_conv(x)))
        x = self.act2(self.bn2(self.dw_conv(x)))
        if self.use_se:
            x = x * self.se(x)
        x = self.bn3(self.project_conv(x))
        return x + identity if self.shortcut else x

class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000, act=nn.Hardswish):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = act(inplace=True)

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU, True, 2),
            Block(3, 16, 72, 24, nn.ReLU, False, 2),
            Block(3, 24, 88, 24, nn.ReLU, False, 1),
            Block(5, 24, 96, 40, act, True, 2),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 240, 40, act, True, 1),
            Block(5, 40, 120, 48, act, True, 1),
            Block(5, 48, 144, 48, act, True, 1),
            Block(5, 48, 288, 96, act, True, 2),
            Block(5, 96, 576, 96, act, True, 1),
            Block(5, 96, 576, 96, act, True, 1),
        )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = act(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(576, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = act(inplace=True)
        self.drop = nn.Dropout(0.2)
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()


# if __name__ == "__main__":
#     # Generating Sample image
#     image_size = (1, 64, 256, 256)
#     image = torch.rand(*image_size)
#     # Model
#     model = C2f_GhostModule_DynamicConv(64,64)
#     # model = DynamicConv(64,64,3,stride=2)
#     out = model(image)
#     print(out.size())