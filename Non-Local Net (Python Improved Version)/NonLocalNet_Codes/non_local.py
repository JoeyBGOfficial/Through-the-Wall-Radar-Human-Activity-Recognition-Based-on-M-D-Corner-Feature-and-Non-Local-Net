# Configurations of your Program
# Former Author: nanting03, JoeyBG;
# Improved By: JoeyBG;
# Affiliation: Beijing Institute of Technology, Radar Research Lab;
# Date: 2023-9-1;
# Language & Platform: Python, Paddlepaddle Framework.
#
# Introduction:
# This is the code used for constructing the modules of Non-Local mechanism, including traditional Non-Local and Non-Local with transformer attention.
#

# Import necessary libs for Non-Local module.
import paddle
from paddle import nn
from paddle.nn import functional as F
import math
import numpy as np
from context_block import ContextBlock

# Non-Local block input.
def get_nonlocal_block(block_type):
    block_dict = {'nl': NonLocal, 'bat': BATBlock, 'gc': ContextBlock}
    if block_type in block_dict:
        return block_dict[block_type]
    else:
        raise ValueError("UNKOWN NONLOCAL BLOCK TYPE:", block_type)

# Construction of the Non-Local module.
class NonLocalModule(nn.Layer):

    def __init__(self, in_channels, **kwargs):
        super(NonLocalModule, self).__init__()

    def init_modules(self):
        for m in self.sublayers():
            if len(m.sublayers()) > 0:
                continue
            if isinstance(m, nn.Conv2D):
                m.weight=m.create_parameter(m.weight.shape, default_initializer=nn.initializer.KaimingNormal())
                if len(list(m.parameters())) > 1:
                    m.bias.set_value(paddle.zeros(m.bias.shape))
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.zeros(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))
            elif isinstance(m, nn.GroupNorm):
                m.weight.set_value(paddle.zeros(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))
            elif len(list(m.parameters())) > 0:
                raise ValueError("UNKOWN NONLOCAL LAYER TYPE:", name, m)

# Construction of the Non-Local module.
class NonLocal(NonLocalModule):
    """Spatial NL block for image classification.
       [https://github.com/facebookresearch/video-nonlocal-net].
    """

    def __init__(self, inplanes, use_scale=True, **kwargs):
        planes = inplanes // 2
        self.use_scale = use_scale

        super(NonLocal, self).__init__(inplanes)
        self.t = nn.Conv2D(inplanes, planes, kernel_size=1,
                           stride=1, bias_attr=True)
        self.p = nn.Conv2D(inplanes, planes, kernel_size=1,
                           stride=1, bias_attr=True)
        self.g = nn.Conv2D(inplanes, planes, kernel_size=1,
                           stride=1, bias_attr=True)
        self.softmax = nn.Softmax(axis=2)
        self.z = nn.Conv2D(planes, inplanes, kernel_size=1,
                           stride=1, bias_attr=True)
        self.bn = nn.BatchNorm2D(inplanes)

    def forward(self, x):
        residual = x

        t = self.t(x)
        p = self.p(x)
        g = self.g(x)

        b, c, h, w = t.shape

        t = paddle.transpose(paddle.reshape(t, (b, c, -1)), (0, 2, 1))
        p = paddle.reshape(p, (b, c, -1))
        g = paddle.transpose(paddle.reshape(g, (b, c, -1)), (0, 2, 1))

        att = paddle.bmm(t, p)

        if self.use_scale:
            att = paddle.divide(att, paddle.to_tensor(c**0.5))

        att = self.softmax(att)
        x = paddle.bmm(att, g)

        x = paddle.transpose(x, (0, 2, 1))
        x = paddle.reshape(x, (b, c, h, w))

        x = self.z(x)
        x = self.bn(x) + residual

        return x


class BATransform(nn.Layer):

    def __init__(self, in_channels, s, k):
        super(BATransform, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2D(in_channels, k, 1),
                                   nn.BatchNorm2D(k),
                                   nn.ReLU())
        self.conv_p = nn.Conv2D(k, s * s * k, [s, 1])
        self.conv_q = nn.Conv2D(k, s * s * k, [1, s])
        self.conv2 = nn.Sequential(nn.Conv2D(in_channels, in_channels, 1),
                                   nn.BatchNorm2D(in_channels),
                                   nn.ReLU())
        self.s = s
        self.k = k
        self.in_channels = in_channels

    def extra_repr(self):
        return 'BATransform({in_channels}, s={s}, k={k})'.format(**self.__dict__)

    def resize_mat(self, x, t):
        n, c, s, s1 = x.shape
        assert s == s1
        if t <= 1:
            return x
        x = paddle.reshape(x, (n * c, -1, 1, 1))
        x = x * paddle.eye(t, t, dtype=x.dtype)
        x = paddle.reshape(x, (n * c, s, s, t, t))
        x = paddle.concat(paddle.split(x, 1, axis=1), axis=3)
        x = paddle.concat(paddle.split(x, 1, axis=2), axis=4)
        x = paddle.reshape(x, (n, c, s * t, s * t))
        return x

    def forward(self, x):
        out = self.conv1(x)
        rp = F.adaptive_max_pool2d(out, (self.s, 1))
        cp = F.adaptive_max_pool2d(out, (1, self.s))
        p = paddle.reshape(self.conv_p(rp), (x.shape[0], self.k, self.s, self.s))
        q = paddle.reshape(self.conv_q(cp), (x.shape[0], self.k, self.s, self.s))
        p = F.sigmoid(p)
        q = F.sigmoid(q)
        p = p / paddle.sum(p, axis=3, keepdim=True)
        q = q / paddle.sum(q, axis=2, keepdim=True)

        p = paddle.reshape(p, (x.shape[0], self.k, 1, self.s, self.s))
        p = paddle.expand(p, (x.shape[0], self.k, x.shape[1] // self.k, self.s, self.s))

        p = paddle.reshape(p, (x.shape[0], x.shape[1], self.s, self.s))

        q = paddle.reshape(q, (x.shape[0], self.k, 1, self.s, self.s))
        q = paddle.expand(q, (x.shape[0], self.k, x.shape[1] // self.k, self.s, self.s))

        q = paddle.reshape(q, (x.shape[0], x.shape[1], self.s, self.s))

        p = self.resize_mat(p, x.shape[2] // self.s)
        q = self.resize_mat(q, x.shape[2] // self.s)
        y = paddle.matmul(p, x)
        y = paddle.matmul(y, q)

        y = self.conv2(y)
        return y

# Construction of the BAT module.
class BATBlock(NonLocalModule):

    def __init__(self, in_channels, r=2, s=4, k=4, dropout=0.2, **kwargs):
        super().__init__(in_channels)

        inter_channels = in_channels // r
        self.conv1 = nn.Sequential(nn.Conv2D(in_channels, inter_channels, 1),
                                   nn.BatchNorm2D(inter_channels),
                                   nn.ReLU())
        self.batransform = BATransform(inter_channels, s, k)
        self.conv2 = nn.Sequential(nn.Conv2D(inter_channels, in_channels, 1),
                                   nn.BatchNorm2D(in_channels),
                                   nn.ReLU())
        self.dropout = nn.Dropout2D(p=dropout)

    def forward(self, x):
        xl = self.conv1(x)
        y = self.batransform(xl)
        y = self.conv2(y)
        y = self.dropout(y)
        return y + x

    def init_modules(self):
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                m.weight=m.create_parameter(m.weight.shape, default_initializer=nn.initializer.KaimingNormal())
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.ones(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))