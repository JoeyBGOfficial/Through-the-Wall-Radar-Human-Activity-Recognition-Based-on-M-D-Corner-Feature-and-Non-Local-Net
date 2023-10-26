# Configurations of your Program
# Former Author: nanting03, JoeyBG;
# Improved By: JoeyBG;
# Affiliation: Beijing Institute of Technology, Radar Research Lab;
# Date: 2023-9-1;
# Language & Platform: Python, Paddlepaddle Framework.
#
# Introduction:
# Make sure the classification output, paths, epoches, batch size, and initial learning rate are set to the value you want.
#

# Import necessary libs for Non-Local Net.
import paddle.nn as nn
import paddle
from non_local import get_nonlocal_block, NonLocalModule

# Choices for different Non-Local model construction.
__all__ = ['NLNet', 'nlnet18', 'nlnet34', 'nlnet50', 'nlnet101',
           'nlnet152', 'nlnext50_32x4d', 'nlnext101_32x8d']

# Self definition of 3×3 conv with padding.
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias_attr=False, dilation=dilation)

# Self definition of 1×1 conv with padding.
def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)

# Self definition of the channel attention module.
class SEBlock(nn.Layer):

    def __init__(self, planes, r=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2D(1)
        self.excitation = nn.Sequential(
            nn.Linear(planes, planes // r),
            nn.ReLU(),
            nn.Linear(planes // r, planes),
            nn.Sigmoid()
        )

    def forward(self, x):
        squeeze = self.squeeze(x)
        squeeze = paddle.reshape(squeeze, (squeeze.shape[0], -1))
        excitation = self.excitation(squeeze)
        excitation = paddle.reshape(excitation, (x.shape[0], x.shape[1], 1, 1))

        return x * paddle.expand_as(excitation, x)

# Definition of the basic block of Non-Local Net.
class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

# Definition of the bottleneck module of Non-Local Net.
class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_se=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when
        # stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.use_se = use_se
        self.se_block = SEBlock(
            planes * self.expansion)
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.se_block(out) if self.use_se else out
        out += identity
        out = self.relu(out)

        return out

# Construction of the Non-Local Net.
class NLNet(nn.Layer):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, nltype='nl', nl_mod=[2, 2, 1000],
                 k=4, transpose=False, nlsize=(7, 7, 7), dropout=0.2, use_se=False):
        super(NLNet, self).__init__()
        self.nltype = nltype
        if norm_layer is None:
            norm_layer = nn.BatchNorm2D
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.transpose = transpose
        self.dropout = dropout
        self.dilation = 1
        self.use_se = use_se
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2D(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias_attr=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0],
                                       nl_mod=nl_mod[0], s=nlsize[0], k=k)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1],
                                       nl_mod=nl_mod[1], s=nlsize[1], k=k)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2],
                                       nl_mod=nl_mod[2], s=nlsize[2], k=k)
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                m.weight=m.create_parameter(m.weight.shape, default_initializer=nn.initializer.KaimingNormal())
            elif isinstance(m, nn.BatchNorm2D):
                m.weight.set_value(paddle.ones(m.weight.shape))
                m.bias.set_value(paddle.zeros(m.bias.shape))

        for m in self.sublayers():
            if isinstance(m, NonLocalModule):
                m.init_modules()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.sublayers():
                if isinstance(m, Bottleneck):
                    m.bn3.weight.set_value(paddle.zeros(m.bn3.weight.shape))
                elif isinstance(m, BasicBlock):
                    m.bn2.weight.set_value(paddle.zeros(m.bn2.weight.shape))

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, nl_mod=1000, s=7, k=4):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []

        for i in range(blocks):
            if i == 0:
                layers.append((str(i), block(self.inplanes, planes, stride, downsample, self.groups,
                                             self.base_width, previous_dilation, norm_layer, use_se=self.use_se)))
                self.inplanes = planes * block.expansion
            else:
                layers.append((str(i), block(self.inplanes, planes, groups=self.groups,
                                             base_width=self.base_width, dilation=self.dilation,
                                             norm_layer=norm_layer, use_se=self.use_se)))
            if i % nl_mod == nl_mod - 1:
                if self.nltype == 'gc':
                    layers.append(
                    ('nl{}'.format(i), get_nonlocal_block(self.nltype)(self.inplanes,16)))
                    print('add {} after block {} with {} planes.'.format(
                        self.nltype, i, self.inplanes))
                else:
                    layers.append(
                        ('nl{}'.format(i), get_nonlocal_block(self.nltype)(self.inplanes, s=s, k=k, transpose=self.transpose, dropout=self.dropout)))
                    print('add {} after block {} with {} planes.'.format(
                            self.nltype, i, self.inplanes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = paddle.reshape(x, (x.shape[0], -1))
        x = self.fc(x)

        return x

# Construction of Non-Local Net 18 layers version.
def nlnet18(pretrained=False, **kwargs):
    """Constructs a NLNet-18 model.
    """
    model = NLNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

# Construction of Non-Local Net 34 layers version.
def nlnet34(pretrained=False, **kwargs):
    """Constructs a NLNet-34 model.
    """
    model = NLNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

# Construction of Non-Local Net 50 layers version.
def nlnet50(pretrained=False, **kwargs):
    """Constructs a NLNet-50 model.
    """
    model = NLNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

# Construction of Non-Local Net 101 layers version.
def nlnet101(pretrained=False, **kwargs):
    """Constructs a NLNet-101 model.
    """
    model = NLNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model

# Construction of Non-Local Net 152 layers version.
def nlnet152(pretrained=False, **kwargs):
    """Constructs a NLNet-152 model.
    """
    model = NLNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

# Construction of Non-Local Net 50_32x4d layers version.
def nlnext50_32x4d(pretrained=False, **kwargs):
    """Constructs a NLNeXt-50 32x4d model.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return NLNet(Bottleneck, [3, 4, 6, 3], **kwargs)

# Construction of Non-Local Net 101_32x8d layers version.
def nlnext101_32x8d(pretrained=False, **kwargs):
    """Constructs a NLNeXt-101 32x8d model.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return NLNet(Bottleneck, [3, 4, 23, 3], **kwargs)

# Main program for Non-Local Net running.
if __name__ == '__main__':
    network = nlnet50(num_classes=10)
    img = paddle.zeros([1, 3, 224, 224])
    outs = network(img)
    print(outs.shape)