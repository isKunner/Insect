from math import sqrt

import torch
from torch import nn
from CBAM_model import CBAM, ChannelAttention, SpatialAttention, i_CBAM, SpatialAttention1d


# 残差神经网络的主要贡献是发现了“退化现象（Degradation）”，并针对退化现象发明了 “快捷连接（Shortcut connection）”，极大的消除了深度过大的神经网络训练困难问题。


# 定义自注意力块，返回添加注意力后的矩阵
class FullAttention(nn.Module):
    def __init__(self, scale=None, attention_dropout=0.1):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        return V.contiguous()


# 注意力层
class AttentionLayer_start(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AttentionLayer_start, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.inner_attention = FullAttention(0.1)
        self.out_projection = nn.Linear(in_channel, out_channel)

    def forward(self, queries, keys, values):
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 1, 3)
        values = values.permute(0, 2, 1, 3)

        out = self.inner_attention(
            queries,
            keys,
            values
        )

        out = out.view(-1, 64, self.in_channel)

        return self.out_projection(out)


class AttentionLayer(nn.Module):
    def __init__(self):
        super(AttentionLayer, self).__init__()
        self.inner_attention = FullAttention(0.1)
        self.relu = nn.ReLU()

    def forward(self, queries, keys, values):

        queries = queries.unsqueeze(2)
        keys = keys.unsqueeze(2)
        values = values.unsqueeze(2)

        out = self.inner_attention(
            queries,
            keys,
            values
        )

        out = out.squeeze()

        return out


# 定义残差块，针对网络层数较少使用（34层）
class BasicBlock(nn.Module):
    expansion = 1  # 残差结构主分支中卷积核数量 在50、101、152层中每个残差结构中的卷积核数量可能不同

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        # downsample 下采样参数：除conv2_x，后面每个残差结构的第一层都是虚线残差结构，对应虚线残差结构（此时stride=2，既减少图像大小又升维）
        #             因为残差块的输出可能变化，因此对x也要进行变化

        super(BasicBlock, self).__init__()

        # 这里使用了倒残差结构

        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride,
                               bias=False,
                               padding=1)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.conv3 = nn.Conv1d(in_channels=out_channel*4, out_channels=out_channel, kernel_size=1, stride=1, bias=False)

        self.bn3 = nn.BatchNorm1d(out_channel)
        # self.relu = nn.ReLU()
        self.relu = nn.Sigmoid() # 当使用relu激活时，用上自注意力机制不收敛，虽然我不是很懂，但似乎是因为relu会导致数据太小，趋近于0？
        # 这里需要主义in_channel为偶数才对
        self.spatial_attention = SpatialAttention1d()
        self.downsample = downsample

        self.self_attention = AttentionLayer()

    def forward(self, x):

        identity = x

        # x, _ = self.spatial_attention(x)

        # 支线（捷径分支）
        if self.downsample is not None:
            # print(f"downsample: {self.downsample}")
            identity = self.downsample(x)

        # 主线
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.relu(out)

        # out = self.conv3(out)
        # out = self.bn3(out)

        out += identity

        # out = self.self_attention(out, out, out)

        out = self.relu(out)

        return out


# 定义残差块，针对高层
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4  # 一个残差块中的卷积核数量有变化

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=width, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(width)
        self.conv2 = nn.Conv1d(in_channels=width, out_channels=width, kernel_size=3, stride=stride, bias=False,
                               padding=1)
        self.bn2 = nn.BatchNorm1d(width)
        self.conv3 = nn.Conv1d(in_channels=width, out_channels=out_channel * self.expansion, kernel_size=1, stride=1,
                               bias=False)
        self.bn3 = nn.BatchNorm1d(out_channel * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 seq_length,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64,
                 visual_field=True,
                 attention=True,
                 audio_type='mfcc'
                 ):

        super(ResNet, self).__init__()

        self.include_top = include_top

        self.in_channel = 64  # 输入特征矩阵的深度（conv1->maxpooling后对应的深度）

        self.groups = groups

        self.width_per_group = width_per_group

        self.visual_field = visual_field

        self.atteneion = attention

        self.audio_type = audio_type

        self.channel_attention = ChannelAttention(4)

        self.self_attention_start = AttentionLayer_start(seq_length*4, seq_length)

        # 输入的是mfcc，单通道，通过不同大小的卷积核获得不同的视野从而得到不同的效果，这里是4个，其实也可以理解为多个头

        self.init_channel = 13

        self.conv0_1 = nn.Conv1d(self.init_channel, 64, kernel_size=1)
        self.conv0_3 = nn.Conv1d(self.init_channel, 64, kernel_size=3, padding=1)
        self.conv0_5 = nn.Conv1d(self.init_channel, 64, kernel_size=5, padding=2)
        self.conv0_7 = nn.Conv1d(self.init_channel, 64, kernel_size=7, padding=3)

        # 标准化
        self.conv0 = nn.Conv2d(4, 1, kernel_size=1)
        self.bn0 = nn.BatchNorm2d(4)
        self.relu = nn.ReLU(inplace=True)

        # 对应论文中的一系列残差结构
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # 进行最后的分类
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool1d(1)  # output size = (1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        # block就是上面的两个残差结构中的一个
        # block_num就是残差结构重复的次数
        # channel对应残差结构中的第一层对应的卷积核的个数
        downsample = None

        # channel * block.expansion 是残差结构输出的通道数
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channel * block.expansion))
            # nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
            # nn.BatchNorm2d(channel * block.expansion))

        layers = []

        # 因为第一层的残差结构会有变化，其余层次都不变，因此单独摘出来

        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))

        self.in_channel = channel * block.expansion

        # 全部都是实线的残差结构
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):

        # 对应conv1d卷积，所以先去除维度（去除通道那个维度，音频可以是单通道多通道，这里是单通道，下面通过可视野设置多个）
        x = x.squeeze()

        x = self.conv0_1(x)

        # if self.audio_type != 'mf-lf-gf' and self.audio_type != 'mf-lf-ng':
        #     if self.visual_field:
        #         x1 = self.conv0_1(x).unsqueeze(1)
        #         x2 = self.conv0_3(x).unsqueeze(1)
        #         x3 = self.conv0_5(x).unsqueeze(1)
        #         x4 = self.conv0_7(x).unsqueeze(1)
        #         x = torch.cat([x1, x2, x3, x4], dim=1)
        #
        # x, _ = self.channel_attention(x)
        # x = self.bn0(x)
        # x = self.relu(x)
        #
        # # print(x.shape)
        # # x = self.conv0(x).squeeze()
        #
        # # 自注意力层
        # x = self.self_attention_start(x, x, x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def resnet18(seq_length, num_classes=1000, include_top=True, visual_field=True, attention=True, audio_type='mfcc'):
    # https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    return ResNet(BasicBlock, [2, 2, 2, 2], seq_length=seq_length, num_classes=num_classes, include_top=include_top, audio_type=audio_type)


def resnet34(seq_length, num_classes=1000, include_top=True, visual_field=True, attention=True, audio_type='mfcc'):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], seq_length=seq_length, num_classes=num_classes, include_top=include_top, audio_type=audio_type)


def resnet50(seq_length, num_classes=1000, include_top=True, visual_field=True, attention=True, audio_type='mfcc'):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], seq_length=seq_length, num_classes=num_classes, include_top=include_top, audio_type=audio_type)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)