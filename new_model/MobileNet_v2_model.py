from torch import nn
import torch

# MobileNet v1 主要点
# Depthwise Convolution DW卷积层 --> Pointwise Convolution PW卷积
# DW卷积其实就是分层卷积的极端情况（分组个数为输入的通道数）层
# 假设一张图片24*24*3
# 要得到24*24*64的特征
# 原来：卷积核大小3*3*3*64(假设卷积核大小为3*3)
# 现在：DW卷积核3*3*3 --> PW卷积核1*1*64
# 增加超参数α(控制卷积核的个数)，β(分辨率)

# MobileNet v2 主要点
# Inverted Residuals 倒残差结构
# 正常残差：通过1*1降维，再卷积，再通过1*1升维 Relu激活函数
# 倒残差结构：通过1*1升维，再卷积，再通过1*1降维 Relu6激活函数（相比于Relu，改变的地方是输入值>6时全为6）
# Linear Bottlenecks


# 将卷积通道的个数ch设置为divisor的整数倍
from CBAM_model import ChannelAttention, CBAM


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            # group==out_channel时不进行特征组合，此时就是DW卷积， groups==1时和普通的卷积一样
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )


# 倒残差结构
class InvertedResidual(nn.Module):
    # expand_ratio 是扩展因子，即1*1卷积升维到原来的expand_ration倍
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        # 倒残差结构，中间通道多，中间隐藏层的通道数
        hidden_channel = in_channel * expand_ratio
        # 判断有没有捷径分支，放置梯度消失/爆炸
        # 捷径分支就是残差结构，论文中当stride==1并且输入输出相同时才有捷径分支
        self.use_shortcut = stride == 1 and in_channel == out_channel

        layers = []
        # 等于1时说明没有升维1*1卷积没必要
        if expand_ratio != 1:
            # 1x1 pointwise conv
            # 通过1*1卷积升维
            layers.append(ConvBNReLU(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3 depthwise conv
            # 注意这里的groups，减少计算量的重点，是DW卷积
            ConvBNReLU(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1 pointwise conv(linear)
            # 降维，linear函数，就不加激活函数了
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        # 使用捷径分支
        if self.use_shortcut:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    # alpha参数降低通道数（0~1）
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8, visual_field=True, attention=True, audio_type='mfcc'):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual

        self.visual_field = visual_field
        self.attention = attention
        self.audio_type = audio_type

        # 首先通过一个普通卷积将3通道（原来是做图形分类）升维至32，在进行倒残差结构
        # 32 * alpha 可能不是round_nearest的整数倍，这里通过此函数设置称为（可能是为了方便并行）
        input_channel = _make_divisible(32 * alpha, round_nearest)
        # 最后再经过普通卷积升维至1280
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        # 论文中网络的结构
        inverted_residual_setting = [
            # t, c, n, s
            # 扩展因子(倒残差结构中，中间升维的倍数)，通道数（卷积后的输出通道数，即当前卷积核的通道数），倒残差结构重复次数，步距（只针对n==1，其他时候都为1）
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # 因为这里的mfcc只有1个通道，所以需要升维
        self.conv0 = nn.Sequential(
            ConvBNReLU(1, 3, kernel_size=1, stride=1)
        )

        # 输入的是mfcc，单通道，通过不同大小的卷积核获得不同的视野从而得到不同的效果
        self.conv0_1 = ConvBNReLU(1, 1, kernel_size=1)
        self.conv0_3 = ConvBNReLU(1, 1, kernel_size=3)
        self.conv0_7 = ConvBNReLU(1, 1, kernel_size=7)

        features = []

        # conv1 layer
        features.append(ConvBNReLU(3, input_channel, stride=2))
        # building inverted residual residual blockes
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1 # 这里的步距，s指针对每个的第一次倒残差，后续几次全为1
                # block为倒残差结构
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        # building last several layers
        features.append(ConvBNReLU(input_channel, last_channel, 1))

        # combine feature layers
        self.features = nn.Sequential(*features)

        # 通道注意力层
        # self.channel_attenton = ChannelAttention(last_channel)
        self.channel_attention = ChannelAttention(3)
        # self.channel_attention = CBAM(3)

        # 分类器
        # building classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 输出结果为1*1
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        if self.audio_type != 'mf-lf-gf' and self.audio_type != 'mf-lf-ng':
            if self.visual_field:
                x1 = self.conv0_1(x)
                x2 = self.conv0_3(x)
                x3 = self.conv0_7(x)
                x = torch.cat([x1, x2, x3], dim=1)
            else:
                x = self.conv0(x) # 这是通道升维
        if self.attention:
            x, _ = self.channel_attention(x)
        x = self.features(x) # 倒残差啥啥的进行特征抽取
        x = self.avgpool(x)  # 平均池化
        x = torch.flatten(x, 1)  # 压扁
        x = self.classifier(x)  # 分类
        return x
