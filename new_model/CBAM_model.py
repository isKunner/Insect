import torch
from torch import nn


# 对通道进行注意力机制
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        if in_channels < ratio:
            ratio = in_channels

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//ratio, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        avg_max = self.fc(self.max_pool(x))
        out = avg_out + avg_max
        out = self.sigmoid(out)
        return x*out, out


# 对空间进行注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size-1)//2

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out*x, out


# 对空间进行注意力机制
class SpatialAttention1d(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1d, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = (kernel_size-1)//2

        self.conv1 = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.sigmoid(self.conv1(out))
        return out*x, out


class CBAM(nn.Module):
    def __init__(self, in_channels, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channelattention = ChannelAttention(in_channels, ratio)
        self.spatialattention = SpatialAttention(kernel_size)

    def forward(self, x):

        x, _ = self.channelattention(x)
        x, _ = self.spatialattention(x)

        return x, 0  # 这里的作用纯粹是为了和上面匹配


class i_CBAM(nn.Module):
    def __init__(self, inchannels, ratio=16, kernel_size=3):
        super(i_CBAM, self).__init__()
        self.channelattention = ChannelAttention(inchannels, ratio)
        self.spatialattention = SpatialAttention(kernel_size)

    def forward(self, x):

        Mc = self.channelattention(x)[1]
        Ms = self.spatialattention(x)[1]

        # print(Mc)

        Mf = Mc * Ms

        return Mf * x