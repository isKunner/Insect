import numpy as np
import pandas as pd
import torch
from torch import nn


class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bias, is_conv1d=True):
        super(ConvLSTMCell, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if is_conv1d:
            self.padding = kernel_size//2
        else:
            self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.is_conv1d = is_conv1d
        if is_conv1d:
            self.conv = nn.Conv1d(in_channels=self.in_channels+self.out_channels,
                                  out_channels=4*self.out_channels, kernel_size=self.kernel_size,
                                  padding=self.padding, bias=self.bias)
        else:
            self.conv = nn.Conv2d(in_channels=self.in_channels+self.out_channels,
                                  out_channels=4*self.out_channels, kernel_size=self.kernel_size,
                                  padding=self.padding, bias=self.bias)

    def forward(self, input_tensor, last_state):

        # LSTM中，h是输出，c是记忆单元

        # input_tensor [bathc_size, in_channels, height, width]
        h_last, c_last = last_state  # 上一时刻输出  [batch_size, out_channels, height, width]
        combined_input = torch.cat([input_tensor, h_last], dim=1)  # 拼接  [batch_size, out_channels+in_channels, height, width]
        combined_conv = self.conv(combined_input)  # 这个直接卷积得到使是我没想到的，记得LSTM中是分别给了四个
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.out_channels, dim=1) # 分割位四部分(卷积后通道数*4)
        i = torch.sigmoid(cc_i)  # 输入门输出结果
        f = torch.sigmoid(cc_f)  # 遗忘门输出结果
        o = torch.sigmoid(cc_o)  # 输出门输出结果
        g = torch.sigmoid(cc_g)  # 输入层对应的输出结果
        c_next = f * c_last + i * g
        h_next = o*torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_layers, batch_first=True, bias=True, return_all_layers=False, is_conv1d=True):
        super(ConvLSTM, self).__init__()

        # num_layers 简单可以理解为进行num_layers次LSTM，所以响应的参数也得有num_layers个
        if not is_conv1d:
            self._check_kernel_size_consistency(kernel_size)
        # 卷积核的大小
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        # 每一层的LSTM中每个时刻输出的通道大小
        out_channels = self._extend_for_multilayer(out_channels, num_layers)
        if not len(kernel_size) == len(out_channels) == num_layers:
            raise ValueError("不合法")
        self.in_channels = in_channels  # 输入通道
        self.out_channels = out_channels  # 输出通道
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.is_conv1d = is_conv1d

        cell_list = []
        for i in range(0, self.num_layers):
            # 初始输入通道已知，后续输入通道就是前一个的输出通道
            cur_in_channels = self.in_channels if i == 0 else self.out_channels[i-1]
            cell_list.append(ConvLSTMCell(in_channels=cur_in_channels, bias=self.bias,
                                          out_channels=self.out_channels[i], kernel_size=self.kernel_size[i], is_conv1d=self.is_conv1d))
        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):

        if self.is_conv1d:
            # 判断批大小是否为第一个维度，不是则换到前面来
            if not self.batch_first:
                input_tensor = torch.permute(1, 0, 2, 3)

            batch_size, time_step, _, length = input_tensor.size()

            if hidden_state is not None:
                raise NotImplementedError()
            else:
                hidden_state = self.init_hidden(batch_size, length)  # 初始化为0，一个为h，一个为c

            cur_layer_input = input_tensor

            # layer_output_list 每个元素保存每一层的输出h [batch_size, time_step, out_channels, length]
            # last_state_list 保存每一层最后一个刻输出的h和c 即[(h, c), (h, c)...]

            layer_output_list, last_state_list = [], []

            # 再对每一层进行迭代
            for layer_idx in range(self.num_layers):
                h, c = hidden_state[layer_idx]
                output_inner = []
                cur_layer_cell = self.cell_list[layer_idx]  # 提取出当前层所对应的卷积层
                # 先对每一层的所有时间进行计算
                for t in range(time_step):
                    # cur_layer_input[:, t, :, :, :] [batch_size, 1, in_channels, height, width]
                    h, c = cur_layer_cell(cur_layer_input[:, t, :, :], [h, c])
                    output_inner.append(h)
                layer_output = torch.stack(output_inner, dim=1)  # 扩张维度，拓展出来时间维度
                cur_layer_input = layer_output  # 把这一层的输出当作下一层的输入
                layer_output_list.append(layer_output)
                last_state_list.append([h, c])
        else:
            # 判断批大小是否为第一个维度，不是则换到前面来
            if not self.batch_first:
                input_tensor = torch.permute(1, 0, 2, 3, 4)
            batch_size, time_step, _, height, width = input_tensor.size()
            if hidden_state is not None:
                raise NotImplementedError()
            else:
                hidden_state = self.init_hidden(batch_size, (height, width)) # 初始化为0，一个为h，一个为c

            cur_layer_input = input_tensor

            # layer_output_list 每个元素保存每一层的输出h [batch_size, time_step, out_channels, heigth, width]
            # last_state_list 保存每一层最后一个刻输出的h和c 即[(h, c), (h, c)...]

            layer_output_list, last_state_list = [], []

            # 再对每一层进行迭代
            for layer_idx in range(self.num_layers):
                h, c = hidden_state[layer_idx]
                output_inner = []
                cur_layer_cell = self.cell_list[layer_idx]  # 提取出当前层所对应的卷积层
                # 先对每一层的所有时间进行计算
                for t in range(time_step):
                    # cur_layer_input[:, t, :, :, :] [batch_size, 1, in_channels, height, width]
                    h, c = cur_layer_cell(cur_layer_input[:, t, :, :, :], [h, c])
                    output_inner.append(h)
                layer_output = torch.stack(output_inner, dim=1)  # 扩张维度，拓展出来时间维度
                cur_layer_input = layer_output  # 把这一层的输出当作下一层的输入
                layer_output_list.append(layer_output)
                last_state_list.append([h, c])

        # 只输出最后一层
        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]
        return layer_output_list, last_state_list

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        # 传入的kernel_size要么是一个元组如(3, 3)，要么是一个包含有多个元组的列表如[(3, 3), (5, 5)]，
        # 前者表示所有层的卷积核窗口大小均为(3,3)，后者表示在两层的ConvLSTM中卷积核的窗口大小分别为(3, 3)和(5, 5)
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and
                 all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('kernel_size must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param

    def init_hidden(self, batch_size, image_size):

        if self.is_conv1d:
            length = image_size
            return [(torch.zeros(batch_size, i, length, device="cuda:0" if torch.cuda.is_available() else "cpu"),
                     torch.zeros(batch_size, i, length, device="cuda:0" if torch.cuda.is_available() else "cpu"))
                    for i in self.out_channels]
        else:
            height, width = image_size
            return [(torch.zeros(batch_size, i, height, width, device="cuda:0" if torch.cuda.is_available() else "cpu"),
                    torch.zeros(batch_size, i, height, width, device="cuda:0" if torch.cuda.is_available() else "cpu")) for i in self.out_channels]


class ConvLSTM_audio(nn.Module):

    def __init__(self, num_classes, width, in_channels, out_channels, kernel_size, num_layers, batch_size, height=0, is_conv1d=True):
        super(ConvLSTM_audio, self).__init__()
        self.conv_lstm = ConvLSTM(in_channels, out_channels, kernel_size, num_layers, batch_size, is_conv1d=is_conv1d)

        if is_conv1d:
            self.max_pool = nn.MaxPool2d(kernel_size=5, stride=2, padding=2)  # 最大池化
            # [batch_size, time_step, out_channels, heigth, width]
            self.hidden_dim = width // 4 * self.conv_lstm.out_channels[-1]  # 计算拉伸后的维度 //2是因为上面的步长为2，啊，这块除以4是我不能理解的
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(self.hidden_dim, num_classes))
        else:
            self.max_pool = nn.MaxPool2d(kernel_size=(5, 5), stride=2, padding=2)  # 最大池化
            # [batch_size, time_step, out_channels, heigth, width]
            self.hidden_dim = (width*height) // 4 * self.conv_lstm.out_channels[-1]  # 计算拉伸后的维度 //4是因为上面的步长为2，长宽均//2了
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(self.hidden_dim, num_classes))

    @classmethod
    def config_method(cls, config):
        return cls(config.num_classes, config.width, config.in_channels, config.out_channels, config.kernel_size, config.num_layers, config.batch_size, config.height, config.is_conv1d)

    def forward(self, x, labels=None):
        # print(x.shape)
        _, layer_output = self.conv_lstm(x)  # layer_output = [h, c]  [batch_size,out_channels,height,width]
        pool_output = self.max_pool(layer_output[-1][0]) # 最后一刻的输出
        # print(pool_output.shape)
        logits = self.classifier(pool_output)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits


if __name__ == '__main__':
    print()