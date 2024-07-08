# -*- coding: utf-8 -*-

from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# 创建了一个上三角掩码（triangular mask），通常用于因果语言模型（如RNN或LSTM）的解码阶段，确保在生成序列时只能看到当前及之前的输出，而不是之后的输出，从而保持因果关系，但是这里没用到吧？
class TriangularCausalMask:
    # 批量大小（batch size） 序列长度（sequence length） device: 模型运行的设备，默认为 "cpu"
    def __init__(self, B, L, device="cpu"):
        # 定义了掩码的形状，这里是一个四维张量，通常用于与模型的输出相乘，以实现掩码效果
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            # 使用 torch.triu 函数创建一个上三角矩阵，其中对角线为1，其余为0
            # diagonal=1 参数指定对角线上方的元素为0，对角线及下方的元素为1。
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    # @property 装饰器定义了一个名为 mask 的属性，它返回私有变量 _mask 的值。这样，外部代码可以通过 mask 属性访问掩码，但不能直接修改它
    @property
    def mask(self):
        return self._mask


# 对编码进行翻转
class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        # [B, F, T] -> [B, T, F]  这里的N指的是特征，也就是Variate，一个因变量，两个协变量
        # x: [Batch Variate Time]\
        # 这里先不调整时间上的序列
        # x = x.permute(0, 2, 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # 这个x_mark有点迷惑
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)


class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        # 这里竟然是一维卷积
        # 感觉这么像残差层呢？但是没有短连接，好的好的后面有
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Q K V
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )

        # 这里+x看不懂哇  Transformer 架构的关键组成部分，有助于缓解深层网络中的梯度消失问题
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class iTransformer(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(iTransformer, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        # 嵌入层，对数据进行编码
        # [B, F, T] -> [B, T, F]
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)

        # Encoder
        # 编码器
        # 主要就是自注意力，其实其他都是卷积啥的，感觉就是套了一个transformer的外壳
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Decoder
        # 解码器就个线性层？不过似乎也是合理的
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        self.my_self_linear = nn.Linear(128, 1)

    def forecast(self, x_enc):
        # 这里主义进行了维度删减
        x_enc = x_enc.squeeze()

        # x_enc [batch_size, T, 37]

        # Normalization from Non-stationary Transformer
        # 计算 x_enc 张量在第二个维度（dim=1）上的平均值，即对每个时间步内的特征进行平均。keepdim=True 参数保证了结果的维度与原张量相同，这样可以直接用于后续的广播操作。
        # .detach() 方法将计算出的平均值从当前计算图中分离出来，这样它就不会在梯度计算中被考虑，通常在进行操作前确保不需要追踪梯度时使用。
        means = x_enc.mean(1, keepdim=True).detach()

        # 这行代码从原始数据 x_enc 中减去刚刚计算出的平均值 means，目的是使处理后的数据具有零均值。
        x_enc = x_enc - means

        # torch.var(x_enc, dim=1, keepdim=True, unbiased=False) 计算 x_enc 在第二个维度上的标准差。unbiased=False 参数表示使用有偏估计（即分母是 N 而不是 N-1），这在深度学习中更常见。
        # torch.sqrt(...) 对方差取平方根，得到标准差。
        # + 1e-5 是为了增加数值稳定性，防止分母为零，这在数值计算中称为“epsilon”。
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)

        # 这行代码将处理后的数据 x_enc 除以计算出的标准差 stdev，完成数据的标准化过程，使得数据具有单位方差。
        x_enc /= stdev

        # print(x_enc.shape)

        # N是特征数量
        _, _, N = x_enc.shape

        # Embedding
        # [B, T, L] -> [B, L, T] -> [B, L, D]
        enc_out = self.enc_embedding(x_enc, None)  # 这里其实就是简单的一个线性层，把T给编码了

        # print(enc_out.shape)

        # enc_out 是计算过注意力以后的结果
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # .permute() 函数用于重新排列张量的维度。当你对一个张量调用 .permute() 并指定一个维度索引的序列时，这个张量的形状将会根据你提供的索引序列重新排列。
        # [B, L, D] -> [B, L, P] -> [B, P, L] pred_len
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        dec_out = self.my_self_linear(dec_out).squeeze()

        # De-Normalization from Non-stationary Transformer
        # 将预测输出 dec_out 乘以每个特征的标准差 stdev，这是去标准化的第一步，目的是恢复数据的原始方差。

        # # 这块维度的变化还需要考虑
        # dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        # # 将均值 means 加回到去标准化后的数据中，完成去标准化过程，恢复数据的原始均值。
        # dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out
        # return dec_out[:, -self.pred_len:, :]  # [B, L, C]
