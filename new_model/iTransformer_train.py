# -*- coding: utf-8 -*-
import math

from iTransformer_model import iTransformer
import iTransformer_model
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from GetDatas import get_datas

# 忽略警告
warnings.filterwarnings('ignore')


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        # 使用了指数衰减的方式，每次训练周期结束时学习率减半
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        # 在第 2 个训练周期，学习率设置为 5e-5，在第 4 个训练周期设置为 1e-5，以此类推 (真巧妙呀)
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == "cosine":
        # 使用余弦退火策略调整学习率。学习率根据余弦函数的变化而变化，公式为 args.learning_rate / 2 * (1 + math.cos(epoch / args.train_epochs * math.pi))。这种策略通常用于在训练后期逐渐减小学习率，以实现更平稳的收敛。
        lr_adjust = {epoch: args.learning_rate /2 * (1 + math.cos(epoch / args.train_epochs * math.pi))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        # 遍历优化器中的所有参数组 optimizer.param_groups，并将它们的学习率更新为新的 lr 值
        # ，optimizer 对象的 param_groups 属性是一个列表，其中包含了优化器将要更新的所有参数组。每个参数组是一个字典，它定义了一组参数的优化设置。使用参数组可以为不同的参数应用不同的优化策略，例如不同的学习率、权重衰减、动量等。
        # 每个参数组是一个字典，包含以下键：
        # 'params': 一个参数列表，指定了这个参数组将要优化的模型参数。
        # 'lr': 学习率，用于此参数组中的参数。
        # 'weight_decay': 权重衰减项，用于正则化以防止过拟合。
        # 'momentum': 动量项，用于加速梯度下降。
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'iTransformer': iTransformer_model,
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    # 是一个抽象方法，用于构建模型。它应该在子类中被实现，以返回具体的模型实例。这里通过raise NotImplementedError表明这个方法需要被子类重写。
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            # CUDA_VISIBLE_DEVICES: 这是一个专门用于CUDA编程的环境变量，用来指定CUDA程序应该使用的GPU设备。通过设置这个环境变量，可以控制程序运行时使用的GPU。
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass


class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        # 使用 super() 调用父类的初始化方法
        super(Exp_Long_Term_Forecast, self).__init__(args)
        # 获取数据的类
        self.data = get_datas(audio_type='mfcc', batch_size=args.batch_size, n_fft=2048, win_length=2048, hop_length=512, n_xxcc=128)

    def _build_model(self):
        # 初始化模型，不过目前样例里面只写了iTransformer一种方法
        model = self.model_dict[self.args.model].iTransformer(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            # nn.DataParallel 是 PyTorch 中实现数据并行性的一个工具，它允许你将模型复制到多个 GPU 上，并且每个 GPU 运行模型的一个副本来处理输入数据的不同部分。
            # 数据分割: 它自动将输入数据分割成多个批次，每个批次由一个 GPU 处理
            # 模型副本: 每个 GPU 上都有一个模型的副本，它们在训练过程中并行运行
            # 在反向传播时，nn.DataParallel 会自动将所有 GPU 上的梯度合并，然后更新模型的参数
            # device_ids 是一个列表，指定了模型副本应该放置的 GPU 编号
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    # 获取数据和数据加载器
    def _get_data(self):
        # 这里没有删减原始的数据
        data_set, data_loader = ["", self.data.train_dl]
        return data_set, data_loader

    # 选择优化器，这里使用的是Adam优化器
    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    # 选择损失函数，这里使用的是均方误差损失（MSELoss），这个是指定的，不可变
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    # setting是路径
    def train(self, setting):

        train_data, train_loader = self._get_data()

        # 检查并创建模型检查点的保存路径
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        # 当前的实践
        time_now = time.time()

        # 训练步长
        train_steps = len(train_loader)

        # 优化器
        model_optim = self._select_optimizer()

        # 损失函数
        criterion = self._select_criterion()

        # torch.cuda.amp.GradScaler() 是 PyTorch 中用于自动混合精度训练的工具，它可以帮助用户在训练深度学习模型时提高性能并减少显存使用量。
        # 梯度缩放: GradScaler 可以将梯度缩放到较小的范围，以避免在训练过程中发生数值下溢或溢出的问题，同时保持足够的精度以避免模型性能下降
        # 自动混合精度: 它与 torch.autocast 结合使用，可以在前向传播中自动选择精度，以提高性能并保持准确性
        # 首先创建一个 GradScaler 对象。
        # 使用 with autocast() 将前向传递包装起来，启用混合精度计算。
        # 调用 scaler.scale(loss).backward() 来计算缩放后的梯度。
        # 使用 scaler.step(optimizer) 更新模型参数。
        # 使用 scaler.update() 更新 GradScaler 对象的内部状态。
        # 通过使用 GradScaler，可以在使用半精度浮点数（float16）进行计算的同时，减少由于梯度下溢导致的训练问题，从而加速模型训练
        # GradScaler 提供了一些参数，如 init_scale（初始缩放因子）、growth_factor（缩放因子增长系数）、backoff_factor（缩放因子下降系数）等，允许用户根据需要调整缩放行为
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 训练轮次
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()

            epoch_time = time.time()

            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                # 梯度清零
                model_optim.zero_grad()

                # 设置类型并放置到对应的设备上
                # 取出一个站点的对应时间步长的数据
                batch_x = batch_x.float().to(self.device)  # [B, F, T]  T是时间帧
                batch_y = batch_y.float().to(self.device)  # [B, F, T]

                # encoder - decoder
                if self.args.use_amp:
                    # 使用混合精度
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x)[0]
                        else:
                            outputs = self.model(batch_x)

                        # f_dim = -1 if self.args.features == 'MS' else 0
                        # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        # print(outputs.shape)
                        #
                        # print(batch_y.shape)

                        # 计算损失
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x)[0]
                    else:
                        outputs = self.model(batch_x)

                    # f_dim = -1 if self.args.features == 'MS' else 0
                    # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # 每100个批次输出一次结果
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    # 最近100次训练的速度 time_now会随时更新
                    speed = (time.time() - time_now) / iter_count
                    # 预计的剩余训练时长
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            # 训练损失
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            # 输出
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                epoch + 1, train_steps, train_loss))
            torch.save(self.model.state_dict(), path + '/' + f'checkpoint_{epoch}.pth')

            # 调整学习率
            adjust_learning_rate(model_optim, epoch + 1, self.args)

        return self.model


class config:
    def __init__(self):
        self.model_id = "V1"
        self.model = "iTransformer"
        self.data = "Meteorology"
        self.checkpoints = "./checkpoints"
        self.features = "MS"
        self.seq_len = 309
        self.label_len = 1
        self.pred_len = 1

        self.d_model = 64
        self.n_heads = 1
        self.e_layers = 1
        self.d_ff = 64
        self.des = 'golbal_temp'

        self.activation = "relu"
        self.devices = 0
        self.use_gpu = True
        self.gpu = 0
        self.use_multi_gpu = False
        self.output_attention = False

        self.loss = "MSE"
        self.lradj = "cosine"
        self.use_amp = "use automatic mixed precision training"
        self.learning_rate = 0.1
        self.batch_size = 64
        self.train_epochs = 10
        self.itr = 1
        self.num_workers = 1
        self.dropout = 0.1
        self.enc_in = 12


class iTransformer_config:
    def __init__(self, train_dl, val_dl, train_num, val_num, epochs=20, lr=0.001):

        self.train_dl = train_dl
        self.val_dl = val_dl
        self.train_num = train_num
        self.val_num = val_num
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 有GPU使用GPU，没有使用CPU
        self.epochs = epochs # 迭代轮次

        self.net = iTransformer(configs=config) # 两种类别，有虫或没虫，创建了模型
        self.model_weight_path = "./mobilenet_v2.pth"  # load pretrain weights  download url: https://download.pytorch.org/models/mobilenet_v2-b0353104.pth
        self.save_path = './MobileNet_v2_weight.pth' # 保存模型权重的未知
        self.freeze = False  # 是否冻结权重
        self.lr = lr  # 优化器
        self.train_steps = len(train_dl)  # 迭代步长

        print("using {} device.".format(self.device))
        # 放入设备
        self.net.to(self.device)
        # 判断此权重是否存在
        assert os.path.exists(self.model_weight_path), "file {} dose not exist.".format(self.model_weight_path)


if __name__ == '__main__':

    args = config()

    args.use_gpu = True if torch.cuda.is_available() else False

    print(torch.cuda.is_available())

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    Exp = Exp_Long_Term_Forecast

    for ii in range(args.itr):
        # setting record of experiments
        exp = Exp(args)  # set experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_df{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_ff,
            args.des, ii)

        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)