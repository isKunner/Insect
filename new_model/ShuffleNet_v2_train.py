# -*- coding:utf-8 -*-
import os
import math
import sys

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm

from ShuffleNet_v2_model import shufflenet_v2_x1_0

from GetDatas import get_datas

# ShuffleNet_v2 输入的图像要求是图像格式，即三通道（大小无限制）
# 为适应模型，预处理得到的是MFCC图，即1*n*m的，所以给模型加了一个1*1*1*3的卷积核以升维conv0

class ShuffleNet_v2_config():
    def __init__(self, train_dl, val_dl, train_num, val_num, epochs=20, lr=0.001):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.train_num = train_num
        self.val_num = val_num
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 有GPU使用GPU，没有使用CPU
        self.epochs = epochs
        self.net = shufflenet_v2_x1_0(num_classes=2) # 两种类别，有虫或没虫
        self.model_weight_path = './shufflenetv2_x1.pth'  # load pretrain weights  download url: https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth
        self.save_path = "./shufflenetv2_x1_weight.pth" # 保存模型权重的未知
        self.freeze = False  # 是否冻结权重
        self.lr = lr  # 优化器
        self.lrf = 0.01
        self.train_steps = len(train_dl)  # 迭代步长

        print("using {} device.".format(self.device))
        # print("train_num: " + str(train_num) + " val_num： " + str(val_num))
        self.net.to(self.device)
        # 判断此权重是否存在
        assert os.path.exists(self.model_weight_path), "file {} dose not exist.".format(self.model_weight_path)


def main():
    datas = get_datas()

    myconfig = ShuffleNet_v2_config(datas.train_dl, datas.val_dl, datas.train_num, datas.val_num)

    # 放置到设备
    myconfig.net.to(myconfig.device)

    # 如果存在预训练权重则载入
    if myconfig.model_weight_path != "":
        if os.path.exists(myconfig.model_weight_path):
            weights_dict = torch.load(myconfig.model_weight_path, map_location=myconfig.device)
            load_weights_dict = {k: v for k, v in weights_dict.items()
                                 if myconfig.net.state_dict()[k].numel() == v.numel()}
            print(myconfig.net.load_state_dict(load_weights_dict, strict=False))
        else:
            raise FileNotFoundError("not found weights file: {}".format(myconfig.model_weight_path))

    # 是否冻结权重
    if myconfig.freeze:
        for name, para in myconfig.net.named_parameters():
            # 除最后的全连接层外，其他权重全部冻结
            if "fc" not in name:
                para.requires_grad_(False)

    # 把未冻结的加进去
    pg = [p for p in myconfig.net.parameters() if p.requires_grad]
    # 优化器
    optimizer = optim.SGD(pg, lr=myconfig.lr, momentum=0.9, weight_decay=4E-5)

    # 用户自定义学习率调整器，lr_lambda可以是lambda表达式，也可以是函数
    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    # lf = lambda x: ((1 + math.cos(x * math.pi / myconfig.epochs)) / 2) * (1 - myconfig.lrf) + myconfig.lrf  # cosine
    # scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    best_acc = 0.0

    for epoch in range(myconfig.epochs):
        # train
        mean_loss = train_one_epoch(model=myconfig.net,
                                    optimizer=optimizer,
                                    data_loader=myconfig.train_dl,
                                    device=myconfig.device,
                                    epoch=epoch)

        # scheduler.step()

        # validate
        acc = evaluate(model=myconfig.net,
                       data_loader=myconfig.val_dl,
                       device=myconfig.device)

        print("[epoch {}] accuracy: {}".format(epoch, round(acc, 3)))

        if acc > best_acc:
            path = os.path.splitext(myconfig.save_path)[0]
            if os.path.exists(path + '_' + str(round(best_acc, 3)) + '.pth'):
                os.remove(path + '_' + str(round(best_acc, 3)) + '.pth')
            best_acc = acc
            torch.save(myconfig.net.state_dict(), path + '_' + str(round(best_acc, 3)) + '.pth')


def train_one_epoch(model, optimizer, data_loader, device, epoch):

    model.train()

    # 损失函数
    loss_function = torch.nn.CrossEntropyLoss()
    mean_loss = torch.zeros(1).to(device)
    optimizer.zero_grad()

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):

        images, labels = data

        pred = model(images.to(device))

        loss = loss_function(pred, labels.to(device))

        loss.backward()

        mean_loss = (mean_loss * step + loss.detach()) / (step + 1)  # update mean losses

        data_loader.desc = "train: [epoch {}] mean loss {}".format(epoch+1, round(mean_loss.item(), 3))

        # 判断输入张量每个元素是否为有限值
        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()

        optimizer.zero_grad()

    return mean_loss.item()


def evaluate(model, data_loader, device):
    model.eval()

    # 验证样本总个数
    total_num = len(data_loader.dataset)

    # 用于存储预测正确的样本个数
    sum_num = torch.zeros(1).to(device)

    data_loader = tqdm(data_loader, file=sys.stdout)

    for step, data in enumerate(data_loader):
        images, labels = data
        pred = model(images.to(device))
        pred = torch.max(pred, dim=1)[1]
        sum_num += torch.eq(pred, labels.to(device)).sum()

    return sum_num.item() / total_num


if __name__ == '__main__':
    main()
