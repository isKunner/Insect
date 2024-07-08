import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from MobileNet_v2_model import MobileNetV2
from GetDatas import get_datas

# MobileNet_v2 输入的图像要求是图像格式，即三通道（大小无限制）
# 为适应模型，预处理得到的是MFCC图，即1*n*m的，所以给模型加了一个1*1*1*3的卷积核以升维conv0


class MobileNetV2_config():
    def __init__(self, train_dl, val_dl, train_num, val_num, epochs=20, lr=0.001, visual_field=True, attention=True, audio_type='mfcc'):

        self.train_dl = train_dl
        self.val_dl = val_dl
        self.train_num = train_num
        self.val_num = val_num
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 有GPU使用GPU，没有使用CPU
        self.epochs = epochs # 迭代轮次
        self.net = MobileNetV2(num_classes=2, visual_field=visual_field, attention=attention, audio_type=audio_type) # 两种类别，有虫或没虫，创建了模型
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


def main():

    datas = get_datas()

    myconfig = MobileNetV2_config(datas.train_dl, datas.val_dl, datas.train_num, datas.val_num)

    # 加载权重
    pre_weights = torch.load(myconfig.model_weight_path, map_location='cpu')

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if myconfig.net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = myconfig.net.load_state_dict(pre_dict, strict=False)

    # # freeze features weights net.parameters会全部冻结
    if myconfig.freeze:
        for param in myconfig.net.features.parameters():
            param.requires_grad = False

    # define loss function 损失函数
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer 优化器
    params = [p for p in myconfig.net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=myconfig.lr)

    # 准确率
    best_acc = 0.0

    for epoch in range(myconfig.epochs):

        # train
        myconfig.net.train()
        # 每次把损失清零
        running_loss = 0.0
        # 进度条，输出到显示器
        train_bar = tqdm(myconfig.train_dl, file=sys.stdout)
        for step, data in enumerate(train_bar):
            # 图像 和 标签
            images, labels = data
            # 梯度清零
            optimizer.zero_grad()
            # 输出分类结果
            logits = myconfig.net(images.to(myconfig.device))
            # 计算损失
            loss = loss_function(logits, labels.to(myconfig.device))
            # 计算发现的值
            loss.backward()
            # 反向传播
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, myconfig.epochs, loss)

        # validate
        # 评估阶段，不会随意dropout等等
        myconfig.net.eval()
        acc = 0.0  # accumulate accurate number / epoch

        # 反向传播不自动求导，节省空间内存
        with torch.no_grad():
            val_bar = tqdm(myconfig.val_dl, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = myconfig.net(val_images.to(myconfig.device))
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(myconfig.device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, myconfig.epochs)
        val_accurate = acc / myconfig.val_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / myconfig.train_steps, val_accurate))

        if val_accurate > best_acc:
            path = os.path.splitext(myconfig.save_path)[0]
            if os.path.exists(path + '_' + str(round(best_acc, 3)) + '.pth'):
                os.remove(path + '_' + str(round(best_acc, 3)) + '.pth')
            best_acc = val_accurate
            torch.save(myconfig.net.state_dict(), path + '_' + str(round(best_acc, 3)) + '.pth')

    print('Finished Training')


if __name__ == '__main__':
    main()
