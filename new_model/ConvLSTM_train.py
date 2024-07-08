import os

import torch
from torch import nn

from ConvLSTM_model import ConvLSTM_audio

from GetDatas import get_datas

import torch.optim as optim
from tqdm import tqdm
import sys

# ResNet18_attention_weight_0.940.pth 是加入了CBAM跑出来的，按照原论文的方式，先通道卷积再空间卷积


class ConvLSTM_config:
    def __init__(self, train_dl, val_dl, train_num, val_num, epochs=20, lr=0.001, is_conv1d=False):

        self.net = None
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.train_num = train_num
        self.val_num = val_num
        self.lr = lr
        self.epochs = epochs
        self.num_classes = 2
        if is_conv1d:
            self.in_channels = 13 # mfcc或stft等变幻出来的在频率方面的维度（stft是频率，mfcc等是滤波器）
            self.kernel_size = 3
        else:
            self.in_channels = 1
            self.kernel_size = (3, 3)
        self.out_channels = 32
        self.batch_size = 16  # 根据之前的弄
        self.height = 14  # mfcc或stft等变幻出来的在频率方面的维度（stft是频率，mfcc等是滤波器），且上取整至偶数
        self.width = 16  # 分出来的时间段，在lstm2D中，首先将音频分成了20短，每段是8000，8000/512上取整，就是这样子
        self.is_conv1d = is_conv1d
        self.num_layers = 3  # 这里设置的3层，可以增大
        self.cell_type = 'LSTM'
        self.bidirectional = False
        self.cat_type = 'last'
        self.train_steps = len(train_dl)
        self.model_weight_path = None
        self.save_path = './ConvLSTM_weight.pth'
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.net = ConvLSTM_audio(self.num_classes, self.width, self.in_channels, self.out_channels, self.kernel_size, self.num_layers, self.batch_size, height=self.height, is_conv1d=self.is_conv1d)
        self.net.to(self.device)
        print("using {} device.".format(self.device))


def main():
    datas = get_datas(audio_type="convLSTM")

    myconfig = ConvLSTM_config(datas.train_dl, datas.val_dl, datas.train_num, datas.val_num)

    if myconfig.model_weight_path is not None:
        # 加载权重
        pre_weights = torch.load(myconfig.model_weight_path, map_location='cpu')

        # delete classifier weights
        pre_dict = {k: v for k, v in pre_weights.items() if myconfig.net.state_dict()[k].numel() == v.numel()}
        missing_keys, unexpected_keys = myconfig.net.load_state_dict(pre_dict, strict=False)

    # define loss function 损失函数
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer 优化器
    params = [p for p in myconfig.net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=myconfig.lr)
    # scheduler = StepLR(optimizer, step_size=6, gamma=0.1)

    # 准确率
    best_acc = 0.0

    # 迭代步长 到底干啥的？？？
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
            # scheduler.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     myconfig.epochs,
                                                                     loss)

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
            if os.path.exists(path+'_'+str(round(best_acc, 3))+'.pth'):
                os.remove(path+'_'+str(round(best_acc, 3))+'.pth')
            best_acc = val_accurate
            torch.save(myconfig.net.state_dict(), path+'_'+str(round(best_acc, 3))+'.pth')

    print('Finished Training')


if __name__ == '__main__':
    main()