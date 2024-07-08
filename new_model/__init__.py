import json
import os
import re
import sys
import time

import numpy as np
import torch
from datasets import tqdm
from torch import nn

from GetDatas import get_datas  # 用以获取数据，包括数据预处理部分
from MobileNet_v2_train import MobileNetV2_config  # MobileNetV2模型
from ShuffleNet_v2_train import ShuffleNet_v2_config  # ShuffleNetV2模型
from Vgg16_train import Vgg16_config  # Vgg16模型
from DenseNet_train import DenseNet_config  # DenseNet模型
from ConvLSTM_train import ConvLSTM_config  # ConvLSTM模型
from ResNet18_train import Resnet18_config  # Resnet18模型

from sklearn.metrics import r2_score
from sklearn.metrics import classification_report


class config:
    def __init__(self, model="ConvLstmNet", audio_type="mfcc", sample="None", epochs=30, batch_size=16, lr=0.001, n_xxcc=13, n_fft=2048, win_length=2018, hop_length=512, visual_field=True, attention=False):
        """
        @param model: 这里是模型的名称，目前有六种可选 ["MobileNetV2", "ShuffleNetV2", "Vgg16Net", "DenseNet", "ConvLstmNet", "ResNet18"]
        @param audio_type: 这是数据预处理的模式，可以啥也不处理，可以转换为mfcc图，或者stft图（主要是适用于不同的模型）
        @param sample: 这是数据预处理的方式，数据有点不平衡，2：1的样子，所以有三种：上采样、下采样、什么也不做 ["up", "down", "None"]
        @param epochs: 迭代的轮次
        @param batch_size: 数据集合的大小
        """

        self.get_datas = None  # 创建读取数据的类
        self.model = model
        self.model_config = None
        self.audio_type = audio_type
        self.sample = sample
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.visual_field = visual_field
        self.attention = attention
        self.n_xxcc = n_xxcc
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.result_path = './result.json'  # 保存文件的参数
        self.time = time.asctime()  # 获取当前时间
        self.result = {}  # 保存训练的参数

        if self.model == "MobileNetV2":
            self.init_result()  # 初始化
            self.get_datas = get_datas(audio_type=self.audio_type, sample=self.sample, batch_size=self.batch_size, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, n_xxcc=self.n_xxcc)
            self.model_config = MobileNetV2_config(self.get_datas.train_dl, self.get_datas.val_dl, self.get_datas.train_num, self.get_datas.val_num, visual_field=self.visual_field, attention=self.attention, epochs=self.epochs, lr=self.lr, audio_type=self.audio_type)
        elif self.model == "ShuffleNetV2":
            self.init_result()  # 初始化
            self.get_datas = get_datas(audio_type=self.audio_type, sample=self.sample, batch_size=self.batch_size, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, n_xxcc=self.n_xxcc)
            self.model_config = ShuffleNet_v2_config(self.get_datas.train_dl, self.get_datas.val_dl, self.get_datas.train_num, self.get_datas.val_num, epochs=self.epochs, lr=self.lr)
        elif self.model == "Vgg16Net":
            self.audio_type = 'stft'
            self.init_result()  # 初始化
            self.get_datas = get_datas(audio_type=self.audio_type, sample=self.sample, batch_size=self.batch_size, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, n_xxcc=self.n_xxcc)
            self.model_config = Vgg16_config(self.get_datas.train_dl, self.get_datas.val_dl, self.get_datas.train_num, self.get_datas.val_num, epochs=self.epochs, lr=self.lr)
        elif self.model == "DenseNet":
            self.audio_type = 'stft'
            self.init_result()  # 初始化
            self.get_datas = get_datas(audio_type=self.audio_type, sample=self.sample, batch_size=self.batch_size, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, n_xxcc=self.n_xxcc)
            self.model_config = DenseNet_config(self.get_datas.train_dl, self.get_datas.val_dl, self.get_datas.train_num, self.get_datas.val_num, epochs=self.epochs, lr=self.lr)
        elif self.model == "ConvLstmNet":
            is_conv1d = True
            if is_conv1d:
                self.audio_type = 'convLSTM_conv1d'
            else:
                self.audio_type = 'convLSTM_conv2d'
            self.init_result()  # 初始化
            self.get_datas = get_datas(audio_type=self.audio_type, sample=self.sample, batch_size=self.batch_size, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, n_xxcc=self.n_xxcc)
            self.model_config = ConvLSTM_config(self.get_datas.train_dl, self.get_datas.val_dl, self.get_datas.train_num, self.get_datas.val_num, epochs=self.epochs, lr=self.lr, is_conv1d=is_conv1d)
        elif self.model == "ResNet18":
            self.audio_type = "mfcc"
            self.init_result()  # 初始化
            seq_length = (160000-self.win_length)//self.hop_length+1 if self.audio_type != 'stft' else int(np.ceil(160000/self.hop_length))
            self.get_datas = get_datas(audio_type=self.audio_type, sample=self.sample, batch_size=self.batch_size, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, n_xxcc=self.n_xxcc)
            self.model_config = Resnet18_config(self.get_datas.train_dl, self.get_datas.val_dl, self.get_datas.train_num, self.get_datas.val_num, seq_length=seq_length, visual_field=self.visual_field, attention=self.attention, epochs=self.epochs, lr=self.lr, audio_type=self.audio_type)

        self.loss_function = None  # 损失函数
        self.optimizer = None  # 优化器
        self.scheduler = None  # 学习率调整器

    def init_result(self):
        self.result[self.model+"_"+str(self.time)] = {
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "lr": self.lr,
            "audio_type": self.audio_type,
            "sample": self.sample,
            "visual_vield": self.visual_field,
            "attention": self.attention,
            "n_fft": self.n_fft,
            "win_length": self.win_length,
            "hop_length": self.hop_length,
            "train": {}
        }

    def pre_load_weights(self):

        if self.model_config.model_weight_path is not None:
            if self.model == "DenseNet":
                load_state_dict(self.model_config.net, self.model_config.model_weight_path)
                if self.model_config.freeze:
                    for name, para in self.model_config.net.named_parameters():
                        # 除最后的全连接层外，其他权重全部冻结
                        if "fc" not in name:
                            para.requires_grad_(False)
            else:

                print(self.model_config.model_weight_path)

                pre_weights = torch.load(self.model_config.model_weight_path, map_location="cpu")

                pre_dict = {k: v for k, v in pre_weights.items() if self.model_config.net.state_dict()[k].numel() == v.numel()}

                self.model_config.net.load_state_dict(pre_dict, strict=False)

                # freeze features weights net.parameters会全部冻结
                if self.model_config.freeze:
                    for param in self.model_config.net.features.parameters():
                        param.requires_grad = False

    # 进行损失和优化器的初始化
    def loss_optim(self):
        self.loss_function = nn.CrossEntropyLoss()
        params = [p for p in self.model_config.net.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(params, lr=self.model_config.lr)

    # 学习率调整器（这个还是有点不大会，只能是多尝试了）
    def lr_scheduler(self):
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=3, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=3, min_lr=0, eps=0.000001)

    # 进行单次训练
    def train_once_epoch(self, epoch):
        # train
        self.model_config.net.train()
        # 每次把损失清零
        running_loss = 0.0
        # 进度条，输出到显示器
        train_bar = tqdm(self.model_config.train_dl, file=sys.stdout)
        for step, data in enumerate(train_bar):
            # 图像 和 标签
            images, labels = data
            # 梯度清零
            self.optimizer.zero_grad()
            # 输出分类结果
            logits = self.model_config.net(images.to(self.model_config.device))
            # 计算损失
            loss = self.loss_function(logits, labels.to(self.model_config.device))
            # 计算发现的值
            loss.backward()
            # 反向传播
            self.optimizer.step()
            # print statistics
            running_loss += loss.item()
            # 传入进度条的前缀
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, self.model_config.epochs, loss)
        return running_loss

    # 进行验证
    def validate(self, epoch, running_loss, best_acc):

        # validate
        # 评估阶段，不会随意dropout等等
        self.model_config.net.eval()

        acc = 0.0  # accumulate accurate number / epoch

        predic = []
        y = []

        # 反向传播不自动求导，节省空间内存
        with torch.no_grad():
            val_bar = tqdm(self.model_config.val_dl, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = self.model_config.net(val_images.to(self.model_config.device))
                predict_y = torch.max(outputs, dim=1)[1]
                y.extend(val_labels)
                predic.extend(predict_y)
                acc += torch.eq(predict_y, val_labels.to(self.model_config.device)).sum().item()
                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, self.model_config.epochs)

        # 计算R2_score系数
        y = torch.stack(y)
        predic = torch.stack(predic)
        R2 = r2_score(y_true=y.cpu().numpy(), y_pred=predic.cpu().numpy())
        report = (classification_report(y_true=y.cpu().numpy(), y_pred=predic.cpu().numpy()))
        print(report)

        val_accurate = acc / self.model_config.val_num

        # 输出相关信息
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f  R2: %.3f  lr: %.5f' %
              (epoch + 1, running_loss / self.model_config.train_steps, val_accurate, R2, self.lr))

        self.result[self.model+"_"+str(self.time)]["train"][str(epoch+1)] = {
            "train_loss": running_loss / self.model_config.train_steps,
            "val_accuracy": val_accurate,
            "classification_report": report,
            "R2_score": R2
        }

        return val_accurate

    def train(self):

        # 加载权重，Vgg16的权重有点问题，我目前没改
        if self.model != "Vgg16":
            self.pre_load_weights()

        # 初始化损失函数、优化器
        self.loss_optim()

        # 学习率调整器
        self.lr_scheduler()

        best_acc = 0.0  # 准确率

        model_best = None

        # 迭代循环
        for epoch in range(self.model_config.epochs):
            running_loss = self.train_once_epoch(epoch)
            val_accurate = self.validate(epoch, running_loss, best_acc)
            # val_accurate = self.validate(epoch, 1, best_acc)
            # 进行模型的保存
            if val_accurate > best_acc:
                best_acc = val_accurate
                self.result[self.model + "_" + str(self.time)]["val_acc"] = best_acc
                model_best = self.model_config.net.state_dict()
            self.scheduler.step(val_accurate)
            print(f"lr: {self.scheduler._last_lr}")

        path = os.path.splitext(self.model_config.save_path)[0]
        if os.path.exists(path + '_' + str(round(best_acc, 3)) + '.pth'):
            os.remove(path + '_' + str(round(best_acc, 3)) + '.pth')
        torch.save(model_best, path + '_' + str(round(best_acc, 3)) + '.pth')

        if os.path.exists(self.result_path):
            with open(self.result_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                # print(old_data)
                # print(self.result)
                old_data.update(self.result)
            with open(self.result_path, "w", encoding="utf-8") as f:
                json.dump(old_data, f, indent=4)
        else:
            with open(self.result_path, "w", encoding="utf-8") as f:
                json.dump(self.result, f, indent=4)

        print('Finished Training')


# DenseNet_model加载权重
def load_state_dict(model: nn.Module, weights_path: str) -> None:

    # '.'s are no longer allowed in module names, but previous _DenseLayer
    # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
    # They are also in the checkpoints in model_urls. This pattern is used
    # to find such keys.
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = torch.load(weights_path)

    num_classes = model.classifier.out_features
    load_fc = num_classes == 1000

    for key in list(state_dict.keys()):
        if load_fc is False:
            if "classifier" in key:
                del state_dict[key]

        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict, strict=load_fc)
    print("successfully load pretrain-weights.")


if __name__ == '__main__':
    myconfig = config()
    myconfig.train()



