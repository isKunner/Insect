import os
import sys

import torch
from sklearn.metrics import r2_score, classification_report
from tqdm import tqdm

from GetDatas import get_datas
from ResNet18_model import resnet18


class config():

    def __init__(self, train_dl, val_dl, train_num, val_num, epochs=20):
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.train_num = train_num
        self.val_num = val_num
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 有GPU使用GPU，没有使用CPU
        self.epochs = epochs
        self.net = resnet18(num_classes=2, visual_field=True, attention=True, audio_type="mf-lf-ng") # 两种类别，有虫或没虫
        self.model_weight_path = './Resnet18_visual_attention_weight_0.97.pth'
        print("using {} device.".format(self.device))
        self.net.to(self.device)
        # 判断此权重是否存在
        assert os.path.exists(self.model_weight_path), "file {} dose not exist.".format(self.model_weight_path)


def inference():
    datas = get_datas(audio_type="mf-lf-ng", sample="None", batch_size=16, n_fft=1024, win_length=1024, hop_length=1024)

    myconfig = config(datas.train_dl, datas.val_dl, datas.train_num, datas.val_num)

    # 加载权重
    pre_weights = torch.load(myconfig.model_weight_path, map_location='cpu')

    # delete classifier weights
    pre_dict = {k: v for k, v in pre_weights.items() if myconfig.net.state_dict()[k].numel() == v.numel()}
    missing_keys, unexpected_keys = myconfig.net.load_state_dict(pre_dict, strict=False)

    myconfig.net.eval()

    acc = 0.0  # accumulate accurate number
    predic = []
    y = []

    with torch.no_grad():
        train_bar = tqdm(myconfig.train_dl, file=sys.stdout)
        for train_data in train_bar:
            train_images, train_labels = train_data
            outputs = myconfig.net(train_images.to(myconfig.device))
            predict_y = torch.max(outputs, dim=1)[1]
            y.extend(train_labels)
            predic.extend(predict_y)
            acc += torch.eq(predict_y, train_labels.to(myconfig.device)).sum().item()
    y = torch.stack(y)
    predic = torch.stack(predic)
    R2 = r2_score(y_true=y.cpu().numpy(), y_pred=predic.cpu().numpy())
    report = (classification_report(y_true=y.cpu().numpy(), y_pred=predic.cpu().numpy()))
    print(report)
    train_accurate = acc / myconfig.train_num
    print('train_accuracy: %.3f R2: %.3f' % (train_accurate, R2))

    acc = 0.0
    predic = []
    y = []

    with torch.no_grad():
        val_bar = tqdm(myconfig.val_dl, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = myconfig.net(val_images.to(myconfig.device))
            predict_y = torch.max(outputs, dim=1)[1]
            y.extend(val_labels)
            predic.extend(predict_y)
            acc += torch.eq(predict_y, val_labels.to(myconfig.device)).sum().item()
    y = torch.stack(y)
    predic = torch.stack(predic)
    R2 = r2_score(y_true=y.cpu().numpy(), y_pred=predic.cpu().numpy())
    report = (classification_report(y_true=y.cpu().numpy(), y_pred=predic.cpu().numpy()))
    print(report)
    val_accurate = acc / myconfig.val_num
    print('val_accuracy: %.3f   R2: %.3f' % (val_accurate, R2))

    # Run inference on trained model with the validation set


if __name__ == '__main__':
    inference()