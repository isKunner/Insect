from torch import nn

vgg_config = {'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
              'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
              'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
              'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                    'M']}


def make_layers(config):
    layers = []
    in_channels = config.in_channels
    cfg = vgg_config[config.vgg_type]
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers.append(conv2d)
            in_channels = v
    return nn.Sequential(*layers)


class VGGNet(nn.Module):
    # features便是上面make_layers函数所返回的结果
    def __init__(self, features, config):
        super(VGGNet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(147456, 4096),
            nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True),
            nn.Dropout(), nn.Linear(4096, config.num_classes)
        )
        if config.init_weights:
            self.initialize_weights()

    def forward(self, x, labels=None):
        # print(x.shape)
        x = self.features(x)
        # print(x.shape)
        logits = self.classifier(x)
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # 用于检查一个对象是否是某个类的实例
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # _的作用是就地操作，和inplace类似
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)  # 偏置置为0
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant(m.bias, 0)


def vgg():
    config = Config()
    cnn_features = make_layers(config)
    model = VGGNet(cnn_features, config)
    return model


class Config(object):
    def __init__(self):
        self.vgg_type = 'B'
        self.num_classes = 2
        self.init_weights = True
        self.in_channels = 1
