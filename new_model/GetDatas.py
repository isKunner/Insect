import os

import pandas as pd
import torch
from torch.utils.data import random_split
from SoundDS import SoundDS
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class get_datas:
    def __init__(self, timestamp=None, target_names=None, pre_path=None, mid_path=None, test_size=None, audio_type='stft', sample=None, n_xxcc=20, batch_size=16, n_fft=2048, win_length=2048, hop_length=512):

        self.batch_size = batch_size
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        # timestamp = int(time.time())
        if timestamp is None:
            self.timestamp = 1441455
            # self.timestamp = 12
        else:
            self.timestamp = timestamp

        if target_names is None:
            self.target_names = ['clean', 'infested']  # 两个文件夹的类别名称
        else:
            self.target_names = target_names

        if pre_path is None:
            self.pre_path = os.path.abspath(os.path.join(os.getcwd(), ".."))  # 获取当前的绝对路径（运行的主程序所在的路径）
        else:
            self.pre_path = pre_path

        # 1.获取数据集中的训练集合

        if mid_path is None:
            self.mid_path = '/TreeData/field/field/train/'  # 从当前绝对的路径到存储数据所经过的路径
            self.mid_path = '/TreeData/new_data/'
        else:
            self.mid_path = mid_path

        if test_size is None:
            self.test_size = 0.2
        else:
            self.test_size = test_size

        self.audio_type = audio_type

        self.X_names = []  # 保存音频的路径
        self.y = []  # 保存音频对应的标签

        # 1.保存每个音频的文件路径并标记上对应的标签
        x = 0
        for i, target in enumerate(self.target_names):
            # i为0时对应的target为clean；i为1时对应的target为infested
            path = self.pre_path + self.mid_path + target + '/'  # train from scratch using field data
            # root 所指的是当前正在遍历的这个文件夹的本身的地址
            # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
            # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
            # 函数会自动改变root的值使得遍历所有的子文件夹。所以返回的三元元组的个数为所有子文件夹（包括子子文件夹，子子子文件夹等等）加上1（根文件夹）
            for [root, dirs, files] in os.walk(path):
                if "3" in root or "4" in root or "18" in root or "22" in root or "23" in root:
                    continue
                for filename in files:
                    # 分离文件名与扩展名
                    name, ext = os.path.splitext(filename)
                    if ext == '.wav':
                        name = os.path.join(root, filename)
                        self.X_names.append(name)
                        self.y.append(i)  # 标签
                        x = x+1

        # 2.获取数据集合中的测试集合

        # self.mid_path = '/TreeData/field/field/test'
        # self.mid_path = '/TreeData/new_data'
        # # i为0时对应的target为clean；i为1时对应的target为infested
        # path = self.pre_path + self.mid_path + '/'  # train from scratch using field data
        # print(path)
        # # root 所指的是当前正在遍历的这个文件夹的本身的地址
        # # dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
        # # files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
        # # 函数会自动改变root的值使得遍历所有的子文件夹。所以返回的三元元组的个数为所有子文件夹（包括子子文件夹，子子子文件夹等等）加上1（根文件夹）
        # for [root, dirs, files] in os.walk(path):
        #     for filename in files:
        #         # 分离文件名与扩展名
        #         name, ext = os.path.splitext(filename)
        #         if ext == '.wav':
        #             name = os.path.join(root, filename)
        #             self.X_names.append(name)
        #             self.y.append(1)  # 标签
        #             x = x+1

        "y==1:2400  y==0:1754"
        # print("长度")
        # print(len(self.y))
        # print(self.y)
        # print(np.sum(np.array(self.y)==1))

        # 上采样
        if sample == 'up':
            self.X_names, self.y = up_sampling(self.X_names, self.y)

        # 下采样，效果似乎不是太好
        elif sample == 'down':
            self.X_names, self.y = sub_sampling(self.X_names, self.y, n=731, seed=1331)

        # 2.打乱顺序
        self.X_names, self.y = shuffle(self.X_names, self.y, random_state=self.timestamp)

        # 3.切分训练集测试集
        # 此处进行了8，2分成
        train_x, test_x, train_y, test_y = train_test_split(self.X_names, self.y, test_size=self.test_size, random_state=self.timestamp)

        # 转换类型
        df_train = pd.DataFrame({'path': train_x, 'ID': train_y})

        # if sample == 'up':
        #
        #     df_train = up_samplings(df_train).reset_index()
        #
        # elif sample == 'down':
        #
        #     df_train = down_samplings(df_train, seed=self.timestamp).reset_index()

        df_test = pd.DataFrame({'path': test_x, 'ID': test_y})

        # 4.集成DataSet，进行重载，生成数据集迭代器
        # 主要是对数据进行batch的划分，除此之外，特别要注意的是输入进函数的数据一定得是可迭代的
        # 如果是自定的数据集的话可以在定义类中用def__len__、def__getitem__定义
        # 这个迭代器猜测可能是使用了yield，可以减少内存消耗，同时读取数据时使用小表读取，会自动调用TrainDatas中的__getitem__函数并返回值，最终返回batch_size个这种值

        TrainDatas = SoundDS(df_train, '', audio_type=self.audio_type, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, n_xxcc=n_xxcc)
        self.train_dl = torch.utils.data.DataLoader(TrainDatas, batch_size=self.batch_size, shuffle=False, drop_last=False)

        TestDatas = SoundDS(df_test, '', audio_type=self.audio_type, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length, n_xxcc=n_xxcc)

        self.val_dl = torch.utils.data.DataLoader(TestDatas, batch_size=self.batch_size, shuffle=False, drop_last=False)

        # 这个是数据集的长度
        self.train_num = len(df_train)
        self.val_num = len(df_test)

        # print(df_test.info())

        print(f'train_num.length: {self.train_num}')
        print(f'val_num.length: {self.val_num}')


# 上面是上采样、下采样的函数，唔，目前用不太着

def sub_sampling(x, y, n, seed=1111):
    x = pd.DataFrame({'path': x, 'y': y})

    # 分层抽样字典定义 组名：数据个数
    typicalNDict = {1: n, 0: n}

    # 函数定义
    def typicalsamling(group, typicalNDict):
        name = group.name
        n = typicalNDict[name]
        return group.sample(n=n, random_state=seed)

    #    返回值：抽样后的数据框
    result = x.groupby(y).apply(typicalsamling, typicalNDict)

    return result['path'].tolist(), result['y'].tolist()


def up_sampling(x, y):

    X = pd.DataFrame({'path': x, 'y': y})

    X_sample = X[X['y'] == 1]

    result = pd.concat([X, X_sample])

    return result['path'].tolist(), result['y'].tolist()


def up_samplings(x: pd.DataFrame):

    id1 = x['ID'].value_counts().index[0]
    val1 = x['ID'].value_counts().iloc[0]

    id2 = x['ID'].value_counts().index[1]
    val2 = x['ID'].value_counts().iloc[1]

    if val1 > 2*val2:

        temp = x[x['ID'] == id2]

        while val1 > val2:

            if val1-val2 > val2:
                x = pd.concat([x, temp])
            else:
                x = pd.concat([x, temp.iloc[:val1-val2]])

            val1 -= 2*val2

    elif val2 > val1:

        temp = x[x['ID'] == id1]

        while val2 > val1:

            if val2-val1 > val1:
                x = pd.concat([x, temp])
            else:
                x = pd.concat([x, temp.iloc[:val2-val1]])

            val2 -= val1

    return x


def down_samplings(x: pd.DataFrame, seed=113):

    # id1 = x['ID'].value_counts().index[0]
    val1 = x['ID'].value_counts().iloc[0]

    # id2 = x['ID'].value_counts().index[1]
    val2 = x['ID'].value_counts().iloc[1]

    if val1 > val2:
        n = val2
    else:
        n = val1

    #  返回值：抽样后的数据框
    result = x.groupby(x['ID']).sample(n=n)

    return result