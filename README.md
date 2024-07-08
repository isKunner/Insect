# Insect
本科科创项目：柱干昆虫检测

在Resnet18和MobileNet上效果最好，可以达到1的准确率，在DenseNet和ShuffleNet上可以达到98%、99%的准确率

你可以在init.py文件中通过修改config类的参数来进行不同模型的训练

你需要在GetDatas.py文件中通过修改文件的路径来进行数据的读取（文件路径为音频所在的文件夹路径）

数据集所在路径：https://www.kaggle.com/datasets/potamitis/treevibes （注：由于数据集的训练集和测试集分类不均衡，所以代码中全部读取并重新切分进行训练，同时文件夹属性CONFIRMATION==NO，其中的数据不能保证标签一定正确）
