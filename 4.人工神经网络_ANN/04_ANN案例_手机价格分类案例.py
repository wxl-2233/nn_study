"""
案例:
    ANN(人工神经网络)案例: 手机价格分类案例.

背景:
    基于手机的20列特征 -> 预测手机的价格区间(4个区间), 可以用机器学习做, 也可以用 深度学习做(推荐)

ANN案例的实现步骤:
    1. 构建数据集.
    2. 搭建神经网络.
    3. 模型训练.
    4. 模型测试.
"""

import torch                                            # PyTorch框架, 封装了张量的各种操作
from torch.utils.data import TensorDataset              # 数据集对象.   数据 -> Tensor -> 数据集 -> 数据加载器
from torch.utils.data import DataLoader                 # 数据加载器.
import torch.nn as nn                                   # neural network, 封装了神经网络的各种操作
import torch.optim as optim                             # 优化器
from sklearn.model_selection import train_test_split    # 训练集和测试集的划分
import matplotlib.pyplot as plt                         # 绘图
import numpy as np                                      # 数组(矩阵)操作
import pandas as pd                                     # 数据处理
import time                                             # 时间模块
from torchsummary import summary                        # 模型参数计算    

# 1. 构建数据集.
def create_dataset():
    # 1. 加载csv文件数据集.
    data = pd.read_csv('./model/手机价格预测.csv')
    # print(f'data: {data.head()}')   # 输出前5行数据
    # print(f'data: {data.shape}')    # (2000, 21)

    # 2. 获取 x特征列 和 y标签列.
    x, y = data.iloc[:, :-1], data.iloc[:, -1] # [:, :-1]表示取所有行，除了最后一列以外的所有列作为特征；[:, -1]表示取所有行的最后一列作为标签。
    # print(f'x: {x.head()}, {x.shape}')  # (2000, 20)
    # print(f'y: {y.head()}, {y.shape}')  # (2000, 1) 1是默认值不显示

    # 3. 把特征列转成浮点型.
    x = x.astype(np.float32)
    # print(f'x: {x.head()}, {x.shape}')   # (2000, 20)

    # 4. 切分训练集和测试集.
    # 参1: 特征, 参2: 标签, 参3: 测试集所占比例, 参4: 随机种子, 参5: 样本的分布(即: 参考y的类别来抽取数据)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3, stratify=y)

    # 5. 把数据集封装成 张量数据集.  思路: 数据 -> 张量Tensor -> 数据集TensorDataSet -> 数据加载器DataLoader
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    # print(f'train_dataset: {train_dataset}, test_dataset: {test_dataset}') # TensorDataset对象包含了特征和标签的张量数据，可以通过索引访问每个样本的特征和标签。

    # 6. 返回结果                         20(输入特征数)      4(输出标签数)
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))


# 2. 搭建神经网络.
class PhonePriceModel(nn.Module):
    # 1. 在init魔法方法中, 初始化父类成员, 及搭建神经网络.
    def __init__(self, input_dim, output_dim):  # 输入: 20, 输出: 4
        # 1.1 初始化父类成员.
        super().__init__()
        # 1.2 搭建神经网络.
        # 隐藏层1
        self.linear1 = nn.Linear(input_dim, 256) # 修改了256和128的顺序，因为一般隐藏层的神经元数量逐渐减少来帮助模型更好地学习数据的特征
        # 隐藏层2
        self.linear2 = nn.Linear(256, 128)
        # 输出层
        self.output = nn.Linear(128, output_dim)


    # 2. 前向传播 forward()
    def forward(self, x):
        # 2.1 隐藏层1: 加权求和 + 激活函数(relu)
        # x = self.linear1(x)
        # x = torch.relu(x)
        x = torch.relu(self.linear1(x))
        # 2.2 隐藏层2: 加权求和 + 激活函数(relu)
        x = torch.relu(self.linear2(x))
        # 2.3 输出层: 加权求和 + 激活函数(softmax)  -> 这里只需要做 加权求和.
        # 正常写法, 但是不需要, 后续用 多分类交叉熵损失函数 CrossEntropyLoss() 替代
          # CrossEntropyLoss() = softmax() + 损失计算
        # x = torch.softmax(self.output(x), dim=1)
        x = self.output(x)
        return x


# 3. 模型训练.
def train(train_dataset, input_dim, output_dim):
    # 1. 创建数据加载器, 流程: 数据 -> 张量 -> 数据集 -> 数据加载器
    # 参1: 数据集对象(1600条), 参2: 每批次的数据条数, 参3: 是否打乱数据(训练集: 打乱, 测试集: 不打乱)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # 2. 创建神经网络模型.
    model = PhonePriceModel(input_dim, output_dim)
    # 3. 定义损失函数, 因为是多分类, 这里用的是: 多分类交叉熵损失函数.
    criterion = nn.CrossEntropyLoss() # 交叉熵损失函数, 适用于多分类问题  softmax + 计算损失.
    # 4. 创建优化器对象.
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # 5. 模型训练.
    # 5.1 定义变量, 记录训练的 总轮数.
    epochs = 50
    # 5.2 开始(每轮的)训练.
    for epoch in range(epochs):
        # 5.2.1 定义变量, 记录每次训练的损失值, 训练批次数.
        total_loss, batch_num = 0.0, 0
        # 5.2.2 定义变量, 表示训练开始的时间.
        start = time.time()
        # 5.2.3 开始本轮的 各个批次的训练.
        for x, y in train_loader:
            # 5.2.3.1 切换模型(状态)
            model.train()   # 训练模式.    model.eval()   # 测试模式.
            # 5.2.3.2 模型预测.
            y_pred = model(x)
            # 5.2.3.3 计算损失.
            loss = criterion(y_pred, y)
            # 5.2.3.4 梯度清零, 反向传播, 优化参数.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 5.2.3.5 累加损失值.
            total_loss += loss.item()   # 把本轮的每批次(16条)的 平均损失累计起来. 第1批次的平均损失 + 第2批次的平均损失 + ...
            batch_num += 1
        # 5.2.4 至此, 本轮训练结束, 打印训练信息.
        print(f'epoch: {epoch + 1}, loss: {total_loss / batch_num:.4f}, time: {time.time() - start:.2f}s')

    # 6. 走到这里, 说明多轮训练结束, 保存模型(参数)
    # 参1: 模型对象的参数(权重矩阵, 偏置矩阵)  参2: 模型保存的文件名.
    # print(f'\n\n模型的参数信息: {model.state_dict()}\n\n')
    torch.save(model.state_dict(), './model/phone.pth') # 后缀名用: pth, pkl, pickle均可.


# 4. 模型测试.
def evaluate(test_dataset, input_dim, output_dim):
    # 1. 创建神经网络分类对象.
    model = PhonePriceModel(input_dim, output_dim)
    # 2. 加载模型参数.
    model.load_state_dict(torch.load('./model/phone.pth'))
    # 3. 创建测试集的 数据加载器对象.
    # 参1: 数据集对象(400条), 参2: 每批次的数据条数, 参3: 是否打乱数据(训练集: 打乱, 测试集: 不打乱)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    # 4. 定义变量, 记录预测正确的样本个数.
    correct = 0
    # 5. 从数据加载器中, 获取到每批次的数据.
    for x, y in test_loader:
        # 5.1 切换模型状态 -> 测试模式.
        model.eval()
        # 5.2 模型预测.
        y_pred = model(x)             # softmax是在损失函数里面使用的，用处是把这些得分转换为概率（如[10%, 5%, 80%, 5%]）
                                      # 但是在训练阶段交叉熵损失函数自动处理了softmax，所以我们没有显式调用softmax函数
                                      # 模型输出的y_pred是每个类别的得分（logits）而不是概率
                                      # 我们需要使用argmax函数来找到得分最高的类别索引作为预测结果。
        # print(f'y_pred: {y_pred}')  # [[0分类概率, 1分类概率, 2分类概率, 3分类概率], [...]...]
        
        # 5.3 根据加权求和, 得到类别, 用argmax()获取最大值对应的下标, 就是类别.
        y_pred = torch.argmax(y_pred, dim=1)    # dim=1 表示逐行处理.
        # print(f'y_pred: {y_pred}')  # [第1条数据的预测分类, ...]
        # print(f'y: {y}')

        # 5.4 统计预测正确的样本个数.
        # print(y_pred == y)          # tensor([ True,  True,  True,  True, False, False,  True, False])
        # print((y_pred == y).sum())  # True:1, False:0
        correct += (y_pred == y).sum()

    # 6.走到这里, 模型预测结束, 打印准确率即可.
    print(f'准确率(Accuracy): {correct / len(test_dataset):.4f}') # 0.6450


if __name__ == '__main__':
    # 1. 准备数据集.
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    # print(f'训练集 数据集对象: {train_dataset}')
    # print(f'测试集 数据集对象: {test_dataset}')
    # print(f'输入特征数: {input_dim}')    # 20
    # print(f'输出标签数: {output_dim}')   # 4

    # 2. 构建神经网络模型.
    # model = PhonePriceModel(input_dim, output_dim)
    # 计算模型参数
    # 参1: 模型对象. 参2: 输入数据的形状(批次大小, 输入特征数), 每批16条, 每条20列特征
    # summary(model, input_size=(16, input_dim))
    # Linear-1  [-1, 16, 256]    5,376     256 * 20(输入特征数) + 256(偏置) = 5,376
    # Linear-2  [-1, 16, 128]   32,896     128 * 256(隐藏层1的神经元数) + 128(偏置) = 32,896
    # Linear-3  [-1, 16, 4]        516     4 * 128(隐藏层2的神经元数) + 4(偏置) = 516

    # 3. 模型训练
    train(train_dataset, input_dim, output_dim)

    # 4. 模型测试.
    evaluate(test_dataset, input_dim, output_dim)
