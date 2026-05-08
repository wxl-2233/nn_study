"""
损失函数介绍:
    概述:
        损失函数也叫成本函数, 目标函数, 代价函数, 误差函数, 就是用来衡量 模型好坏(模型拟合情况)的.
    分类:
        分类问题:
            多分类交叉熵损失: CrossEntropyLoss
            二分类交叉熵损失: BCELoss
        回归问题:
            MAE: Mean Absolute Error, 平均绝对误差.
            MSE: Mean Squared Error, 均方误差.
            Smooth L1: 结合上述两个的特点做的升级, 优化.

多分类交叉熵损失: CrossEntropyLoss
    多分类交叉熵常用于：一个样本只能属于多个类别中的一个类别。
    比如有 3 个类别： 类别0：猫  类别1：狗  类别2：鸟    样本真实类别是“狗”，真实标签：[0, 1, 0]   one-hot 标签
    交叉熵的核心思想是：模型给真实类别的预测概率越高，损失越小；模型给真实类别的预测概率越低，损失越大。

    设计思路:
        Loss = - Σylog(S(f(x)))
    简单记忆:
        x:          样本
        f(x):       加权求和
        S(f(x)):    处理后的概率(softmax函数)
        y:          样本x属于某一个类别的 真实概率.
    解释:
        损失函数结果 = 最小化 正确类别所对应的 预测概率的对数的 负值(损失值最小)...
    细节:
        CrossEntropyLoss = Softmax() + 损失计算, 后续如果用这个损失函数, 则: 输出层就不用额外调用 softmax()激活函数了.

二分类任务的损失函数(BCELoss):
    公式:
        Loss = -ylog(预测值) - (1 - y)log(1 - 预测值)
    细节:
        因为公式中没有包含Sigmoid激活函数, 所以使用BCELoss的时候, 还需要手动指定 Sigmoid.

回归任务常用损失函数:
    MAE:   Mean Absolute Error, 平均绝对误差.
        公式:
            误差绝对值之和 / 样本总数
        类似于L1正则化, 权重可以降维0, 数据会变得稀疏.
        弊端:
            在0点不平滑, 可能错过最小值.（因为在0点不可导，可能导致优化算法无法找到最优解）

    MSE:   Mean Squared Error, 均方误差.
        公式:
            误差平方之和 / 样本总数
        弊端:
            如果差值过大, 可能存在梯度爆炸的情况. W新 = W旧 - α * ∇L (因为误差被平方了，较大的误差会导致损失值急剧增加，从而导致梯度爆炸)

    Smooth L1:
        就是基于MAE 和 MSE做的综合, 在 [-1, 1]是 L2(MSE), 其它段时L1.
        这样即解决了L1不平滑的问题(0点不可导, 可能错过最小值)
        又解决了L2(MSE)的 梯度爆炸的问题.        
"""


import torch
import torch.nn as nn

# 多分类任务的损失函数 CrossEntropyLoss
def dm01():
    # 1. 手动创建样本的真实值 -> y
    y_true = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.float) # 创建两个样本的真实标签，soft label
    # y_true = torch.tensor([1, 0]) # 也可直接创建类别索引,表示第一个样本的第1列是正确类别, 第二个样本的第0列是正确类别 

    # 2. 手动创建样本的预测值 -> f(x)
    y_pred = torch.tensor([[0.1, 0.8, 0.1], [0.7, 0.2, 0.1]], requires_grad=True, dtype=torch.float)
    # CrossEntropyLoss 要求输入的是 logits 即 原始分数，这里的[0.1, 0.8, 0.1] 是原始得分，需进行 softmax 处理（CrossEntropyLoss 内部会自动处理）
    # logits 原始分数可以是任意实数：可以是负数、0、小数、大数，不要求在 0 到 1 之间，也不要求加起来等于 1
    # requires_grad=True 表示这个张量需要参与梯度计算 执行：loss.backward()时 PyTorch 会自动计算损失函数对 y_pred 的梯度 但这里没有执行反向传播

    # 3. 创建多分类交叉熵损失函数
    criterion = nn.CrossEntropyLoss()       # 平均损失, 参数: reduction: str = "mean"
    
    # 4. 计算损失值
    loss = criterion(y_pred, y_true)
    print(f'损失值: {loss}') # 值越小，说明预测越接近真实标签

# 二分类任务的损失函数 BCELoss
def dm02():
    y_true = torch.tensor([0, 1, 0], dtype=torch.float)
    y_pred = torch.tensor([0.6901, 0.5423, 0.2639]) # 经过 Sigmoid 激活函数处理后的概率值，表示每个样本属于正类的概率，[0, 1]
    # BCEWithLogitsLoss = Sigmoid + 二分类交叉熵损失函数, 直接输入 logits 原始分数，内部会自动进行 Sigmoid 激活处理
    criterion = nn.BCELoss()    # reduction: str = "mean" -> 均值
    loss = criterion(y_pred, y_true)
    print(f'损失值: {loss}')

# MAE 损失函数
def dm11():
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)
    criterion = nn.L1Loss()
    loss = criterion(y_pred, y_true)
    print(f'MAE: {loss}')


# MSE 损失函数
def dm12():
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)
    criterion = nn.MSELoss()
    loss = criterion(y_pred, y_true)
    print(f'MSE: {loss}')


# Smooth L1 损失函数
def dm13():
    y_true = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float)
    y_pred = torch.tensor([1.0, 1.0, 1.9], requires_grad=True)
    criterion = nn.SmoothL1Loss()
    loss = criterion(y_pred, y_true)
    print(f'Smooth L1: {loss}')


if __name__ == '__main__':
    dm01()
    dm02()
    dm11()
    dm12()
    dm13()