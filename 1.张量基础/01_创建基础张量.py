"""
张量(Tensor):
    PyTorch框架属于最常用的深度学习框架
    ANN(人工神经网络), CNN(卷积神经网络), RNN(循环神经网络)
    底层在处理数据时, 都是使用 张量 来处理的.

    张量 -> 存储同一类型元素的容器, 且元素值必须是 数值才可以.

张量的基本创建方式:
    torch.tensor 根据指定数据（如type(t1))创建张量
    torch.Tensor 根据形状（如两行三列）创建张量 + tensor
    torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量

需要你掌握的方式:
    tensor(值, 类型), 例如:
        data = np.random.randint(0, 10, size=(2, 3))
        t3 = torch.tensor(data, dtype=torch.float)
"""
import torch
import numpy as np

# torch.tensor 根据指定数据创建张量 torch.Tensor
def dm01():
    # 场景1: 标量 张量
    t1 = torch.tensor(10)
    print(f't1: {t1}, type: {type(t1)}')
    print('-' * 30)

    # 场景2: 二维列表 -> 张量
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.tensor(data)
    print(f't2: {t2}, type: {type(t2)}')
    print('-' * 30)

    # 场景3: numpy nd数组 -> 张量
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.tensor(data, dtype=torch.float) # 指定浮点型
    # t3 = torch.Tensor(data, dtype=torch.float) # 不支持dtype，默认类型是float32
    print(f't3: {t3}, type: {type(t3)}')
    print('-' * 30)

    # 场景4: 尝试直接创建 指定维度(例如: 2行3列的)张量
    # t4 = torch.tensor(2, 3) # 报错，tensor() 是“用数据创建”，Tensor() 是“按形状创建”
    # print(f't4: {t4}, type: {type(t4)}')
    t4 = torch.Tensor(2, 3) # 创建一个 2×3 的张量，里面是随机的垃圾值
    print(f't4: {t4}, type: {type(t4)}')

# torch.IntTensor、torch.FloatTensor、torch.DoubleTensor 创建指定类型的张量
def dm02():
    # 场景1: 标量 张量
    # 场景2: 二维列表 -> 张量
    # 场景3: numpy nd数组 -> 张量
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.IntTensor(data)
    print(f't3: {t3}, type: {type(t3)}')
    print('-' * 30)

    # 场景4: 如果类型不匹配, 会尝试自动转换类型
    data = np.random.randint(0, 10, size=(2, 3))
    t4 = torch.FloatTensor(data) # 默认: float32，所以type: {type(t4)不打印
    print(f't4: {t4}, type: {type(t4)}')

if __name__ == '__main__':
    dm01()
    dm02()