"""
API(函数):
    场景1: 张量 -> numpy nd数组对象
        张量对象.numpy()            共享内存
        张量对象.numpy().copy()     不共享内存, 链式编程 != 函数嵌套调用numpy(len(t1))
    场景2: numpy nd数组 -> 张量
        from_numpy()               共享内存
        torch.tensor(nd数组)       不共享内存
    场景3: 从标量张量中 提取其内容.
        标量张量.item()
"""
import torch
import numpy as np

# 张量 -> numpy
def dm01():
    t1 = torch.tensor([1, 2, 3, 4, 5])

    n1 = t1.numpy()           # 共享内存
    # n1 = t1.numpy().copy()      # 不共享内存

    n1[0] = 100
    print(f'n1: {n1}')  # [100, 2, 3, 4, 5]
    print(f't1: {t1}')  # [100/1, 2, 3, 4, 5]

# numpy -> 张量
def dm02():
    n1 = np.array([11, 22, 33])

    # t1 = torch.from_numpy(n1).type(torch.float32)   # 转换+转类型
    t1 = torch.from_numpy(n1)               # 共享内存
    t2 = torch.tensor(n1)                   # 不共享内存

    n1[0] = 100
    print(f'n1: {n1}')  # 100, 22, 33
    print(f't1: {t1}')  # 100, 22, 33
    print(f't2: {t2}')  # 11, 22, 33

# 从标量张量(只有1个值的张量)中提取其内容.
def dm03():
    t1 = torch.tensor(100)              # 可以
    # t1 = torch.tensor([100, ])        # 可以
    # t1 = torch.tensor([100, 200])     # 不可以.
    print(f't1: {t1}, type: {type(t1)}')

    a = t1.item()
    print(f'value: {a}, type: {type(a)}')

if __name__ == '__main__':
    dm01()
    dm02()
    dm03()