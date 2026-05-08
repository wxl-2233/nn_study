"""
函数:
    torch.arange() 和 torch.linspace() 创建线性张量 指定范围，等差数列
    torch.random.initial_seed() 和 torch.random.manual_seed() 随机种子设置 基于时间戳变化，固定
    torch.rand/randn() 创建随机浮点类型张量 n表示正态分布
    torch.randint(low, high, size=()) 创建随机整数类型张量
"""

import torch

# 创建线性张量
def dm01():
    # 场景1: 创建指定范围的 线性张量
    # 参1: 起始值, 参2: 结束值, 参3: 步长
    t1 = torch.arange(0, 10, 2) # 0, 2, 4, 6, 8
    print(f't1: {t1}, type: {type(t1)}')
    print('-' * 30)

    # 场景2: 创建指定范围的 线性张量 -> 等差数列
    # 参1: 起始值, 参2: 结束值, 参3: 元素的个数
    t2 = torch.linspace(1, 10, 4) # 1., 4., 7., 10.
    print(f't2: {t2}, type: {type(t2)}')

# 创建随机张量
def dm02():
    # step1: 设置随机种子
    # torch.initial_seed()    # 默认采用当前系统的时间戳作为随机种子
    torch.manual_seed(3)      # 固定的随机种子
    # 随机种子固定，则运行多次step2创建的张量一致，且管理dm02随机种子以下所有内容
    # 可在 main 里统一控制，以管理所有的张量创建

    # step2: 创建随机张量
    # 场景1: 均匀分布的(0, 1) 随机张量
    t1 = torch.rand(size=(2, 3))
    print(f't1: {t1}, type: {type(t1)}')
    print('-' * 30)

    # 场景2: 符合正态分布的随机张量
    t2 = torch.randn(size=(2, 3))
    print(f't2: {t2}, type: {type(t2)}')
    print('-' * 30)

    # 场景3: 创建随机整数张量
    t3 = torch.randint(low=1, high=10, size=(3, 5))
    print(f't3: {t3}, type: {type(t3)}')

if __name__ == '__main__':
    dm01()
    dm02()