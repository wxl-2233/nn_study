"""
函数:
    torch.ones 和 torch.ones_like 创建全1张量 _like基于给定形状创建
    torch.zeros 和 torch.zeros_like 创建全0张量
    torch.full 和 torch.full_like 创建全为指定值张量
"""
import torch

# 场景1: torch.ones 和 torch.ones_like 创建全1张量
t1 = torch.ones(2, 3)   # 创建2行3列全1张量
print(f't1: {t1}, type: {type(t1)}')
print('-' * 30)

# t2: 3行2列
t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])

# t3 -> 基于t2的形状, 创建全1张量.
t3 = torch.ones_like(t2)
print(f't3: {t3}, type: {type(t3)}')    # 3行2列 -> 全1矩阵
print('*' * 30)


# 场景2: torch.zeros 和 torch.zeros_like 创建全0张量
t1 = torch.zeros(2, 3)
print(f't1: {t1}, type: {type(t1)}')
print('-' * 30)

t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
t3 = torch.zeros_like(t2)
print(f't3: {t3}, type: {type(t3)}')
print('*' * 30)


# 场景3: torch.full 和 torch.full_like 创建全为指定值张量
t1 = torch.full(size=(2, 3), fill_value=255)
print(f't1: {t1}, type: {type(t1)}')
print('-' * 30)

t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
t3 = torch.full_like(t2, 255)
print(f't3: {t3}, type: {type(t3)}')
