"""
API:
    sum(), max(), min(), mean()                     -> 都有 dim 参数, 0表示列, 1表示行
    pow(), sqrt(), exp(), log(), log2(), log10()    -> 没有dim参数
"""

import torch
t1 = torch.tensor([
    [1, 2, 3],
    [4, 5, 6]
], dtype=torch.float)

# sum() 求和
print(t1.sum(dim=0))        # 按 列 求和
print(t1.sum(dim=1))        # 按 行 求和
print(t1.sum())             #  整体 求和
print('-' * 30)

# max()求最大值, min()同理
print(t1.max(dim=0))        # 按 列 求最大值
print(t1.max(dim=1))        # 按 行 求最大值
print(t1.max())             #  整体 求最大值
print('-' * 30)

# mean(), 计算平均值 必须是浮点型
print(t1.mean(dim=0))        # 按 列 求平均值
print(t1.mean(dim=1))        # 按 行 求平均值
print(t1.mean())             #  整体 求平均值
print('*' * 30)

# 没有dim参数的 函数
# pow() n次幂
print(t1.pow(2))            # 每个数的平方
print(t1 ** 2)              # 效果同上
print('-' * 30)

# sqrt() 平方根
print(t1.sqrt())            # 每个数的平方根
print('-' * 30)

# exp() e的n次幂, n就是矩阵中的每个元素, 这里是: e^1, e^2, e^3, e^4, e^5, e^6
print(t1.exp())
print('-' * 30)

# log(), log2(), log10()  对数
print(t1.log())
print(t1.log2())
print(t1.log10())
