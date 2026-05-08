"""
点乘:
    要求: 两个张量的维度保持一致, 对应元素直接做 乘法操作
    API:
        t1 * t2
        t1.mul(t2)          # multiply: 乘法

矩阵乘法:
    要求: 两个张量, 第一个张量 的列数, 等于 第二个张量 的行数(A列 = B行)
    结果: A行B列
    API:
        t1 @ t2
        t1.matmul(t2)
        t1.dot(t2)          扩展: 只针对于一维张量有效
"""

import torch

# 点乘
def dm01():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    t2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    # t3 = t1 * t2
    t3 = t1.mul(t2) # 效果同上
    print(f't3: {t3}')

# 矩阵乘法 条件: A列=B行, 结果: A行B列
def dm02():
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    t2 = torch.tensor([[1, 2], [3, 4], [5, 6]])
    t3 = t1 @ t2
    # t3 = t1.matmul(t2)  # 效果同上, matrix multiply(矩阵乘法)
    print(f't3: {t3}')

    # t3 = t1.dot(t2)       # 报错, dot() 只针对于一维张量有效.
    t4 = torch.tensor([1, 2, 3])
    t5 = torch.tensor([4, 5, 6])
    t6 = t4.dot(t5)
    print(f't6: {t6}')

if __name__ == '__main__':
    dm01()
    dm02()