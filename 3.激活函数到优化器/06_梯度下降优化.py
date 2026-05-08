"""
梯度下降:
    概述:
        梯度下降 是结合 本次损失函数的导数(作为梯度) 基于学习率 来更新权重的.
    公式:
        W新 = W旧 - 学习率 * (本次的)梯度
    存在的问题:
        1. 遇到平缓区域, 梯度下降(权重更新)可能会慢
        2. 可能会遇到 鞍点(梯度为0)
        3. 可能会遇到 局部最小值
    解决思路:
        从上述的 学习率 或者 梯度入手, 进行优化, 于是有了: 动量法Momentum, 自适应学习率AdaGrad, RMSProp, 综合衡量: Adam

    动量法Momentum:
        动量法公式:
            St = β * St-1 + (1 - β) * Gt
        解释:
            St:     本次的指数移动加权平均结果.
            β:      调节权重系数, 越大, 数据越平缓, 历史指数移动加权平均 比重越大, 本次梯度权重越小.
            St-1:   历史的指数移动加权平均结果.
            Gt:     本次计算出的梯度(不考虑历史梯度).
        加入动量法后的 梯度更新公式:
            W新 = W旧 - 学习率 * St

    自适应学习率: AdaGrad(Adaptive Gradient Estimation)
        公式:
            累计平方梯度:
                St = St-1 + Gt * Gt
                解释:
                    St:     累计平方梯度.
                    St-1:   历史累计平方梯度.
                    Gt:     本次的梯度.
            学习率:
                学习率 = 学习率 / (sqrt(St) + 小常数)
                解释:
                    小常数: 1e-10, 目的: 防止分母变为0
            梯度下降公式:
                W新 = W旧 - 调整后的学习率 * Gt
        缺点:
            可能会导致学习率过早, 过量的降低, 导致模型后期学习率太小, 较难找到最优解.


    自适应学习率: RMSProp(Root Mean Square Propagation) -> 可以看做是 对AdaGrad做的优化, 加入 调和权重系数.
        公式:
            指数加权平均 累计历史平方梯度:
                St = β * St-1 +  (1 - β) * Gt * Gt
                β: 调和权重系数.
        优点:
           RMSProp通过引入 衰减系数β, 控制历史梯度 对 历史梯度信息获取的多少.

    自适应矩估计: Adam(Adaptive Moment Estimation)
        思路:
            既优化学习率, 又优化梯度. Adam = RMSProp + Momentum
        公式:
            一阶矩: 算均值.
                Mt = β1 * Mt-1 + (1 - β1) * Gt          充当: 梯度
                St = β2 * St-1 + (1 - β2) * Gt * Gt     充当: 学习率
            二阶矩: 梯度的方差.
                Mt^ = Mt / (1 - β1 ^ t)
                St^ = St / (1 - β2 ^ t)
            权重更新公式:
                W新 = W旧 - 学习率 / (sqrt(St^) + 小常数)  *  Mt^

总结: 
    简单任务和较小的模型:
        SGD, 动量法
    复杂任务或者有大量数据:
        Adam
    需要处理稀疏数据或者文本数据:
        AdaGrad, RMSProp
"""

import torch
import torch.optim as optim

"""
    动量法：代码里用的和讲的有区别

    写法一：指数移动平均形式
    S_t = β * S_{t-1} + (1 - β) * grad_t
    
    写法二: PyTorch SGD Momentum 形式
    v_t = β * v_{t-1} + grad_t

    (31) EMA Exponential Moving Average(指数移动平均/指数加权移动平均)
    公式: EMA_t = β * EMA_{t-1} + (1 - β) * x_t
"""
# 1. 动量法(Momentum)
def dm01_momentum():
    # 1. 初始化权重参数 定义初始参数 w=1.0，允许 PyTorch 自动计算它的梯度
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    # 2. 定义损失函数（这里实际是损失值loss） 其导数d(loss)/dw = w  w=1.0时，grad=1.0
    criterion = ((w ** 2) / 2.0)
    # 3. 创建优化器(函数对象) -> 基于SGD(随机梯度下降), 加入参数 momentum, 就是 动量法.
    # 参1: (待优化的)参数列表, 参2: 学习率, 参3: 动量参数.
    optimizer = optim.SGD(params=[w], lr=0.01, momentum=0.9)  # 细节: momentum=0(默认), 只考虑: 本次梯度.
    # 4. 计算梯度值: 梯度清零 + 反向传播 + 参数更新  清空旧梯度 → 反向传播计算新梯度 → 优化器更新参数
    optimizer.zero_grad() # PyTorch 的梯度默认是累加的，第一次反向传播得到 w.grad=1，如果不清零，第二次再算出 0.99，它可能变成 1.99
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')
    # 第1次：
    # 初始 w = 1.0
    # loss = 0.5 * w^2 = 0.5 * 1^2 = 0.5
    # grad = d(loss)/dw = w = 1.0
    # v1 = 0.9 * 0 + grad = 1.0
    # w新 = w旧 - 学习率 * v1 = 1.0 - 0.01 * 1.0 = 0.99

    # 第2次：
    # 当前 w = 0.99
    # loss = 0.5 * w^2 = 0.5 * 0.99^2 = 0.49005
    # grad = d(loss)/dw = w = 0.99
    # v2 = 0.9 * v1 + grad = 0.9 * 1.0 + 0.99 = 1.89
    # w新 = w旧 - 学习率 * v2 = 0.99 - 0.01 * 1.89 = 0.9711
    # 5. 第2次 更新权重参数.
    criterion = ((w ** 2) / 2.0)
    # 计算梯度值: 梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

# 2. 自适应学习率(AdaGrad)
def dm02_adagrad():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    criterion = ((w ** 2) / 2.0)
    # 思路2: 基于AdaGrad(自适应学习率).
    optimizer = optim.Adagrad(params=[w], lr=0.01)

    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')
    # 第1次：
    # 初始 w = 1.0
    # loss = 0.5 * w^2 = 0.5 * 1^2 = 0.5
    # grad = d(loss)/dw = w = 1.0
    # AdaGrad 累计平方梯度：s1 = 0 + grad^2 = 0 + 1.0^2 = 1.0
    # 更新后的学习率：update1 = lr / sqrt(s1) = 0.01 / sqrt(1.0) = 0.01
    # w新 = 1.0 - update1 * grad = 1.0 - 0.01 * 1.0 = 0.99

    # 第2次：
    # 当前 w = 0.99
    # loss = 0.5 * w^2 = 0.5 * 0.99^2 = 0.49005
    # grad = d(loss)/dw = w = 0.99
    # AdaGrad 继续累计平方梯度：s2 = s1 + grad^2 = 1.0 + 0.99^2 = 1.0 + 0.9801 = 1.9801
    # 更新后的学习率：update2 = lr / sqrt(s2) = 0.01 / sqrt(1.9801) ≈ 0.0071067
    # w新 = 0.99 - update2 * grad = 0.99 - 0.0071067 * 0.99 ≈ 0.98296
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

# 3. 自适应学习率(RMSProp)
def dm03_rmsprop():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    criterion = ((w ** 2) / 2.0)
    # 思路3: 基于RMSProp(自适应学习率).
    optimizer = optim.RMSprop(params=[w], lr=0.01, alpha=0.99) # alpha 就是 β, 默认就是0.99
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')
    # 第1次：
    # 初始 w = 1.0
    # loss = 0.5 * w^2 = 0.5 * 1^2 = 0.5
    # grad = d(loss)/dw = w = 1.0
    # RMSProp 累计平方梯度：s1 = 0.99 * 0 + 0.01 * grad^2 = 0 + 0.01 * 1.0^2 = 0.01
    # 更新后的学习率：update1 = lr / sqrt(s1) = 0.01 / sqrt(0.01) = 0.01 / 0.1 = 0.1
    # w新 = 1.0 - update1 * grad = 1.0 - 0.1 * 1.0 = 0.9

    # 第2次：
    # 当前 w = 0.9
    # loss = 0.5 * w^2 = 0.5 * 0.9^2 = 0.405
    # grad = d(loss)/dw = w = 0.9
    # RMSProp 继续累计平方梯度：s2 = 0.99 * s1 + 0.01 * grad^2 = 0.99 * 0.01 + 0.01 * 0.9^2 = 0.0099 + 0.0081 = 0.018
    # 更新后的学习率：update2 = lr / sqrt(s2) = 0.01 / sqrt(0.018) ≈ 0.01 / 0.1342 ≈ 0.0745356
    # w新 = 0.9 - update2 * grad = 0.9 - 0.0745356 * 0.9 ≈ 0.832918
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

# 4. 自适应矩估计(Adam)
def dm04_adam():
    w = torch.tensor([1.0], requires_grad=True, dtype=torch.float32)
    criterion = ((w ** 2) / 2.0)
    # 思路4: 基于Adam(自适应矩估计).
    optimizer = optim.Adam(params=[w], lr=0.01, betas=(0.9, 0.999)) # betas=(梯度用的 衰减系数, 学习率用的 衰减系数)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')
    # 第1次：
    # 初始 w = 1.0
    # loss = 0.5 * w^2 = 0.5 * 1^2 = 0.5
    # grad = d(loss)/dw = w = 1.0
    # Adam 一阶矩: Mt = 0.9 * 0 + 0.1 * grad = 0 + 0.1 * 1.0 = 0.1
    # Adam 二阶矩: St = 0.999 * 0 + 0.001 * grad^2 = 0 + 0.001 * 1.0^2 = 0.001
    # Adam 一阶矩修正: Mt^ = Mt / (1 - 0.9 ^ 1) = 0.1 / (1 - 0.9) = 0.1 / 0.1 = 1.0
    # Adam 二阶矩修正: St^ = St / (1 - 0.999 ^ 1) = 0.001 / (1 - 0.999) = 0.001 / 0.001 = 1.0
    # 权重更新公式: w新 = w旧 - 学习率 / (sqrt(St^) + 小常数) * Mt^ = 1.0 - 0.01 / (sqrt(1.0) + 1e-10) * 1.0 = 0.99    

    # 第2次：
    # 当前 w = 0.99
    # loss = 0.5 * w^2 = 0.5 * 0.99^2 = 0.49005
    # grad = d(loss)/dw = w = 0.99
    # Adam 一阶矩: Mt = 0.9 * 0.1 + 0.1 * grad = 0.09 + 0.1 * 0.99 = 0.09 + 0.099 = 0.189
    # Adam 二阶矩: St = 0.999 * 0.001 + 0.001 * grad^2 = 0.000999 + 0.001 * 0.99^2 = 0.0019791
    # Adam 一阶矩修正: Mt^ = Mt / (1 - 0.9 ^ 2) = 0.189 / (1 - 0.81) ≈ 0.994737
    # Adam 二阶矩修正: St^ = St / (1 - 0.999 ^ 2) = 0.0019791 / (1 - 0.998001) ≈ 0.990045
    # 权重更新公式: w新 = w旧 - 学习率 / (sqrt(St^) + 小常数) * Mt^ ≈ 0.98
    criterion = ((w ** 2) / 2.0)
    optimizer.zero_grad()
    criterion.sum().backward()
    optimizer.step()
    print(f'w: {w}, w.grad: {w.grad}')

if __name__ == '__main__':
    dm01_momentum()
    dm02_adagrad()
    dm03_rmsprop()
    dm04_adam()