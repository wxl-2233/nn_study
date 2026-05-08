# 导入相关模块
import torch
# numpy对象 -> 张量Tensor -> 数据集对象TensorDataset -> 数据加载器Dataloader
from torch.utils.data import TensorDataset  # 构造数据集对象
from torch.utils.data import DataLoader  # 数据加载器(可以分批次获取数据)
from torch import nn  # nn模块中有 平方损失函数 和 假设函数
from torch import optim  # optim模块中有 优化器函数
from sklearn.datasets import make_regression  # 创建线性回归模型数据集
import matplotlib.pyplot as plt  # 可视化

plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 创建线性回归样本数据
def create_dataset():
    # 创建数据集对象
    # y = w * x + b  x是生成的100个点，w=coef，b=14.5
    x, y, coef = make_regression(
        n_samples=100,  # 100条样本(100个样本点)
        n_features=1,  # 1个特征(1个特征点)
        noise=10,  # 噪声, 噪声越大, 样本点越散, 噪声越小, 样本点越集中
        coef=True,  # 是否返回系数, 默认为False, 返回值为None
        bias=14.5,  # 偏置
        random_state=3  # 随机种子, 随机种子相同, 输出数据相同
    )

    # 把数据, 封装成 张量对象
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    return x, y, coef

# 模型训练
def train(x, y, coef):
    # 创建数据集对象 tensor -> 数据集对象 -> 数据加载器.
    dataset = TensorDataset(x, y)
    # 创建数据加载器对象 参1: 数据集对象, 参2: 批次大小, 参3: 是否打乱数据(训练集打乱, 测试集不打乱)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # 定义初始的 线性回归模型，自动包含权重和偏置 参1: 输入特征维度, 参2: 输出特征维度
    model = nn.Linear(1, 1)
    # 定义损失函数 均方误差：loss = (预测值 - 真实值)^2 的平均
    criterion = nn.MSELoss()
    # 定义优化器 w = w - 学习率 × 梯度 参1: 模型参数, 参2: 学习率
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 具体的训练过程
    # 训练轮数, 每轮的(平均)损失值, 训练总损失值, 训练的样本数
    epochs, loss_list, total_loss, total_sample = 100, [], 0.0, 0
    # 开始训练, 按轮训练
    for epoch in range(epochs):  # epoch的值: 0, 1, 2...99
        total_loss, total_sample = 0.0, 0 # 应该放在这里，每一轮的loss先清0
        # 每轮是分 批次 训练的, 所以从 数据加载器中 获取 批次数据
        for train_x, train_y in dataloader:  # 7批(16, 16, 16, 16, 16, 16, 4)
            y_pred = model(train_x) #前向传播进行预测 y_pred = w*x + b
            # 计算(每批的平均)损失值
            loss = criterion(y_pred, train_y.reshape(-1, 1))  # -1（自动计算，能转多少行就转多少行） 1列
            # 计算总损失 和 样本(批次)数
            total_loss += loss.item()
            total_sample += 1
            optimizer.zero_grad()  # 梯度清零
            loss.backward()        # 反向传播, 计算梯度
            optimizer.step()       # 梯度更新，更新w, b

        # 把本轮的(平均)损失值, 添加到列表中
        loss_list.append(total_loss / total_sample)
        print(f'轮数: {epoch + 1}, 平均损失值: {total_loss / total_sample}')

    print(f'{epochs} 轮的平均损失分别为: {loss_list}')
    print(f'模型参数, 权重: {model.weight}, 偏置: {model.bias}')

    #               100轮 每轮的平均损失值
    plt.plot(range(epochs), loss_list)
    plt.title('损失值曲线变化图')
    plt.grid()      # 绘制网格线
    plt.show()

    # 绘制预测值和真实值的关系
    # 绘制样本点分布情况
    plt.scatter(x, y)
    # 绘制训练模型的预测值
    # x: 100个样本点的特征
    y_pred = torch.tensor(data = [v * model.weight + model.bias for v in x]) # y_pred = model(x)
    # 计算真实值
    y_true = torch.tensor(data = [v * coef + 14.5 for v in x])
    # 绘制预测值 和 真实值的 折线图
    plt.plot(x, y_pred, color='red', label='预测值')
    plt.plot(x, y_true, color='green', label='真实值')
    # 图例, 网格.
    plt.legend()
    plt.grid()
    # 显示图像
    plt.show()

if __name__ == '__main__':
    # 创建数据集
    x, y, coef = create_dataset()
    # print(f'x: {x}, y: {y}, coef: {coef}')
    
    # 模型训练
    train(x, y, coef)