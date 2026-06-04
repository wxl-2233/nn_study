# 神经网络与深度学习

---

## 1. 数值表示与张量运算

---

### `01_创建基础张量.py`

| 层次 | 需掌握内容 |
|------|------------|
| **概念** | 张量是「同类型数值」的多维容器；ANN/CNN/RNN 底层都以张量承载数据与参数 |
| **`torch.tensor`** | 从**已有数据**创建；**`dtype=`** 指定类型 |
| **`torch.Tensor`** | 从**形状**创建占位张量；与 `tensor` 语义不同 |
| **类型构造** | **`IntTensor`/`FloatTensor`/`DoubleTensor`** 等及自动类型转换 |
| **易错** | `tensor(2,3)` 无效；按形状占位用 **`Tensor(2, 3)`** |

### `02_创建线性和随机张量.py`

| 层次 | 需掌握内容 |
|------|------------|
| **概念** | 等差序列张量与可复现随机张量 |
| **线性** | **`torch.arange(start, end, step)`**；**`torch.linspace(start, end, steps)`**（按个数插值） |
| **随机种子** | **`torch.manual_seed`**；固定种子使 **`rand`/`randn`/`randint`** 可重复 |
| **随机张量** | **`torch.rand`**（\[0,1) 均匀）、**`torch.randn`**（标准正态）、**`torch.randint(low, high, size)`** |

### `03_创建全0_1_指定值张量.py`

| 层次 | 需掌握内容 |
|------|------------|
| **概念** | 用常数或「与某张量同形」快速占位 |
| **API** | **`ones`/`zeros`/`full`** 及 **`ones_like`/`zeros_like`/`full_like`** |
| **要点** | **`full`** 用 **`fill_value`**；**`*_like`** 继承参考张量的形状与设备（侧重形状） |

### `04_创建指定类型的张量.py`

| 层次 | 需掌握内容 |
|------|------------|
| **概念** | 元素 **dtype** 与创建后类型转换 |
| **创建时** | **`dtype=torch.float`** 等 |
| **转换** | **`.type(torch.int16)`**；**`.half()`/`.float()`/`.double()`/`.short()`/`.int()`/`.long()`** |

### `05_张量和numpy之间相互转换.py`

| 层次 | 需掌握内容 |
|------|------------|
| **概念** | 互转时注意是否**共享内存** |
| **Tensor→NumPy** | **`.numpy()`** 共享；**`.numpy().copy()`** 不共享 |
| **NumPy→Tensor** | **`torch.from_numpy`** 共享；**`torch.tensor(ndarray)`** 拷贝、不共享 |
| **标量** | 单元素张量 **`.item()`** 取 Python 数 |

### `06_张量的加减乘除.py`

| 层次 | 需掌握内容 |
|------|------------|
| **概念** | 逐元素算术；算子函数与 **`+ - * /`** 等价 |
| **函数** | **`add`/`sub`/`mul`/`div`/`neg`** |
| **in-place** | **`add_`** 等带 **`_`** 会改原张量（类似 pandas `inplace`） |
| **广播** | 张量与标量运算时对元素逐一对齐 |

### `07_张量的点乘和矩阵乘法.py`

| 层次 | 需掌握内容 |
|------|------------|
| **点乘** | 同形对应元素相乘：**`*`**、**`mul`** |
| **矩阵乘** | 内维匹配：**`@`**、**`matmul`**；结果形状「前行×后列」 |
| **`dot`** | **仅一维**向量点积；高维误用会报错 |

### `08_张量的求和最大值等.py`

| 层次 | 需掌握内容 |
|------|------------|
| **归约** | **`sum`/`mean`/`max`/`min`** 的 **`dim`**；不写 `dim` 为全局 |
| **`mean`** | 需浮点 dtype |
| **无 dim** | **`pow`/`sqrt`/`exp`/`log`/`log2`/`log10`** 等对逐元素作用 |

### `09_张量的索引操作.py`

| 层次 | 需掌握内容 |
|------|------------|
| **基础** | **`t[row, col]`**；**`:`** 表示整行/整列 |
| **列表索引** | 多坐标成对取元；**`[[r1],[r2]], [c1,c2]`** 取子矩阵块 |
| **切片** | 起止步长 **`start:end:step`** |
| **布尔索引** | 用条件掩码选行/列（如 **`t1[:, t1[1] > 5]`**） |
| **多维** | 三维 **`[0, :, :]`** 等按轴选取 |

### `10_张量的形状操作.py`

| 层次 | 需掌握内容 |
|------|------------|
| **`reshape`** | 元素总数不变改形状 |
| **升/降维** | **`unsqueeze(dim)`**、**`squeeze()`**（去掉长度为 1 的维） |
| **换维** | **`transpose`** 交换两维；**`permute`** 任意重排 |
| **`view`** | 要求内存**连续**；**`transpose` 后常为 False** |
| **`contiguous`** | **`is_contiguous()`**；不连续时先 **`contiguous()`** 再 **`view`** |

### `11_张量的拼接.py`

| 层次 | 需掌握内容 |
|------|------------|
| **`cat`** | 沿已有维拼接；除拼接维外其余维尺寸须一致；**`dim`** 指定沿哪一维堆长 |
| **`stack`** | 在新维上叠；参与张量**各维形状须全一致** |
| **易错** | `dim` 越界或形状不匹配会报错 |

**易混点**：逐元素乘 vs 矩阵乘；**`transpose` 后常需 `contiguous()` 再 `view`**

---

## 2. 自动微分与线性回归

---

### `01_自动微分模块入门案例.py`

| 层次 | 需掌握内容 |
|------|------------|
| **概念** | 损失对参数求导 → **梯度**；更新 **W新 = W旧 − lr × 梯度** |
| **`requires_grad`** | 需要求导的参数/中间量设为 True；计算宜用浮点 |
| **`backward`** | 标量 loss 才能 **`loss.backward()`**；非标量常用 **`loss.sum().backward()`** |
| **更新** | 用手动 **`w.data = w.data - lr * w.grad`**（与 `optimizer.step()` 思想一致） |

### `02_自动微分模块循环案例.py`

| 层次 | 需掌握内容 |
|------|------------|
| **循环训练** | 每步：**前向算 loss → `grad` 清零 → `backward` → 更新 `w.data`** |
| **梯度累加** | PyTorch 默认**累加**梯度，故每步需 **`w.grad.zero_()`**（首次注意 **`grad is None`**） |

### `03_自动微分_detach.py`

| 层次 | 需掌握内容 |
|------|------------|
| **问题** | **`requires_grad=True`** 的张量不能直接 **`.numpy()`** |
| **`detach()`** | 得到不追踪梯度的视图；**`requires_grad` 为 False**；仍可与原张量共享 storage |
| **用法** | **`t.detach().numpy()`** 安全导出 |

### `04_自动微分完整流程.py`

| 层次 | 需掌握内容 |
|------|------------|
| **流程** | 前向 **`z = x @ w + b`**（广播理解 bias）→ **`MSELoss(z, y)`** → **`loss.backward()`** |
| **梯度** | **`w.grad`**、**`b.grad`** 供后续 **`optimizer.step()`** 或手动更新使用 |

### `05_PyTorch框架_模拟线性回归.py`

| 层次 | 需掌握内容 |
|------|------------|
| **数据** | **`TensorDataset`** → **`DataLoader(batch_size, shuffle)`** |
| **模型** | **`nn.Linear(in, out)`** |
| **损失与优化** | **`nn.MSELoss()`**；**`optim.SGD(model.parameters(), lr)`** |
| **标准三步** | **`optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`** |
| **其它** | **`loss.item()`** 记录标量；标签形状与 **`reshape`** 对齐 |

---

## 3. 激活、初始化、模块化、损失与优化思想

---

### `01_激活函数_图解.py`

| 层次 | 需掌握内容 |
|------|------------|
| **目的** | 引入**非线性**；不同激活适用隐藏层/输出层（脚本小结：ReLU 系 vs Sigmoid/Tanh、Softmax 多分类） |
| **API** | **`torch.sigmoid`/`tanh`/`relu`/`softmax`**；**`dim`** 在多分类概率上的用法 |
| **导数图** | **`requires_grad=True`** 的 **`linspace`** → **`y.sum().backward()`** → **`x.grad`** 绘图 |

### `02_参数初始化.py`

| 层次 | 需掌握内容 |
|------|------------|
| **目的** | 缓解梯度消失/爆炸、加快收敛、**打破对称性**（全 0/全 1/常数一般不利） |
| **`nn.init`** | **`uniform_`/`normal_`/`constant_`/`zeros_`/`ones_`** 等**就地**末尾 **`_`** |
| **Xavier / Kaiming** | **`xavier_uniform_`/`xavier_normal_`**；**`kaiming_uniform_`/`kaiming_normal_`**；脚本结论：ReLU 系常配 Kaiming，非 ReLU 常配 Xavier |

### `03_搭建神经网络.py`

| 层次 | 需掌握内容 |
|------|------------|
| **步骤** | 类继承 **`nn.Module`**；**`__init__`** 里定义子层；**`forward`** 里写计算图 |
| **调用** | **`model(x)`** 等价 **`model.forward(x)`** |
| **初始化** | 可在 **`__init__`** 里对 **`self.linear.weight`** 等调用 **`nn.init.*`** |
| **观察** | **`torchsummary.summary(model, input_size=...)`**；**`named_parameters()`** 查看参数名与张量 |

### `04_损失函数.py`

| 层次 | 需掌握内容 |
|------|------------|
| **多分类** | **`CrossEntropyLoss`** ≈ LogSoftmax + NLL；输入 **logits**，通常**不再**在输出层手写 Softmax |
| **二分类** | **`BCELoss`** 要求概率在 (0,1)，需配合 Sigmoid；**`BCEWithLogitsLoss`**（脚本注释）可省 Sigmoid |
| **回归** | **`L1Loss`**（MAE）、**`MSELoss`**、**`SmoothL1Loss`** 及各自特点 |

### `05_指数移动加权平均.py`

| 层次 | 需掌握内容 |
|------|------------|
| **公式** | **`EMA_t = β·EMA_{t-1} + (1-β)·x_t`** |
| **β** | 越大曲线越平滑、越偏重历史；越小越贴近当前值 |
| **联系** | 与动量、Adam 中一阶/二阶矩的指数平均思想一致 |

### `06_梯度下降优化.py`

| 层次 | 需掌握内容 |
|------|------------|
| **基线** | **`optim.SGD`**；**`momentum`** 用速度平滑梯度 |
| **自适应** | **`Adagrad`**、**`RMSprop`**、**`Adam`**（公式与适用场景） |
| **用法** | **`optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()`**；注意梯度累加 |

---

## 4. 学习率、正则与 ANN 实战

---

### `00_学习率与梯度.py`

| 层次 | 需掌握内容 |
|------|------------|
| **现象** | 学习率过小收敛慢；过大越过最优点、震荡甚至**梯度爆炸** |
| **API** | **`backward`**；**`x.data.sub_(lr * x.grad)`**；**`x.grad.zero_()`** |
| **可视化** | 迭代 loss 轨迹与在抛物线上的落点 |


### `01_学习率衰减策略.py`

| 层次 | 需掌握内容 |
|------|------------|
| **`StepLR`** | 每 **`step_size`** 个 epoch，**`lr *= gamma`** |
| **`MultiStepLR`** | 在 **`milestones`** 的 epoch 上乘以 **`gamma`** |
| **`ExponentialLR`** | 每 epoch **`lr *= gamma`**（指数衰减） |
| **调用** | 每个 epoch 结束 **`scheduler.step()`**；可 **`get_last_lr()`** 记录曲线 |

### `02_dropout随机失活.py`

| 层次 | 需掌握内容 |
|------|------------|
| **目的** | 减轻过拟合；训练时随机屏蔽部分激活 |
| **`nn.Dropout(p)`** | 保留神经元输出按 **`1/(1-p)`** 缩放（脚本演示） |
| **注意** | 推理阶段应 **`model.eval()`** 关闭 Dropout（完整项目见 ANN 脚本） |

### `03_批量归一化.py`

| 层次 | 需掌握内容 |
|------|------------|
| **思想** | 标准化后再用可学习 **`γ, β`** 缩放平移，稳定中间层分布 |
| **`BatchNorm2d`** | 输入 **`(N,C,H,W)`**，**`num_features=C`** |
| **`BatchNorm1d`** | 常用于 **`(N, features)`** 或全连接后 **`(N, C, L)`**（脚本对线性输出做 1d BN） |

### `04_ANN案例_手机价格分类案例.py`

| 层次 | 需掌握内容 |
|------|------------|
| **数据** | **`pd.read_csv`**；**`train_test_split(..., stratify=y)`**；**`TensorDataset`/`DataLoader`** |
| **模型** | 多层 **`Linear` + ReLU**；多分类输出** logits**（不配 Softmax）+ **`CrossEntropyLoss`** |
| **训练** | **`model.train()`**、循环内三步；记录 loss/准确率；**`model.eval()`** 与 **`torch.no_grad()`** 做测试 |

### `05_手机价格分类_优化.py`

| 层次 | 需掌握内容 |
|------|------------|
| **特征** | **`StandardScaler`**：`fit_transform` 训练集、`transform` 测试集 |
| **结构/优化** | 更深、更宽；**`Adam`** + 更小 **`lr`**；**`model.train()`** 在循环内显式调用 |
| **持久化** | **`torch.save(model.state_dict(), ...)`**（及对应加载流程） |


---

## 5. 卷积神经网络（CNN）

---

**一句话（`5.卷积神经网络/`）**：从 **HWC/CHW** 与读图，到 **`Conv2d`** 特征图尺寸、**池化**降采样，再用 **CIFAR-10** 走通 **Conv→Pool→Flatten→全连接** 的分类训练

### `01_绘制图像.py`

| 层次 | 需掌握内容 |
|------|------------|
| **概念** | 灰度/RGB、**HWC** 布局 |
| **API** | **`plt.imshow`**（HWC）；**`plt.imread`/`imsave`** |
| **张量** | 可用 **`torch.full`** 等构造图像尺寸演示 |

脚本内：**`dm01()`**、**`dm02()`**

### `02_卷积层API介绍.py`

| 层次 | 需掌握内容 |
|------|------------|
| **CNN 组成** | Conv 提局部特征；Pooling 降维；Linear 输出 |
| **尺寸公式** | **`N = (W - F + 2P) / S + 1`**（脚本与注释一致） |
| **流程** | 图像 **HWC → tensor → permute 为 CHW → unsqueeze 得 NCHW**；**`nn.Conv2d(in_ch, out_ch, k, stride, padding)`** |
| **可视化** | 取某通道 **`permute` 回 HW** 再 **`imshow`** |

脚本内：**`dm01()`**

### `03_池化层API介绍.py`

| 层次 | 需掌握内容 |
|------|------------|
| **目的** | 降空间分辨率；**通道数不变** |
| **API** | **`nn.MaxPool2d`**、**`nn.AvgPool2d`**（kernel、stride、padding） |
| **形状** | 输入 **`(N,C,H,W)`**，脚本含单通道与多通道例 |

脚本内：**`dm01()`**、**`dm02()`**

### `04_CIFAR10数据集探索.py`

| 层次 | 需掌握内容 |
|------|------------|
| **数据** | **`torchvision.datasets.CIFAR10(root, train, transform, download)`** |
| **`ToTensor`** | PIL → 张量、数值范围 \[0,1]；**`CHW`** |
| **探索** | **`class_to_idx`**、**`data`/`targets` 形状**；**`imshow`** 注意显示 uint8 HWC |

脚本内：**`create_dataset()`**

### `05_CNN案例_图像分类.py`

| 层次 | 需掌握内容 |
|------|------------|
| **结构** | 交替 **Conv + ReLU + MaxPool**；**`reshape`/`view` 展平** 后接多层 **Linear**；输出 **10 类 logits** |
| **训练** | **`CrossEntropyLoss`** + **`optim.SGD`**（或脚本配置）；**`DataLoader`** |
| **工具** | **`summary`** 查参数量；**`model.train()`/`eval()`** 区分训练与测试 |

脚本内：**`create_dataset()`**、**`ImageModel`**、训练与测试流程

---

## 6. 循环神经网络（RNN）

**一句话（`6.循环神经网络/`）**：用 **Embedding** 把词变为向量，用 **`nn.RNN`** 理解 **(seq, batch, feat)** 与 **`output`/`h_n`**，再在歌词语料上搭 **语言模型式** 训练与采样生成

### `01_词嵌入层演示.py`

| 层次 | 需掌握内容 |
|------|------------|
| **作用** | 离散词（或索引）→ 稠密向量 |
| **`nn.Embedding(num_embeddings, embedding_dim)`** | 词表大小 × 向量维；输入索引张量查表 |
| **分词** | **`jieba.lcut`** 得到词序列再枚举索引 |

脚本内：**`dm01()`**

### `02_RNN层简介.py`

| 层次 | 需掌握内容 |
|------|------------|
| **输入形状** | **`nn.RNN(input_size, hidden_size, num_layers)`**；输入 **`x`** 常为 **`(seq_len, batch, input_size)`** |
| **初态** | **`h0`** 形状 **`(num_layers, batch, hidden_size)`** |
| **输出** | **`output`**：每个时间步最上层输出；**`h_n`**：最后时刻层叠隐藏状态 |

脚本为单文件顺序：`rnn(x, h0)` 打印 **`output.shape`**、**`h1.shape`**

### `03_RNN案例_AI歌词生成器.py`

| 层次 | 需掌握内容 |
|------|------------|
| **词表** | 分词、去重、**`word_to_index`**、语料转索引序列 **`corpus_idx`** |
| **数据集** | 继承 **`Dataset`**；**`__getitem__`** 用滑动窗口：**`x` 与 `y` 错一位**（预测下一词） |
| **模型** | **`Embedding` → `RNN` → `Linear(vocab)`**；**`CrossEntropyLoss`**；**`transpose(0,1)`** 适配 RNN 输入维序 |
| **生成** | 给定起始词与长度，用 **`argmax`/采样** 递推输出 |

脚本内：**`build_vocab()`**、**`LyricsDataset`**、**`TextGenerator`**、训练与预测函数

---

## 仓库说明与配套材料

- 理论讲义可与 **`md/`**、**`课堂笔记/`** 对照阅读
- 在仓库 **`代码`** 根目录运行脚本，保证 **`../data`**、**`./model`** 等相对路径正确
- 部分环境需安装：**`torchvision`**、**`torchsummary`**、**`jieba`**、**`pandas`/`sklearn`** 等

---
