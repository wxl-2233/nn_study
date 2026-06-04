"""
优化思路：
    1. 优化方法 SGD -> Adam
    2. 学习率 lr=0.001 -> lr=0.0001
    3. 对数据进行标准化
    4. 增加网络的深度，每层的神经元数目
    5. 增加训练的轮数
    6. ......
"""

import torch                                            
from torch.utils.data import TensorDataset              
from torch.utils.data import DataLoader                
import torch.nn as nn                              
import torch.optim as optim                          
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler                      
import numpy as np                             
import pandas as pd                              
import time          

def create_dataset():
    data = pd.read_csv('./model/手机价格预测.csv')
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    x = x.astype(np.float32)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=88, stratify=y)
    # 优化3：数据标准化
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test.values))
    return train_dataset, test_dataset, x_train.shape[1], len(np.unique(y))


class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 优化4：增加网络的深度，每层的神经元数目
        self.linear1 = nn.Linear(input_dim, 128) 
        self.linear2 = nn.Linear(128, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 128)
        self.output = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.relu(self.linear3(x))
        x = torch.relu(self.linear4(x))
        x = self.output(x)
        return x 


def train(train_dataset, input_dim, output_dim):
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    model = PhonePriceModel(input_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    # 优化1：优化方法 SGD -> Adam
    # 优化2：学习率 lr=0.001 -> lr=0.0001
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    # 可优化5：增加训练的轮数
    epochs = 50
    for epoch in range(epochs):
        total_loss, batch_num = 0.0, 0
        start = time.time()
        for x, y in train_loader:
            model.train() 
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_num += 1
        print(f'epoch: {epoch + 1}, loss: {total_loss / batch_num:.4f}, time: {time.time() - start:.2f}s')

    torch.save(model.state_dict(), './model/phone.pth') 


def evaluate(test_dataset, input_dim, output_dim):
    model = PhonePriceModel(input_dim, output_dim)
    model.load_state_dict(torch.load('./model/phone.pth'))
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    correct = 0
    for x, y in test_loader:
        model.eval()
        y_pred = model(x)
        y_pred = torch.argmax(y_pred, dim=1)
        correct += (y_pred == y).sum()

    print(f'准确率(Accuracy): {correct / len(test_dataset):.4f}') # 0.9200


if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()
    train(train_dataset, input_dim, output_dim)
    evaluate(test_dataset, input_dim, output_dim)
