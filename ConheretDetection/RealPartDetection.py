import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 生成数据：10000个样本，每个样本8个特征
random_data = np.random.randn(5000, 8)  # 噪声
signal_data = np.random.randn(5000, 8) + 1  # 带噪声信号
X = np.zeros((10000, 8))
X[:5000, :] = random_data
X[5000:, :] = signal_data

# 生成标签：前5000个标签为[0, 1]，后5000个标签为[1, 0]（one-hot编码）
y = np.zeros((10000, 2))  # 初始化标签为[0, 0]，大小为(10000, 2)
y[:5000, 1] = 1  # 前5000个标签设为[0, 1]
y[5000:, 0] = 1  # 后5000个标签设为[1, 0]

# 划分数据集，80%作为训练集，20%作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# 构建神经网络模型
class MLPModel(nn.Module):
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(8, 15)  # 输入层到第一隐藏层
        self.fc2 = nn.Linear(15, 16)  # 第一隐藏层到第二隐藏层
        self.fc3 = nn.Linear(16, 2)  # 第二隐藏层到输出层（2个输出：用于one-hot编码的标签）

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 第一层激活
        x = torch.relu(self.fc2(x))  # 第二层激活
        x = F.softmax(torch.relu(self.fc3(x)))  # 输出层
        return x

# 创建模型
model = MLPModel()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 多类别交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X_train_tensor)
    
    # 计算损失
    loss = criterion(y_pred, torch.max(y_train_tensor, 1)[1])  # CrossEntropyLoss需要标签为索引形式
    x = torch.max(y_train_tensor, 1)[1]
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 打印训练过程中的损失
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 测试模型
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    y_test_pred_class = torch.max(y_test_pred, 1)[1]  # 获取预测类别索引
    
    # 计算准确率
    accuracy = accuracy_score(torch.max(y_test_tensor, 1)[1].numpy(), y_test_pred_class.numpy())
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # 计算 ROC 曲线
    
