import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from complexPyTorch.complexLayers import ComplexLinear
from complexPyTorch.complexFunctions import complex_relu
import torch.nn.functional as F
# 生成数据：N个样本，每个样本8个复数特征
N = 1000000
k = 8
magnitude = np.abs(np.random.randn(N//2, k))  # 瑞利分布
phase = np.random.uniform(0, 2 * np.pi, (N//2, k))  # 均匀分布，相位

# 生成复数信号：复数信号 = 幅度 * (cos(相位) + i * sin(相位))
signal_complex = magnitude * (np.cos(phase) + 1j * np.sin(phase))

# 为信号加噪声：信号的实部和虚部都加上噪声
noise = np.random.randn(N//2, k)//np.sqrt(2)  # 噪声

# 将噪声加到信号的实部和虚部
signal_complex_noisy = (signal_complex.real + noise) + 1j * (signal_complex.imag + noise)

# 合并信号和噪声
X = np.zeros((N, k), dtype=np.complex64)
X[:N//2, :] = noise + 1j * noise  # 前5000个为纯噪声
X[N//2:, :] = signal_complex_noisy  # 后5000个为信号加噪声

# 转换为torch.complex64类型
X_complex_tensor = torch.tensor(X, dtype=torch.complex64)

# 生成标签：前5000个标签为[0, 1]，后5000个标签为[1, 0]（one-hot编码）
y = np.zeros((N, 2))  # 初始化标签为[0, 0]，大小为(10000, 2)
y[:N//2, 0] = 1  # 前5000个标签设为[1, 0]
y[N//2:, 1] = 1  # 后5000个标签设为[0, 1]

# 划分数据集，80%作为训练集，20%作为测试集
X_train, _, y_train,_= train_test_split(X_complex_tensor.numpy(), y, test_size=0.9, random_state=42)
####
magnitude = 1*np.abs(np.random.randn(N//2, k))  # 瑞利分布
phase = np.random.uniform(0, 2 * np.pi, (N//2, k))  # 均匀分布，相位

# 生成复数信号：复数信号 = 幅度 * (cos(相位) + i * sin(相位))
signal_complex = magnitude * (np.cos(phase) + 1j * np.sin(phase))

# 为信号加噪声：信号的实部和虚部都加上噪声
noise = np.random.randn(N//2, k)//np.sqrt(2)  # 噪声

# 将噪声加到信号的实部和虚部
signal_complex_noisy = (signal_complex.real + noise) + 1j * (signal_complex.imag + noise)

# 合并信号和噪声
X = np.zeros((N, k), dtype=np.complex64)
X[:N//2, :] = noise + 1j * noise  # 前5000个为纯噪声
X[N//2:, :] = signal_complex_noisy  # 后5000个为信号加噪声

# 转换为torch.complex64类型
X_complex_tensor = torch.tensor(X, dtype=torch.complex64)
_, X_test, _, y_test = train_test_split(X_complex_tensor.numpy(), y, test_size=0.9, random_state=42)
#### 训练集
# 转换为 PyTorch 张量
X_train_tensor = torch.tensor(X_train, dtype=torch.complex64)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.complex64)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


def complexRelu(z):
    # """
    # 实现一个复数 ReLU 函数：
    # - 如果复数的角度在 0 到 pi/2 之间，返回原复数；
    # - 否则返回 0 + 0j。
    # """
    # 获取复数的相位
    angle = torch.angle(z)
    # 判断相位是否在 [0, pi/2] 之间
    mask = (angle >= 0) & (angle <= torch.pi / 2)
    # 使用 mask 来选择复数值
    result = torch.where(mask, z, torch.zeros_like(z))
    return result
    
# 构建神经网络模型
class MLPModel(nn.Module):
    
    def __init__(self):
        super(MLPModel, self).__init__()
        self.fc1 = ComplexLinear(8, 15)  # 输入层到第一隐藏层（每个样本8个复数特征）
        self.fc2 = ComplexLinear(15, 16)  # 第一隐藏层到第二隐藏层
        self.fc3 = ComplexLinear(16, 2)  # 第二隐藏层到输出层（2个输出：用于one-hot编码的标签）

    def forward(self, x):
        # 网络的后续传播
        # x = complexRelu(self.fc1(x))  # 第一层激活
        # x = complexRelu(self.fc2(x))  # 第二层激活
        x = complex_relu(self.fc1(x))  # 第一层激活
        x = complex_relu(self.fc2(x))  # 第二层激活
        x = self.fc3(x)  # 输出层
        x = torch.abs(x)**2
        return x
    
# 创建模型
model = MLPModel()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 多类别交叉熵损失
complex_optim = optim.AdamW(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X_train_tensor)

    # 计算损失
    # real_loss = criterion(torch.real(y_pred), y_train_tensor)
    # # imaginary part loss
    # imag_loss = criterion(torch.imag(y_pred), y_train_tensor)
    # # loss = (real_loss *100+ imag_loss) / 2
    # loss = real_loss
    loss = criterion(y_pred, y_train_tensor)
    
    # 反向传播和优化
    complex_optim.zero_grad()
    loss.backward()
    complex_optim.step()
    
    # 打印训练过程中的损失
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 测试模型
with torch.no_grad():
    y_test_pred = model(X_test_tensor)
    # y_test_pred = torch.abs(y_test_pred)
    y_test_pred = torch.real(y_test_pred)
    y_test_pred_class = torch.max(y_test_pred, 1)[1]  # 获取预测类别索引
    y_test_tensor = torch.max(y_test_tensor, 1)[1]
    # 计算准确率
    accuracy = accuracy_score(y_test_tensor.numpy(), y_test_pred_class.numpy())
    print(f"Test Accuracy: {accuracy:.4f}")

 
