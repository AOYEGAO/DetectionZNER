import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
# 生成数据：5000000个样本，每个样本8个特征
random_data = np.random.randn(5000000, 8)  # 噪声数据
signal_data = np.random.randn(5000000, 8) + 1  # 带噪声信号数据
X = np.zeros((10000000, 8))
X[:5000000, :] = signal_data
X[5000000:, :] = random_data

# 生成标签：前5000000个标签为[0, 1]，后5000000个标签为[1, 0]（one-hot编码）
y = np.zeros((10000000, 2))  # 初始化标签为[0, 0]，大小为(10000000, 2)
y[:5000000, 1] = 1  # 前5000000个标签设为[0, 1]
y[5000000:, 0] = 1  # 后5000000个标签设为[1, 0]

# 划分数据集，500000个样本作为训练集，其余作为测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=500000 / 10000000, random_state=42)

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
        x = self.fc3(x)
        #最后一层不要relu
        return x
# 创建模型，phase参数可以根据需要调整
model = MLPModel()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 多类别交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 800
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X_train_tensor)

    # 计算损失
    # loss = criterion(y_pred[1], y_train_tensor[1])  # CrossEntropyLoss需要标签为索引形式
    loss = criterion(y_pred, y_train_tensor)  # CrossEntropyLoss需要标签为索引形式  ????
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 打印训练过程中的损失
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 门限范围
thresholds = [1e-4,1e-3,1e-2,5e-2,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] #虚警率
# 用于保存每个门限下的虚警率和检测率
far_values = []
pd_values = []

for threshold in tqdm(thresholds, desc="Testing thresholds"):
    with torch.no_grad():
        y_test_pred = model(X_test_tensor)
        y_test_pred = F.softmax(y_test_pred, dim=1) # dim=1 对每行softmax
        # 假设y_test_pred是一个(batch_size, num_classes)的张量，表示每个样本在各个类别上的预测概率
        prob_class_0 = y_test_pred[:, 0]  # 获取预测值0的概率
        
        # 根据当前门限判断预测类别：如果预测值0的概率大于当前门限，则选择类别0，否则选择类别1
        y_test_pred_class = torch.where(prob_class_0 > threshold, torch.tensor(0), torch.tensor(1))

        # 计算混淆矩阵
        y_test_true_class = torch.max(y_test_tensor,1)[1]  # 获取真实类别的索引
        tn, fp, fn, tp = confusion_matrix(y_test_true_class.numpy(), y_test_pred_class.numpy()).ravel()

        # 计算虚警率 (PRF)
        false_alarm_rate = fp / (tn + fp) if (tn + fp) > 0 else 0

        # 计算检测率 (PD)
        detection_rate = tp / (tp + fn) if (tp + fn) > 0 else 0

        # 保存虚警率和检测率
        far_values.append(false_alarm_rate)
        pd_values.append(detection_rate)

# 绘制门限与虚警率的关系图
plt.figure(figsize=(10, 5))

# 绘制 PD vs PRF 图
plt.semilogx(far_values, pd_values, label="Deep Learning", color='red')

# # 加载理论数据并绘制
data = np.load("prf_pd_data.npz")
pfa_Theory = data['pfa']
pd_theory = data['pd']
plt.semilogx(pfa_Theory, pd_theory, label="Theory", color='blue')

plt.xlabel("False Alarm Rate (PRF)")
plt.ylabel("Detection Rate (PD)")
plt.title("PD vs PRF")
plt.grid(True)

# 显示图表
plt.tight_layout()
plt.show()
