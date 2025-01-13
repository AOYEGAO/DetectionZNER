from scipy import special as sp
import numpy as np

def qfuncinv(y):
    return np.sqrt(2) * sp.erfinv(1 - 2 * y)

def qfunc(x):
    return 0.5 - 0.5 * sp.erf(x / np.sqrt(2))

def coherentPd(pfa, snrdb, numPulses):
    snr = np.power(10, snrdb/10)
    return qfunc(qfuncinv(pfa) - np.sqrt(numPulses * snr))  # 通道数的积累到信噪比上

## 相参积累检测器的检测性能
K = 8 #积累脉冲个数
pfa = np.power(10,np.arange(-6.,0.,0.2)) #虚警率
snrdb = 0 #输入信噪比
pd = []
for i in range(0,len(pfa)):
    pd_result = coherentPd(pfa[i], snrdb, K)
    pd.append(pd_result)

# 画图
import matplotlib.pyplot as plt
plt.semilogx(pfa, pd, '-')  # '*'表示标记点，'-'表示连接线

# 添加标题和标签
plt.title('ROC curve')
plt.xlabel('PFA')
plt.ylabel('PD')
plt.ylim(0,1.1)

# 显示图例
# plt.legend(['Coherent Detector'])

# 显示图表
plt.show()

# 创建文件保存数据PD和PRF
# 将数据保存为 .npz 文件
filename = "prf_pd_data.npz"
np.savez(filename, pfa=pfa, pd=pd)
print(f"PRF和PD数据已保存到文件: {filename}")
# 相参积累检测器的检测性能

