import torch
import torch.nn.functional as F
import torch.nn as nn

"""网络结构"""
class fc3layer(torch.nn.Module):


    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.fc3layer = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 32),
            torch.nn.SiLU(),
            torch.nn.Linear(32, 64),
            torch.nn.SiLU(),
            torch.nn.Linear(64, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 256),
            torch.nn.SiLU(),
            # torch.nn.Linear(256, out_channels)
        )


    def forward(self, x):
        return self.fc3layer(x)
class FC(torch.nn.Module):
    def __init__(self,phase):
        super(FC, self).__init__()
        # self.channel=int(8*(phase+1))
        if phase:
            self.channel = 8
        else:
            self.channel = 8
        self.batch=100
        self.amp=fc3layer(self.channel,2)
        self.pha = fc3layer(self.channel, 2)
        self.phase=phase
        self.relu = torch.nn.SiLU()
        self.sq= torch.nn.Linear(2, 64)
        self.cls=torch.nn.Linear(256, 2)



    def forward(self, x):
        if self.phase:
            ...
        else:
            x = x.view(-1, self.channel)
        # print(x.shape)
            x = self.amp(x)
            x = self.relu(x)
        return x

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
# from net import FC, SignalClassifier
# from dataset import randsn
import numpy as np
import torch.nn.functional as F
"""roc曲线"""
lPfa=[]
lPd=[]
# gates=[1e-4,1e-3,1e-2,5e-2,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
gates = np.power(10,np.arange(-20,-10.1,0.9)) #虚警率
len_test=int(1e7)
ktest=0.5
def test(gate):
    correct = 0
    total = 0
    fa=0
    d=0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            outputs= softmax(outputs, dim=0)
            predicted = torch.where(outputs.data[0]< gate, 1, 0)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            fa+= (predicted *(1-labels) ).sum().item()
            d+= (predicted *labels ).sum().item()

        print('FC trained model: accuracy on mymnist set:%d %%' % (100 * correct / total))
        return fa,d
# gates=np.insert(gates,0,0)
#gates=np.append(gates,1)
for gate in gates:
    fa, d = test(gate)
    Pfa = fa / len_test / ktest
    Pd = d / len_test /(1-ktest)
    print(Pfa,Pd)
    lPfa.append(Pfa)
    lPd.append(Pd)
