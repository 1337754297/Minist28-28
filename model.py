import torch
from torch import nn
# 定义一个神经网络类，继承自 nn.Module，这是所有神经网络模块的基类
class Network(nn.Module):
    def __init__(self):
        # 调用父类的构造函数，初始化 nn.Module 的相关功能
        super(Network, self).__init__()
        # 定义第一层全连接层，输入维度为 784 (28x28), 输出维度为 256
        self.layer1 = nn.Linear(784,256)
        # 定义第二层全连接层，输入维度为 256,
        # 输出维度为 10 (用于分类任务的输出，通常对应于类的数量)
        self.layer2 = nn.Linear(256,10)
    #定义前向传播,这里没有定义softmax
    def forward(self,x):
        # 将输入的图像数据展平成一维向量 (28x28 -> 784)
        x = x.view(-1,28*28)
        x = self.layer1(x)
        x = torch.relu(x)
        return self.layer2(x)