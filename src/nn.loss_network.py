import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision

dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, 
                                       transform=torchvision.transforms.ToTensor())

dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, 5, padding=2)# 输入通道数3，输出通道数32，卷积核大小5，padding=2保持输入输出尺寸不变
        # self.maxpool1 = nn.MaxPool2d(2)# 池化层，窗口大小2，步长2，输出尺寸减半
        # self.conv2 = nn.Conv2d(32, 32, 5, padding=2)# 输入通道数32，输出通道数32，卷积核大小5，padding=2保持输入输出尺寸不变
        # self.maxpool2 = nn.MaxPool2d(2)# 池化层，窗口大小2，步长2，输出尺寸减半
        # self.conv3 = nn.Conv2d(32, 64, 5, padding=2)# 输入通道数32，输出通道数64，卷积核大小5，padding=2保持输入输出尺寸
        # self.maxpool3 = nn.MaxPool2d(2)# 池化层，窗口大小2，步长2，输出尺寸减半
        # self.flatten = nn.Flatten()# 展平层，将多维输入展平为一维
        # self.Linear1 = nn.Linear(64*4*4, 64)# 全连接层，输入特征数64*4*4，输出特征数64
        # self.Linear2 = nn.Linear(64, 10)# 全连接层，输入特征数64，输出特征数10，对应10个类别
        
        self.model1=nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )
        
    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.Linear1(x)
        # x = self.Linear2(x)
        x = self.model1(x)
        
        return x
loss = nn.CrossEntropyLoss()
tudui = Tudui()
for data in dataloader:
    imgs, targets = data
    output = tudui(imgs)
    # print(output)
    # print(targets)
    # print(output.shape)
    # print(targets.shape)
    loss_result = loss(output, targets)
    # print(loss_result)
    loss_result.backward()# 反向传播计算梯度
    print("ok") 