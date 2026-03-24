import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

# train_data = torchvision.datasets.ImageNet(root="./data_image_data", split="train", download=True, 
#                                           transform=transform)

# False表示不加载预训练权重，True表示加载预训练权重
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_True = torchvision.models.vgg16(pretrained=True)


train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
'''增加一个全连接层，输入特征数1000，输出特征数10，对应10个类别'''
vgg16_True.classifier.add_module('add_Linear', nn.Linear(1000, 10))
print(vgg16_True)
'''替换掉原来的全连接层，输入特征数4096，输出特征数10，对应10个类别'''
vgg16_false.classifier[6] = nn.Linear(4096, 10) # 替换掉原来的全连接层，输入特征数4096，输出特征数10，对应10个类别
print(vgg16_false) 