import torchvision
import torch
from torch.utils.data import DataLoader
from torch import nn
from model import *


# 设定transform，正则化使用CIFAR10数据集的均值和标准差进行归一化
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
# 准备数据集

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

# Length长度
train_data_size = len(train_data)
test_data_size = len(test_data)

print(f"训练数据集的长度为：{train_data_size}")
print(f"测试数据集的长度为：{test_data_size}")

train_dataloader = DataLoader(train_data, batch_size=128, shuffle=True, num_workers=2,pin_memory=(device.type == "cuda"))
test_dataloader = DataLoader(test_data, batch_size=128, shuffle=False, num_workers=2,pin_memory=(device.type == "cuda"))

# 定义模型

#创建模型实例
tudui = Tudui().to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.001
optimizer = torch.optim.Adam(tudui.parameters(), learning_rate) 


# 设置参数
total_train_step = 0
total_test_step = 0
epoch = 15

# 添加tensorboard
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("./train_log")

for i in range(epoch):
    print("-------------------第{}轮训练开始-------------------".format(i+1))
    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tudui.parameters(), max_norm=1.0) # 梯度裁剪，防止梯度爆炸
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0: # 每100次训练输出一次结果
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)
    # c测试步骤开始
    total_accuracy = 0
    total_teat_loss = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs, targets = data
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_teat_loss += loss.item()
            total_test_step += 1
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy.item()
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    print("整体测试集上的Loss：{}".format(total_teat_loss))
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    writer.add_scalar("test_loss", total_teat_loss, total_test_step)
    
    torch.save(tudui, "./model/tudui_{}.pth".format(i))
    
writer.close()