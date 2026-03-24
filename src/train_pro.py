from pathlib import Path

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import *


# 配置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 数据增强和预处理
train_transform = torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])


# Dataset
train_data = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
test_data = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度为：{train_data_size}")
print(f"测试数据集的长度为：{test_data_size}")


# Dataloader
train_dataloader = DataLoader(
    train_data,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
)
test_dataloader = DataLoader(
    test_data,
    batch_size=128,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
)


# 加载模型网络
tudui = Tudui().to(device)
# tudui = Tudui().cuda()


# 损失函数和优化器
loss_fn = nn.CrossEntropyLoss() #交叉熵损失函数，适用于多分类问题
learning_rate = 1e-3 
weight_decay = 5e-4 # L2正则化系数，常用值为1e-4或5e-4，可以根据需要调整
optimizer = torch.optim.Adam(tudui.parameters(), lr=learning_rate, weight_decay=weight_decay) # Adam优化器，结合了动量和自适应学习率调整，适用于大多数深度学习任务
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50) # T_max是一个周期的长度，单位是epoch数，通常设置为总训练轮数的一半或总训练轮数 ，可以用来控制学习率的下降速度和周期长度，适用于需要在训练过程中逐渐降低学习率的情况


# Training settings
total_train_step = 0
epoch = 50
patience = 5
best_test_accuracy = 0.0
no_improve_epochs = 0


# Path and logger
project_root = Path(__file__).resolve().parent.parent
log_dir = project_root / "train_log_pro"
model_dir = project_root / "model"
log_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(str(log_dir))


# 训练开始 training loop
for i in range(epoch):
    current_epoch = i + 1
    print("-------------------第{}轮训练开始-------------------".format(current_epoch))

    tudui.train() # 设置模型为训练模式，启用dropout和batchnorm等训练时特有的层
    running_train_loss = 0.0
    for imgs, targets in train_dataloader:
        imgs, targets = imgs.to(device), targets.to(device)  # 将数据和标签移动到gpu上

        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)

        optimizer.zero_grad() # 清空梯度
        loss.backward()     # 误差反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(tudui.parameters(), max_norm=1.0) # 梯度裁剪，防止梯度爆炸
        optimizer.step()   # 更新参数

        total_train_step += 1
        running_train_loss += loss.item()

        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train/loss_step", loss.item(), total_train_step)

    avg_train_loss = running_train_loss / len(train_dataloader)

    tudui.eval() # 设置模型为评估模式，关闭dropout和batchnorm等训练时特有的层
    total_correct = 0
    total_test_loss = 0.0
    with torch.no_grad(): # 在评估阶段不需要计算梯度，使用torch.no_grad()上下文管理器可以节省内存和计算资源
        for imgs, targets in test_dataloader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)

            total_test_loss += loss.item()
            total_correct += (outputs.argmax(1) == targets).sum().item()

    avg_test_loss = total_test_loss / len(test_dataloader)
    test_accuracy = total_correct / test_data_size
    current_lr = optimizer.param_groups[0]["lr"] # 获取当前学习率

    print("整体测试集上的正确率：{}".format(test_accuracy))
    print("整体测试集上的平均Loss：{}".format(avg_test_loss))
    print("当前学习率：{}".format(current_lr))

    writer.add_scalar("train/loss_epoch", avg_train_loss, current_epoch)
    writer.add_scalar("test/accuracy", test_accuracy, current_epoch)
    writer.add_scalar("test/loss", avg_test_loss, current_epoch)
    writer.add_scalar("train/lr", current_lr, current_epoch)

    # 保存最后一轮的模型
    torch.save(tudui.state_dict(), model_dir / "tudui_latest_state_dict.pt")

    # 保存最优模型
    if test_accuracy > best_test_accuracy:
        best_test_accuracy = test_accuracy
        no_improve_epochs = 0
        torch.save(tudui.state_dict(), model_dir / "tudui_best_state_dict.pt")
        print("保存最佳模型，当前最佳正确率：{}".format(best_test_accuracy))
    else:
        no_improve_epochs += 1
        print("连续{}轮没有提升".format(no_improve_epochs))

    scheduler.step()

    if no_improve_epochs >= patience:
        print("触发早停，训练结束。")
        break


writer.close()
print("训练完成。最佳测试正确率：{}".format(best_test_accuracy))