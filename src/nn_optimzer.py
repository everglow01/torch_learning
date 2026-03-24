import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=128,
    shuffle=True,
    num_workers=2,
    pin_memory=(device.type == "cuda"),
)

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
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.ReLU(),
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
loss = nn.CrossEntropyLoss().to(device)
tudui = Tudui().to(device)
optimizer = torch.optim.Adam(tudui.parameters(), lr=1e-3)
num_epochs = 40
model_dir = "/home/owen/桌面/torch learning/model"
model_name = "tudui_cifar10_state_dict.pt"

for epoch in range(num_epochs):
    tudui.train()
    running_loss = 0.0
    for data in dataloader:
        optimizer.zero_grad()# 清空梯度
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        output = tudui(imgs)
        # print(output)
        # print(targets)
        # print(output.shape)
        # print(targets.shape)
        loss_result = loss(output, targets)
        if not torch.isfinite(loss_result):
            print(f"Epoch {epoch + 1}: loss is NaN/Inf, stop training.")
            break
        # print(loss_result)
        loss_result.backward()# 反向传播计算梯度
        torch.nn.utils.clip_grad_norm_(tudui.parameters(), max_norm=1.0) # 梯度裁剪，防止梯度爆炸
        optimizer.step()# 更新参数
        running_loss += loss_result.item()
    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], avg_loss: {avg_loss:.4f}")

os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, model_name)
torch.save(tudui.state_dict(), model_path)
print(f"Model saved to: {model_path}")
    
