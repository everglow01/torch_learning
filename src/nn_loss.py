import torch
import torch.nn as nn

input = torch.tensor([1, 2, 3], dtype=torch.float32)
target = torch.tensor([1, 2, 5], dtype=torch.float32)

input = torch.reshape(input, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

loss = nn.L1Loss(reduction="sum")# 定义L1损失函数
result = loss(input, target)# 计算损失

loss_mse = nn.MSELoss(reduction="sum")# 定义MSE损失函数
result_mse = loss_mse(input, target)# 计算MSE损失

print(result)
print(result_mse) 

x = torch.tensor([0.1,0.2,0.3], dtype=torch.float32)
y = torch.tensor([1], dtype=torch.long)
x = torch.reshape(x, (1, 3))# 将输入数据x调整为二维张量，形状为(1, 3)，表示一个样本有三个特征
loss_cross = nn.CrossEntropyLoss()# 定义交叉熵损失函数
result_cross = loss_cross(x, y)# 计算交叉熵损失，输入x是模型的输出，y是目标标签
'''-0.2+ln(exp(0.1)+exp(0.2)+exp(0.3))'''
print(result_cross)