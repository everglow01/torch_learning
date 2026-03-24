import torch
import torchvision
# 加载方式1：
model = torch.load("./vgg16/vgg16_model.pth")

print(model)

# 加载方式2：
vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("./vgg16/vgg16_model_state_dict.pt"))
print(vgg16)
vgg16_state_dict = torch.load("./vgg16/vgg16_model_state_dict.pt")
# print(vgg16_state_dict)