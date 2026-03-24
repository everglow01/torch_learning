from pathlib import Path

import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

save_dir = Path(__file__).resolve().parent.parent / "vgg16"
save_dir.mkdir(parents=True, exist_ok=True)

# 保存方式1：
torch.save(vgg16, save_dir / "vgg16_model.pth")

# 保存方式2：
torch.save(vgg16.state_dict(), save_dir / "vgg16_model_state_dict.pt")