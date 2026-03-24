import os
import random

import cv2
import torch
import torch.nn as nn
import torchvision


class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model1 = nn.Sequential(
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
            nn.Linear(64 * 4 * 4, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.model1(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])

dataset = torchvision.datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform,
)

classes = dataset.classes
model_path = "/home/owen/桌面/torch learning/model/tudui_cifar10_state_dict.pt"
image_save_path = "/home/owen/桌面/torch learning/model/random_inference_image.jpg"

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

model = Tudui().to(device)
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

random_index = random.randint(0, len(dataset) - 1)
image, target = dataset[random_index]
raw_image = dataset.data[random_index]
image = image.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(image)
    predicted_index = output.argmax(dim=1).item()

print(f"Random image index: {random_index}")
print(f"True label: {classes[target]}")
print(f"Predicted label: {classes[predicted_index]}")

display_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
display_image = cv2.resize(display_image, (320, 320), interpolation=cv2.INTER_NEAREST)
title_text = f"True: {classes[target]} | Pred: {classes[predicted_index]}"
cv2.putText(display_image, title_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
cv2.imwrite(image_save_path, display_image)
print(f"Image saved to: {image_save_path}")

try:
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if has_display:
        cv2.imshow("CIFAR10 Inference", display_image)
        print("Press 'q' or 'Esc' to close the OpenCV window.")
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q") or key == 27:
                break
        cv2.destroyAllWindows()
        print("OpenCV window closed by keyboard input.")
    else:
        print("No GUI display detected. Skip cv2.imshow and exit directly.")
except cv2.error as error:
    print(f"OpenCV preview could not be opened automatically: {error}")
    