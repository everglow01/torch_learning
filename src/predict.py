import argparse

import torch
import torchvision
from PIL import Image

from model import Tudui


CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CIFAR10 单图推理")
    parser.add_argument(
        "--image",
        default="./imgs/11.jpeg",
        help="待推理图片路径",
    )
    parser.add_argument(
        "--weights",
        default="./model/tudui_best_state_dict.pt",
        help="模型权重路径(state_dict)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(args.image).convert("RGB")
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)

    model = Tudui().to(device)
    state_dict = torch.load(args.weights, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(logits.argmax(dim=1).item())
        confidence = float(probs[0, pred_idx].item())

    print(f"device: {device}")
    print(f"image: {args.image}")
    print(f"weights: {args.weights}")
    print(f"predicted class: {CLASSES[pred_idx]} (index={pred_idx})")
    print(f"confidence: {confidence:.4f}")
    print(logits.argmax(1))
    print(logits)


if __name__ == "__main__":
    main()