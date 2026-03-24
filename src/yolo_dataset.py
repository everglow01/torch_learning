import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import torchvision

class YOLOdataset(Dataset):
    def __init__(self, image_folder, label_folder, transform=None, label_transform=None):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        self.image_name = os.listdir(self.image_folder)
        self.classes_list = ["no helmet", "motor", "number", "with helmet"]
        
    def __len__(self):
        return len(self.image_name)
    
    def __getitem__(self, index):
        img_name = self.image_name[index]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        # new.png -> new.txt
        # new.png -> [new,png] -> new+".txt" -> new.txt
        label_name = img_name.split(".")[0] + ".txt"
        label_path = os.path.join(self.label_folder, label_name)
        with open(label_path, "r", encoding="utf-8") as f:
            label_content = f.read()
        object_infos = label_content.strip().split("\n") # 去掉头尾的空行，并按行分割
        target = []
        for object_info in object_infos:
            class_id, x_center, y_center, width, height = object_info.strip().split(" ") # 去掉行首尾的空格，并按空格分割
            class_id = int(class_id)
            x_center = float(x_center)
            y_center = float(y_center)
            width = float(width) 
            height = float(height)
            # 这里可以根据需要将这些信息转换成你想要的格式，例如：
            # label_dict = {
            #     "class_id": class_id,
            #     "x_center": x_center,
            #     "y_center": y_center,
            #     "width": width,
            #     "height": height
            # }
            target.append((class_id, x_center, y_center, width, height))
        if self.transform:
            image = self.transform(image)
        if self.label_transform:
            target = self.label_transform(target)
        return image, target
    
if __name__ == "__main__":
    dataset = YOLOdataset(image_folder = "/home/owen/桌面/torch learning/HelmetDataset-YOLO-Val/images"
                          , label_folder = "/home/owen/桌面/torch learning/HelmetDataset-YOLO-Val/labels",
                          transform=torchvision.transforms.Compose([torchvision.transforms.Resize((640, 640)),torchvision.transforms.ToTensor()]),
                          label_transform=torchvision.transforms.Compose([torch.tensor])
                          )
    print(f"Dataset size: {len(dataset)}")
    for i in range(5):
        image, target = dataset[i]
        print(f"Image {i} shape: {tuple(image.shape)}")
        print(f"Target {i}: {target}")
