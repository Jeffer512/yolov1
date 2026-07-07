import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from config import S, B, C, IMAGE_SIZE
  
class YOLOv1Dataset(Dataset):
    def __init__(self, data_dir, split, normalize=True, augment=True):
        self.img_dir = os.path.join(data_dir, split, "images")
        self.label_dir = os.path.join(data_dir, split, "labels")
        self.S = S
        self.B = B
        self.C = C
        self.normalize = normalize
        self.augment = augment

        # Filter specifically for image extensions to prevent crashes
        img_extensions = {".jpg", ".jpeg", ".png"}

        self.img_files = [
            os.path.join(self.img_dir, file) for file in os.listdir(self.img_dir)
            if os.path.splitext(file)[1].lower() in img_extensions
        ]

        self.img_files.sort()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")

        label_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_filename)

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    class_id, x, y, w, h = map(float, line.split())
                    boxes.append([class_id, x, y, w, h])

        # Define the transforms pipeline in the correct logical order
        transforms_list = [T.Resize((IMAGE_SIZE, IMAGE_SIZE))]
        
        if self.augment:
            transforms_list.append(T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
            
        transforms_list.append(T.ToTensor())
        
        if self.normalize:
            transforms_list.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))        

        transforms = T.Compose(transforms_list)
        
        image = transforms(image)
        
        # Build the YOLO Target Tensor (S, S, B*5 + C)
        target = torch.zeros((self.S, self.S, self.B * 5 + self.C))

        for box in boxes:
            class_id, x, y, w, h = box
            class_id = int(class_id)

            col = int(x * self.S)
            row = int(y * self.S)

            if target[row, col, 0] == 0:
                target[row, col, 0] = 1.0
                x_cell = (x * self.S) - col
                y_cell = (y * self.S) - row
                target[row, col, 1:5] = torch.tensor([x_cell, y_cell, w, h])
                target[row, col, 5 + class_id] = 1.0

        return image, target