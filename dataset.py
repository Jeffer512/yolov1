import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from config import S_H, S_W, B, C
  

class YOLOv1Dataset(Dataset):
    """Dataset class for YOLOv1 object detection.
    
    Loads images, applies transforms, and maps targets to a grid tensor. 
    Images must satisfy W = 32*S_W and H = 32*S_H (or be resized via transforms).
    Targets should be in YOLOv5 format.
    """    
    def __init__(self, data_dir, split, S_H=S_H, S_W=S_W, num_classes=C, transforms=None):
        self.img_dir = os.path.join(data_dir, split, "images")
        self.label_dir = os.path.join(data_dir, split, "labels")
        self.S_H = S_H
        self.S_W = S_W
        self.B = B
        self.C = num_classes
        self.transforms = transforms

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
        # Load image and convert to NumPy array (required for Albumentations)
        img_path = self.img_files[idx]
        image = Image.open(img_path).convert("RGB")
        image_np = np.array(image)

        label_filename = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_filename)

        # Separate coordinates and class IDs for Albumentations compatibility
        bboxes = []
        class_labels = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    class_id, x, y, w, h = map(float, line.split())
                    bboxes.append([x, y, w, h])
                    class_labels.append(int(class_id))

        # Apply spatial transformations and coordinate recalculations
        if self.transforms:
            augmented = self.transforms(image=image_np, bboxes=bboxes, class_labels=class_labels)
            image = augmented["image"]
            bboxes = augmented["bboxes"]
            class_labels = augmented["class_labels"]
            
            # If Albumentations pipeline doesn't contain 'ToTensorV2()'
            if not isinstance(image, torch.Tensor):
                # Convert NumPy HWC (H, W, C) to PyTorch CHW (C, H, W) tensor
                image = torch.from_numpy(image).permute(2, 0, 1)
                if image.dtype == torch.uint8:
                    image = image.float() / 255.0
        else:
            # Fallback tensor conversion and normalization if no pipeline is provided
            import torchvision.transforms as T
            fallback_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            image = fallback_transform(image)

        # Build target tensor of shape (S_H, S_W, B * 5 + C)
        target = torch.zeros((self.S_H, self.S_W, self.B * 5 + self.C))

        for box_idx, box in enumerate(bboxes):
            x, y, w, h = box
            class_id = class_labels[box_idx]

            # Map absolute center coordinates to grid cells
            col = int(x * self.S_W)
            row = int(y * self.S_H)

            # Assign object parameters to cell if currently unoccupied
            if target[row, col, 0] == 0:
                target[row, col, 0] = 1.0

                # Calculate cell-relative offsets
                x_cell = (x * self.S_W) - col
                y_cell = (y * self.S_H) - row
                target[row, col, 1:5] = torch.tensor([x_cell, y_cell, w, h])
                target[row, col, 5 + int(class_id)] = 1.0

        return image, target