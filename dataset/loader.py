# dataset/loader.py

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from .utils import parse_yolo_annotation
from config import IMAGE_DIR, ANNOTATION_DIR, INPUT_SIZE

class CarDataset(Dataset):
    def __init__(self, image_dir=IMAGE_DIR, annotation_dir=ANNOTATION_DIR, transforms=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.transforms = transforms

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        ann_path = os.path.join(self.annotation_dir, img_name.replace('.jpg', '.txt'))

        image = Image.open(img_path).convert("RGB")
        filename, boxes, labels = parse_yolo_annotation(ann_path, img_path)

        target = {}
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.tensor(labels, dtype=torch.int64)
        
        if self.transforms:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.image_files)
