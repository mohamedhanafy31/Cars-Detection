# test.py

import torch
from models.faster_rcnn import build_faster_rcnn
from dataset.loader import CarDataset
from torch.utils.data import DataLoader
import torchvision.transforms as T
from config import *

def test():
    transform = T.Compose([
        T.Resize(INPUT_SIZE),
        T.ToTensor()
    ])
    dataset = CarDataset(transforms=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = build_faster_rcnn()
    model.load_state_dict(torch.load('faster_rcnn_car.pth'))
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(DEVICE) for img in images)
            outputs = model(images)

            print(outputs)

if __name__ == "__main__":
    test()
