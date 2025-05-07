# Vehicle Detection using Faster R-CNN

This project implements a vehicle detection system using Faster R-CNN with ResNet18 backbone. The system can detect and classify five types of vehicles: Ambulance, Bus, Car, Motorcycle, and Truck.

## Project Structure
```
project/
├── data/
│   └── cars_detection/
│       └── Cars Detection/
│           ├── train/
│           │   ├── images/
│           │   └── labels/
│           ├── valid/
│           │   ├── images/
│           │   └── labels/
│           ├── test/
│           │   ├── images/
│           │   └── labels/
│           └── data.yaml
├── models/
│   └── faster_rcnn.py
├── feature_extraction/
│   └── extractor.py
├── dataset/
│   └── loader.py
├── train.py
├── inference.py
└── config.py
```

## Features
- Faster R-CNN implementation with ResNet18 backbone
- Support for 5 vehicle classes
- YOLO format dataset support
- Training with learning rate scheduling
- Comprehensive metrics tracking
- Model checkpointing
- Inference module for predictions

## Requirements
- Python 3.8+
- PyTorch with CUDA support
- torchvision
- OpenCV
- tqdm
- numpy
- PIL

## Installation
1. Clone the repository

2. Install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

3. Install other dependencies:
```bash
pip install -r requirements.txt
```

Note: Choose the appropriate CUDA version based on your NVIDIA driver. You can check your CUDA version using:
```bash
nvidia-smi
```

## Dataset
The dataset is organized in YOLO format with the following structure:
- Images in .jpg format
- Annotations in .txt format (normalized coordinates)
- 5 vehicle classes: Ambulance, Bus, Car, Motorcycle, Truck
- Split into train/valid/test sets

## Training
To train the model:
```bash
python train.py
```

Training features:
- Adam optimizer with weight decay
- Learning rate scheduling (ReduceLROnPlateau)
- Automatic checkpoint saving
- Metrics logging
- Best model tracking

## Inference
To run inference on new images:
```bash
python inference.py --image path/to/image.jpg --model path/to/model.pth
```

## Model Architecture
- Backbone: ResNet18 (non-pretrained)
- Output channels: 512
- Faster R-CNN components:
  - Region Proposal Network (RPN)
  - Feature Pyramid Network (FPN)
  - Classification head
  - Bounding box regression head

## Output
Training outputs are saved in timestamped directories under `runs/`:
- Training metrics (CSV)
- Model checkpoints
- Best model
- Configuration file
- Final model

## Configuration
Key parameters can be modified in `config.py`:
- Number of classes
- Batch size
- Learning rate
- Input size
- Training epochs
- Device settings

## Author
Mohamed Hanafy