# models/faster_rcnn.py

import torchvision
from feature_extraction.extractor import build_backbone
from config import NUM_CLASSES
from segmentation.segmenter import threshold_segmentation, edge_segmentation, region_segmentation, clustering_segmentation
import torch
import numpy as np

class SegmentedFasterRCNN(torch.nn.Module):
    def __init__(self, segmentation_method='threshold'):
        super().__init__()
        self.backbone = build_backbone()
        self.anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),) * 5
        )
        
        self.roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'], output_size=7, sampling_ratio=2)
            
        self.model = torchvision.models.detection.FasterRCNN(
            self.backbone, num_classes=NUM_CLASSES,
            rpn_anchor_generator=self.anchor_generator,
            box_roi_pool=self.roi_pooler
        )
        
        self.segmentation_method = segmentation_method
        
    def preprocess_image(self, image):
        # Convert tensor to numpy for segmentation
        if isinstance(image, torch.Tensor):
            image_np = image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
        else:
            image_np = image
            
        # Apply segmentation
        if self.segmentation_method == 'threshold':
            segmented = threshold_segmentation(image_np)
        elif self.segmentation_method == 'edge':
            segmented = edge_segmentation(image_np)
        elif self.segmentation_method == 'region':
            segmented = region_segmentation(image_np)
        elif self.segmentation_method == 'clustering':
            segmented = clustering_segmentation(image_np)
        else:
            raise ValueError(f"Unknown segmentation method: {self.segmentation_method}")
            
        # Convert back to tensor
        segmented = torch.from_numpy(segmented).float() / 255.0
        if len(segmented.shape) == 2:
            segmented = segmented.unsqueeze(0).repeat(3, 1, 1)
        return segmented.unsqueeze(0)
        
    def forward(self, images, targets=None):
        # Preprocess images with segmentation
        segmented_images = [self.preprocess_image(img) for img in images]
        segmented_images = torch.stack(segmented_images)
        
        # Forward pass through Faster R-CNN
        return self.model(segmented_images, targets)

def build_faster_rcnn(segmentation_method='threshold'):
    return SegmentedFasterRCNN(segmentation_method=segmentation_method)
