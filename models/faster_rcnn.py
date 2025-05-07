# models/faster_rcnn.py

import torchvision
from feature_extraction.extractor import build_backbone
from config import NUM_CLASSES

def build_faster_rcnn():
    backbone = build_backbone()
    anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'], output_size=7, sampling_ratio=2)

    model = torchvision.models.detection.FasterRCNN(
        backbone, num_classes=NUM_CLASSES,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )

    return model
