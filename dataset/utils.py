# dataset/utils.py

import xml.etree.ElementTree as ET
from PIL import Image
import os

def parse_yolo_annotation(txt_file, img_path):
    # Get image dimensions
    img = Image.open(img_path)
    img_width, img_height = img.size
    
    boxes = []
    labels = []
    
    with open(txt_file, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            
            # Convert YOLO format (x_center, y_center, width, height) to (xmin, ymin, xmax, ymax)
            x_center = x_center * img_width
            y_center = y_center * img_height
            width = width * img_width
            height = height * img_height
            
            xmin = int(x_center - width/2)
            ymin = int(y_center - height/2)
            xmax = int(x_center + width/2)
            ymax = int(y_center + height/2)
            
            # Ensure coordinates are within image bounds
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(img_width, xmax)
            ymax = min(img_height, ymax)
            
            # Ensure box has positive area
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                # YOLO uses 0-based indexing, and we want to keep it that way
                # since Faster R-CNN will handle the background class internally
                labels.append(int(class_id))
    
    return os.path.basename(img_path), boxes, labels

def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    boxes = []
    labels = []
    filename = root.find('filename').text

    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)

    return filename, boxes, labels
