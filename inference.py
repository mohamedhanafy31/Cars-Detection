# inference.py

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from models.faster_rcnn import build_faster_rcnn
from config import DEVICE, INPUT_SIZE, NUM_CLASSES
import cv2

class CarDetector:
    def __init__(self, model_path, confidence_threshold=0.5, segmentation_method='threshold'):
        """
        Initialize the car detector with a trained model.
        
        Args:
            model_path (str): Path to the trained model weights (.pth file)
            confidence_threshold (float): Minimum confidence score for detections (0-1)
            segmentation_method (str): Method to use for segmentation ('threshold', 'edge', 'region', 'clustering')
        """
        self.device = torch.device(DEVICE)
        self.model = build_faster_rcnn(segmentation_method=segmentation_method).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        self.confidence_threshold = confidence_threshold
        self.transform = T.Compose([
            T.Resize(INPUT_SIZE),
            T.ToTensor()
        ])
        
        # Class names from data.yaml
        self.class_names = ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
        
        # Define colors for each class (BGR format)
        self.colors = {
            'Ambulance': (0, 0, 255),    # Red
            'Bus': (255, 0, 0),         # Blue
            'Car': (0, 255, 0),         # Green
            'Motorcycle': (255, 255, 0), # Cyan
            'Truck': (0, 255, 255)      # Yellow
        }

    def preprocess_image(self, image):
        """
        Preprocess the input image for the model.
        
        Args:
            image: PIL Image or numpy array (BGR format from cv2)
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if image is from cv2
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transformations
        image_tensor = self.transform(image)
        return image_tensor

    def postprocess_predictions(self, predictions, original_size):
        """Convert predictions to original image coordinates and filter by confidence"""
        detections = []
        
        # Get boxes, scores, and labels
        boxes = predictions['boxes'].cpu().detach().numpy()
        scores = predictions['scores'].cpu().detach().numpy()
        labels = predictions['labels'].cpu().detach().numpy()
        
        # Convert normalized coordinates to pixel coordinates
        height, width = original_size
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        
        # Filter predictions by confidence threshold
        mask = scores > self.confidence_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        labels = labels[mask]
        
        # Convert to list of dictionaries
        for box, score, label in zip(boxes, scores, labels):
            detections.append({
                'box': box.tolist(),  # [x1, y1, x2, y2]
                'score': float(score),
                'class_id': int(label),
                'class_name': self.class_names[int(label)]
            })
        
        return detections

    def detect(self, image):
        """
        Detect vehicles in an image.
        
        Args:
            image: PIL Image or numpy array (BGR format from cv2)
            
        Returns:
            list: List of dictionaries containing detection results
        """
        # Get original image size
        if isinstance(image, np.ndarray):
            original_size = image.shape[:2]  # (height, width)
        else:
            original_size = image.size[::-1]  # (height, width)

        # Preprocess image
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)

        # Get predictions
        with torch.no_grad():
            predictions = self.model([image_tensor])

        # Process predictions
        detections = self.postprocess_predictions(predictions[0], original_size)
        return detections

    def draw_detections(self, image, detections):
        """
        Draw segmentation masks, bounding boxes, and labels on the image.
        
        Args:
            image: PIL Image or numpy array (BGR format from cv2)
            detections: List of detection dictionaries from detect()
            
        Returns:
            numpy.ndarray: Image with detections drawn (BGR format)
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Create a copy of the image for drawing
        result_image = image.copy()
        
        # Create a separate image for masks
        mask_image = np.zeros_like(image)

        for det in detections:
            box = det['box']
            score = det['score']
            class_name = det['class_name']
            
            # Get color for this class
            color = self.colors[class_name]
            
            # Draw bounding box
            cv2.rectangle(result_image, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Draw label with background
            label = f"{class_name}: {score:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(result_image, (box[0], box[1] - label_h - 10), 
                         (box[0] + label_w, box[1]), color, -1)
            cv2.putText(result_image, label, (box[0], box[1] - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Blend the mask image with the result image
        alpha = 0.5  # Transparency factor
        result_image = cv2.addWeighted(result_image, 1, mask_image, alpha, 0)

        return result_image

def main():
    # Example usage
    detector = CarDetector('runs/run_20250507_190227/final_model.pth')
    
    # Load and process an image
    image_path = 'data/cars_detection/Cars Detection/train/images/0df5a680a412fa0b_jpg.rf.TAvwaNBqf1CxVWb2bgXP.jpg'
    image = cv2.imread(image_path)
    
    # Get detections
    detections = detector.detect(image)
    
    # Draw detections
    result_image = detector.draw_detections(image, detections)
    
    # Save or display result
    cv2.imwrite('result.jpg', result_image)
    cv2.imshow('Detections', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 