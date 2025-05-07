# train.py

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from dataset.loader import CarDataset
from models.faster_rcnn import build_faster_rcnn
from config import *
from tqdm import tqdm
import time
import csv
import os
import json
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau

def validate(model, dataloader):
    model.train()  # Temporarily switch to training mode to get loss dictionary
    val_loss = 0
    val_cls_loss = 0
    val_box_loss = 0
    with torch.no_grad():
        for images, targets in dataloader:
            images = list(img.to(DEVICE) for img in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()
            val_cls_loss += loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
            val_box_loss += loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
    avg_val_loss = val_loss / len(dataloader)
    avg_val_cls_loss = val_cls_loss / len(dataloader)
    avg_val_box_loss = val_box_loss / len(dataloader)
    return avg_val_loss, avg_val_cls_loss, avg_val_box_loss

def save_checkpoint(model, optimizer, scheduler, epoch, metrics, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'metrics': metrics,
        'segmentation_method': model.segmentation_method if hasattr(model, 'segmentation_method') else 'threshold'
    }
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Save regular checkpoint
    checkpoint_path = f'checkpoints/checkpoint_epoch_{epoch}.pth'
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if this is the best so far
    if is_best:
        best_model_path = 'checkpoints/best_model.pth'
        torch.save(checkpoint, best_model_path)
        print(f"New best model saved with validation loss: {metrics['avg_val_loss']:.4f}")

def train(segmentation_method='threshold'):
    transform = T.Compose([
        T.Resize(INPUT_SIZE),
        T.ToTensor()
    ])
    dataset = CarDataset(transforms=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

    # Validation set
    val_image_dir = os.path.join(os.path.dirname(os.path.dirname(IMAGE_DIR.rstrip('/'))), 'valid/images')
    val_ann_dir = os.path.join(os.path.dirname(os.path.dirname(ANNOTATION_DIR.rstrip('/'))), 'valid/labels')
    val_dataset = CarDataset(image_dir=val_image_dir, annotation_dir=val_ann_dir, transforms=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = build_faster_rcnn(segmentation_method=segmentation_method).to(DEVICE)
    
    # Initialize Adam optimizer with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=3,
        verbose=True,
        min_lr=1e-6
    )

    metrics_history = []
    best_val_loss = float('inf')
    start_time = time.time()

    # Create a unique run ID based on timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs('runs', exist_ok=True)
    run_dir = f'runs/run_{run_id}'
    os.makedirs(run_dir, exist_ok=True)

    model.train()
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        epoch_cls_loss = 0
        epoch_box_loss = 0
        epoch_start_time = time.time()
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}") as pbar:
            for images, targets in pbar:
                images = list(img.to(DEVICE) for img in images)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                epoch_loss += losses.item()
                cls_loss = loss_dict.get('loss_classifier', torch.tensor(0.0)).item()
                box_loss = loss_dict.get('loss_box_reg', torch.tensor(0.0)).item()
                epoch_cls_loss += cls_loss
                epoch_box_loss += box_loss

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                pbar.set_postfix({
                    'Batch Loss': losses.item(),
                    'Cls Loss': cls_loss,
                    'Box Loss': box_loss,
                    'LR': optimizer.param_groups[0]['lr']
                })
        
        epoch_end_time = time.time()
        avg_loss = epoch_loss / len(dataloader)
        avg_cls_loss = epoch_cls_loss / len(dataloader)
        avg_box_loss = epoch_box_loss / len(dataloader)
        epoch_time = epoch_end_time - epoch_start_time

        # Validation
        avg_val_loss, avg_val_cls_loss, avg_val_box_loss = validate(model, val_dataloader)
        
        # Update learning rate based on validation loss
        scheduler.step(avg_val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Prepare metrics for this epoch
        epoch_metrics = {
            'epoch': epoch + 1,
            'avg_loss': avg_loss,
            'avg_cls_loss': avg_cls_loss,
            'avg_box_loss': avg_box_loss,
            'avg_val_loss': avg_val_loss,
            'avg_val_cls_loss': avg_val_cls_loss,
            'avg_val_box_loss': avg_val_box_loss,
            'epoch_time': epoch_time,
            'learning_rate': current_lr,
            'total_time': time.time() - start_time
        }
        
        metrics_history.append(epoch_metrics)

        # Print epoch summary
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}]")
        print(f"  Loss: {avg_loss:.4f}, Cls Loss: {avg_cls_loss:.4f}, Box Loss: {avg_box_loss:.4f}")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Cls Loss: {avg_val_cls_loss:.4f}, Val Box Loss: {avg_val_box_loss:.4f}")
        print(f"  Time: {epoch_time:.2f}s, LR: {current_lr:.6f}")

        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, epoch_metrics)

        # Save best model if validation loss improved
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_checkpoint(model, optimizer, scheduler, epoch + 1, epoch_metrics, is_best=True)

        # Save metrics to CSV after each epoch
        metrics_file = os.path.join(run_dir, 'training_metrics.csv')
        with open(metrics_file, 'w', newline='') as csvfile:
            fieldnames = list(epoch_metrics.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in metrics_history:
                writer.writerow(row)

        # Save training config
        config = {
            'num_epochs': NUM_EPOCHS,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'input_size': INPUT_SIZE,
            'num_classes': NUM_CLASSES,
            'device': DEVICE,
            'optimizer': 'Adam',
            'weight_decay': 1e-4,
            'scheduler': 'ReduceLROnPlateau',
            'scheduler_patience': 3,
            'scheduler_factor': 0.1,
            'min_lr': 1e-6,
            'segmentation_method': segmentation_method
        }
        with open(os.path.join(run_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=4)

    # Save final model
    torch.save(model.state_dict(), os.path.join(run_dir, 'final_model.pth'))
    print(f"\nTraining completed. Results saved in {run_dir}")

if __name__ == "__main__":
    # You can specify which segmentation method to use
    train(segmentation_method='threshold')  # Options: 'threshold', 'edge', 'region', 'clustering'
