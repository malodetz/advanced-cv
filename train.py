# train.py
import torch
import numpy as np
from config import Config
from dataset import create_dataloaders
from model import YOLOv1
from loss import YOLOLoss
from utils import non_max_suppression, mean_average_precision

from tqdm import tqdm

def train():
    # Initialize
    train_loader, val_loader = create_dataloaders()
    model = YOLOv1().to(Config.device)
    criterion = YOLOLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    
    best_map = 0.0
    
    for epoch in range(Config.epochs):
        # Training
        model.train()
        train_loss = []
        
        for images, targets in tqdm(train_loader):
            images = images.to(Config.device)
            targets = targets.to(Config.device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        
        # Validation
        model.eval()
        val_loss = []
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader):
                images = images.to(Config.device)
                targets = targets.to(Config.device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss.append(loss.item())
                
                # Convert outputs to boxes
                pred_boxes = non_max_suppression(outputs.cpu())
                all_preds.extend(pred_boxes)
                
                # Convert targets to boxes
                true_boxes = non_max_suppression(targets.cpu())
                all_targets.extend(true_boxes)
        
        # Calculate metrics
        print(train_loss)
        train_loss = np.mean(train_loss)
        val_loss = np.mean(val_loss)
        current_map = mean_average_precision(all_preds, all_targets)
        
        # Save best model
        if current_map > best_map:
            torch.save(model.state_dict(), "best_model.pth")
            best_map = current_map
        
        # Console logging
        print(f"\nEpoch {epoch+1}/{Config.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Current mAP: {current_map:.4f} | Best mAP: {best_map:.4f}")

if __name__ == "__main__":
    train()