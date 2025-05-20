import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import wandb
from config import Config
from dataset import create_dataloaders
from loss import YOLOLoss
from model import YOLOv1
from utils import mean_average_precision, non_max_suppression


def train():
    wandb.init(project="yolo-pigs-detection")

    train_loader, val_loader = create_dataloaders()
    model = YOLOv1().to(Config.device)
    criterion = YOLOLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.epochs, eta_min=1e-5)

    wandb.watch(model, log_freq=100)

    best_map = 0.0
    logging_interval = 20
    current_step = 0

    for epoch in range(Config.epochs):
        model.train()
        train_loss = []

        for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
            current_step += 1
            images = images.to(Config.device)
            targets = targets.to(Config.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            grad_norm = total_norm**0.5

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=50)

            optimizer.step()

            train_loss.append(loss.item())
            if current_step % logging_interval == 0:
                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "grad_norm": (
                            grad_norm.item()
                            if isinstance(grad_norm, torch.Tensor)
                            else grad_norm
                        ),
                    }
                )
        scheduler.step()

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

                pred_boxes = non_max_suppression(outputs.cpu())
                all_preds.extend(pred_boxes)

                true_boxes = non_max_suppression(targets.cpu())
                all_targets.extend(true_boxes)

        train_loss = np.mean(train_loss)
        val_loss = np.mean(val_loss)
        current_map = mean_average_precision(all_preds, all_targets)

        if current_map > best_map:
            torch.save(model.state_dict(), "best_model.pth")
            best_map = current_map

        print(f"\nEpoch {epoch+1}/{Config.epochs}")
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Current mAP: {current_map:.4f} | Best mAP: {best_map:.4f}")

        wandb.log(
            {
                "val_loss": val_loss,
                "mAP": current_map,
            }
        )

    wandb.finish()


if __name__ == "__main__":
    train()
