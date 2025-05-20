import os
import random

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from config import Config


class YOLODataset(Dataset):
    def __init__(self, image_paths, labels, transform=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.grid_size = Config.grid_size
        self.num_boxes = Config.num_boxes
        self.num_classes = Config.num_classes

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
        image = cv2.resize(image, (Config.img_size, Config.img_size))
        image = image / 255.0
        target = torch.zeros(
            (self.grid_size, self.grid_size, self.num_boxes * 5 + self.num_classes)
        )
        augment1 = False
        augment2 = False
        if self.transform:
            augment1 = random.random() > 0.5
            augment2 = random.random() > 0.5
        if augment1:
            image = np.fliplr(image)
        if augment2:
            image = np.flipud(image)
        with open(self.labels[idx], "r") as f:
            for line in f.readlines():
                class_id, xc, yc, bw, bh = map(float, line.strip().split())

                if augment1:
                    xc = 1 - xc
                if augment2:
                    yc = 1 - yc

                grid_x = int(xc * self.grid_size)
                grid_y = int(yc * self.grid_size)
                x_cell = xc * self.grid_size - grid_x
                y_cell = yc * self.grid_size - grid_y
                bw_cell = bw * self.grid_size
                bh_cell = bh * self.grid_size

                class_vec = torch.zeros(self.num_classes)
                class_vec[int(class_id)] = 1
                box_data = torch.tensor([x_cell, y_cell, bw_cell, bh_cell, 1.0])
                target[grid_y, grid_x, : 5 * self.num_boxes] = torch.cat(
                    [box_data] * self.num_boxes
                )
                target[grid_y, grid_x, 5 * self.num_boxes :] = class_vec

        image = torch.tensor(image.copy()).permute(2, 0, 1).float()
        return image, target


def create_dataloaders():
    images = sorted(
        [
            os.path.join(Config.data_dir, f)
            for f in os.listdir(Config.data_dir)
            if f.endswith(".jpg")
        ]
    )
    labels = [f.replace(".jpg", ".txt") for f in images]

    train_img, val_img, train_lbl, val_lbl = train_test_split(
        images, labels, train_size=Config.train_ratio, random_state=42
    )

    train_ds = YOLODataset(train_img, train_lbl, transform=True)
    val_ds = YOLODataset(val_img, val_lbl)

    train_loader = DataLoader(
        train_ds, batch_size=Config.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=Config.batch_size, shuffle=False, num_workers=4
    )

    return train_loader, val_loader
