# config.py
import torch


class Config:
    # Dataset
    data_dir = "frames"
    img_size = 448
    grid_size = 7
    num_boxes = 2
    num_classes = 2
    train_ratio = 0.8

    # Model
    initial_channels = 64
    backbone_channels = [192, 256, 512, 1024]

    # Training
    batch_size = 16
    lr = 2e-5
    epochs = 100
    lambda_coord = 5
    lambda_noobj = 0.5

    # NMS
    conf_threshold = 0.25
    iou_threshold = 0.5

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
