import torch


class Config:
    # Dataset
    data_dir = "frames"
    img_size = 448
    grid_size = 10
    num_boxes = 2
    num_classes = 2
    train_ratio = 0.9

    # Training
    batch_size = 16
    lr = 5e-5
    epochs = 300
    lambda_coord = 5
    lambda_noobj = 1

    # NMS
    conf_threshold = 0.25
    iou_threshold = 0.5

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
