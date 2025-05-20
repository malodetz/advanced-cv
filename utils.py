import numpy as np
import torch
from torchvision.ops import nms

from config import Config


def non_max_suppression(predictions, conf_thresh=0.25, iou_thresh=0.5):
    all_boxes = []

    for pred in predictions:
        grid_size = pred.shape[0]
        boxes = []

        for i in range(grid_size):
            for j in range(grid_size):
                for b in range(Config.num_boxes):
                    data = pred[i, j, b * 5 : (b + 1) * 5]
                    conf = data[4]

                    if conf < conf_thresh:
                        continue

                    # Convert cell coordinates to image coordinates
                    x = (j + data[0]) / grid_size
                    y = (i + data[1]) / grid_size
                    w = data[2] / grid_size
                    h = data[3] / grid_size

                    x1 = x - w / 2
                    y1 = y - h / 2
                    x2 = x + w / 2
                    y2 = y + h / 2

                    # Class probabilities
                    class_probs = pred[i, j, Config.num_boxes * 5 :]
                    class_id = torch.argmax(class_probs)

                    boxes.append([x1, y1, x2, y2, conf, class_id])

        if len(boxes) == 0:
            all_boxes.append(torch.zeros((0, 6)))
            continue

        boxes = torch.tensor(boxes)
        keep = nms(boxes[:, :4], boxes[:, 4], iou_thresh)
        all_boxes.append(boxes[keep])

    return all_boxes


def calculate_iou(box1, box2):
    if not isinstance(box1, torch.Tensor):
        box1 = torch.tensor(box1)
    if not isinstance(box2, torch.Tensor):
        box2 = torch.tensor(box2)
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])
    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    if union == 0:
        return 0.0

    return intersection / union


def mean_average_precision(pred_boxes, true_boxes, iou_threshold=0.5):
    epsilon = 1e-6
    average_precisions = []

    for c in range(Config.num_classes):
        detections = []
        ground_truths = {}
        for img_idx, (img_preds, img_trues) in enumerate(zip(pred_boxes, true_boxes)):
            pred_mask = img_preds[:, 5] == c
            for det in img_preds[pred_mask]:
                detections.append([img_idx, det[4].item(), det[:4].tolist()])
            gt_mask = img_trues[:, 5] == c
            ground_truths[img_idx] = img_trues[gt_mask][:, :4].tolist()
        detections.sort(key=lambda x: x[1], reverse=True)
        tp = []
        fp = []
        total_gt = sum(len(gts) for gts in ground_truths.values())
        matched_gts = {
            img_idx: [False] * len(gts) for img_idx, gts in ground_truths.items()
        }
        for detection in detections:
            img_idx, conf, box = detection
            gts_in_image = ground_truths.get(img_idx, [])
            if len(gts_in_image) == 0:
                fp.append(1)
                tp.append(0)
                continue
            best_iou = 0.0
            best_gt = -1
            for gt_idx, gt_box in enumerate(gts_in_image):
                if not matched_gts[img_idx][gt_idx]:
                    iou = calculate_iou(box, gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = gt_idx
            if best_iou >= iou_threshold:
                matched_gts[img_idx][best_gt] = True
                tp.append(1)
                fp.append(0)
            else:
                fp.append(1)
                tp.append(0)
        if total_gt == 0:
            ap = 1.0 if len(detections) == 0 else 0.0
            average_precisions.append(ap)
            continue
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        precisions = tp_cum / (tp_cum + fp_cum + epsilon)
        recalls = tp_cum / (total_gt + epsilon)
        ap = 0.0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        average_precisions.append(ap)

    return np.mean(average_precisions) if average_precisions else 0.0
