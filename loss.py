import torch
from torch import nn as nn
from torch.nn import functional as F

from config import Config


def bbox_attr(data, i):
    attr_start = Config.num_classes + i
    return data[..., attr_start::5]


def get_iou(p, a):
    p_tl, p_br = bbox_to_coords(p)  # (batch, S, S, B, 2)
    a_tl, a_br = bbox_to_coords(a)

    # Largest top-left corner and smallest bottom-right corner give the intersection
    coords_join_size = (-1, -1, -1, Config.num_boxes, Config.num_boxes, 2)
    tl = torch.max(
        p_tl.unsqueeze(4).expand(
            coords_join_size
        ),  # (batch, S, S, B, 1, 2) -> (batch, S, S, B, B, 2)
        a_tl.unsqueeze(3).expand(
            coords_join_size
        ),  # (batch, S, S, 1, B, 2) -> (batch, S, S, B, B, 2)
    )
    br = torch.min(
        p_br.unsqueeze(4).expand(coords_join_size),
        a_br.unsqueeze(3).expand(coords_join_size),
    )

    intersection_sides = torch.clamp(br - tl, min=0.0)
    intersection = (
        intersection_sides[..., 0] * intersection_sides[..., 1]
    )  # (batch, S, S, B, B)

    p_area = bbox_attr(p, 2) * bbox_attr(p, 3)  # (batch, S, S, B)
    p_area = p_area.unsqueeze(4).expand_as(
        intersection
    )  # (batch, S, S, B, 1) -> (batch, S, S, B, B)

    a_area = bbox_attr(a, 2) * bbox_attr(a, 3)  # (batch, S, S, B)
    a_area = a_area.unsqueeze(3).expand_as(
        intersection
    )  # (batch, S, S, 1, B) -> (batch, S, S, B, B)

    union = p_area + a_area - intersection

    # Catch division-by-zero
    zero_unions = union == 0.0
    union[zero_unions] = 1e-6
    intersection[zero_unions] = 0.0

    return intersection / union


def bbox_to_coords(t):
    width = bbox_attr(t, 2)
    x = bbox_attr(t, 0)
    x1 = x - width / 2.0
    x2 = x + width / 2.0

    height = bbox_attr(t, 3)
    y = bbox_attr(t, 1)
    y1 = y - height / 2.0
    y2 = y + height / 2.0

    return torch.stack((x1, y1), dim=4), torch.stack((x2, y2), dim=4)


class YOLOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_coord = 5
        self.l_noobj = 0.5

    def forward(self, p, a):
        # Calculate IOU of each predicted bbox against the ground truth bbox
        iou = get_iou(p, a)  # (batch, S, S, B, B)
        max_iou = torch.max(iou, dim=-1)[0]  # (batch, S, S, B)

        # Get masks
        bbox_mask = bbox_attr(a, 4) > 0.0
        p_template = bbox_attr(p, 4) > 0.0
        obj_i = bbox_mask[..., 0:1]  # 1 if grid I has any object at all
        responsible = torch.zeros_like(p_template).scatter_(  # (batch, S, S, B)
            -1,
            torch.argmax(max_iou, dim=-1, keepdim=True),  # (batch, S, S, B)
            value=1,  # 1 if bounding box is "responsible" for predicting the object
        )
        obj_ij = obj_i * responsible  # 1 if object exists AND bbox is responsible
        noobj_ij = ~obj_ij  # Otherwise, confidence should be 0

        # XY position losses
        x_losses = mse_loss(obj_ij * bbox_attr(p, 0), obj_ij * bbox_attr(a, 0))
        y_losses = mse_loss(obj_ij * bbox_attr(p, 1), obj_ij * bbox_attr(a, 1))
        pos_losses = x_losses + y_losses
        # print('pos_losses', pos_losses.item())

        # Bbox dimension losses
        p_width = bbox_attr(p, 2)
        a_width = bbox_attr(a, 2)
        width_losses = mse_loss(
            obj_ij * torch.sign(p_width) * torch.sqrt(torch.abs(p_width) + 1e-6),
            obj_ij * torch.sqrt(a_width),
        )
        p_height = bbox_attr(p, 3)
        a_height = bbox_attr(a, 3)
        height_losses = mse_loss(
            obj_ij * torch.sign(p_height) * torch.sqrt(torch.abs(p_height) + 1e-6),
            obj_ij * torch.sqrt(a_height),
        )
        dim_losses = width_losses + height_losses
        # print('dim_losses', dim_losses.item())

        # Confidence losses (target confidence is IOU)
        obj_confidence_losses = mse_loss(
            obj_ij * bbox_attr(p, 4), obj_ij * torch.ones_like(max_iou)
        )
        # print('obj_confidence_losses', obj_confidence_losses.item())
        noobj_confidence_losses = mse_loss(
            noobj_ij * bbox_attr(p, 4), torch.zeros_like(max_iou)
        )
        # print('noobj_confidence_losses', noobj_confidence_losses.item())

        # Classification losses
        class_losses = mse_loss(
            obj_i * p[..., : Config.num_classes], obj_i * a[..., : Config.num_classes]
        )
        # print('class_losses', class_losses.item())

        total = (
            self.l_coord * (pos_losses + dim_losses)
            + obj_confidence_losses
            + self.l_noobj * noobj_confidence_losses
            + class_losses
        )
        return total / Config.batch_size


def mse_loss(a, b):
    flattened_a = torch.flatten(a, end_dim=-2)
    flattened_b = torch.flatten(b, end_dim=-2).expand_as(flattened_a)
    return F.mse_loss(flattened_a, flattened_b, reduction="sum")
