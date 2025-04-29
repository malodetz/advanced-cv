import cv2
import torch

from config import Config
from model import YOLOv1
from utils import non_max_suppression


def draw_boxes(image_path, output_path="a.jpg"):
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (Config.img_size, Config.img_size))

    # Convert to tensor
    image_tensor = torch.tensor(image_resized, dtype=torch.float32) / 255.0
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0).to(Config.device)

    # Load model
    model = YOLOv1().to(Config.device)
    model.load_state_dict(torch.load("best_model.pth", map_location=Config.device))
    model.eval()

    # Prediction
    with torch.no_grad():
        outputs = model(image_tensor)
    pred_boxes = non_max_suppression(
        outputs.cpu(),
        conf_thresh=Config.conf_threshold,
        iou_thresh=Config.iou_threshold,
    )

    # Draw boxes
    boxes = pred_boxes[0]
    colors = {
        0: (0, 0, 255),
        1: (255, 0, 0),
    }  # Red for class 0, Blue for class 1 (in RGB)

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        cls = int(cls)
        color = colors.get(cls, (0, 255, 0))  # Green for unknown classes

        x1 = int(x1 * Config.img_size)
        y1 = int(y1 * Config.img_size)
        x2 = int(x2 * Config.img_size)
        y2 = int(y2 * Config.img_size)

        cv2.rectangle(image_resized, (x1, y1), (x2, y2), color, 2)

    # Save result
    result_image = cv2.cvtColor(image_resized, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, result_image)


if __name__ == "__main__":
    draw_boxes("frames/Movie_10_frame_00587.jpg")
