import os
import random
import shutil

import cv2
import numpy as np
import torch

from config import Config
from model import YOLOv1
from utils import non_max_suppression


def create_or_clear_dir(directory):
    """Create or clear the specified directory"""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    print(f"Created directory: {directory}")


def load_model(model_path="best_model.pth"):
    """Load the trained model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = YOLOv1().to(Config.device)
    model.load_state_dict(torch.load(model_path, map_location=Config.device))
    model.eval()
    print(f"Model loaded from {model_path}")
    return model


def preprocess_image(image_path):
    """Preprocess an image for the model"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    original_image = image.copy()

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV_FULL)
    image = cv2.resize(image, (Config.img_size, Config.img_size))
    image = image / 255.0
    image = torch.tensor(image).permute(2, 0, 1).float().unsqueeze(0).to(Config.device)

    return image, original_image


def draw_boxes(image, boxes):
    """Draw bounding boxes with labels on the image"""
    if len(boxes) == 0:
        return image

    result = image.copy()
    class_names = ["pig", "person"]
    colors = [(0, 255, 0), (0, 0, 255)]

    for box in boxes:
        x1, y1, x2, y2, confidence, class_id = box

        height, width = result.shape[:2]
        x1 = max(0, min(int(x1 * width), width - 1))
        y1 = max(0, min(int(y1 * height), height - 1))
        x2 = max(0, min(int(x2 * width), width - 1))
        y2 = max(0, min(int(y2 * height), height - 1))

        if x2 <= x1 or y2 <= y1:
            continue

        class_idx = (
            int(class_id.item())
            if isinstance(class_id, torch.Tensor)
            else int(class_id)
        )
        class_name = class_names[class_idx % len(class_names)]
        conf_val = (
            confidence.item() if isinstance(confidence, torch.Tensor) else confidence
        )
        label = f"{class_name}: {conf_val:.2f}"
        color = colors[class_idx % len(colors)]
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y1_label = max(y1, label_size[1]) - 5
        cv2.rectangle(
            result,
            (x1, y1_label - label_size[1]),
            (x1 + label_size[0], y1_label + base_line),
            color,
            -1,
        )
        cv2.putText(
            result,
            label,
            (x1, y1_label),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

    return result


def main():
    """Main function to visualize predictions on random images"""
    try:
        visualization_dir = "visualization"
        create_or_clear_dir(visualization_dir)
        data_dir = Config.data_dir
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
        print(image_files)

        if not image_files:
            raise ValueError(f"No images found in {data_dir}")
        # num_samples = 10
        # selected_images = random.sample(image_files, min(num_samples, len(image_files)))
        selected_images = image_files
        print(f"Selected {len(selected_images)} images for visualization")
        model = load_model()
        for idx, image_file in enumerate(selected_images):
            print(f"Processing image {idx+1}/{len(selected_images)}: {image_file}")
            image_path = os.path.join(data_dir, image_file)
            input_tensor, original_image = preprocess_image(image_path)
            with torch.no_grad():
                predictions = model(input_tensor)
            filtered_boxes = non_max_suppression(
                predictions.cpu(), conf_thresh=0.9995, iou_thresh=0.8
            )[0]
            result_image = draw_boxes(original_image, filtered_boxes)
            output_path = os.path.join(visualization_dir, f"{image_file}")
            cv2.imwrite(output_path, result_image)
            print(f"  → Detected {len(filtered_boxes)} objects, saved to {output_path}")
        print(f"Visualization complete. Results saved to {visualization_dir}/")
    except Exception as e:
        print(f"Error during visualization: {str(e)}")


if __name__ == "__main__":
    main()
