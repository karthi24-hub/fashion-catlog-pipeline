from ultralytics import YOLO
from PIL import Image
import os

YOLO_MODEL = YOLO("models/yolov8n.pt")

def detect_and_crop(image_path, output_dir, conf=0.3):
    os.makedirs(output_dir, exist_ok=True)

    image = Image.open(image_path).convert("RGB")
    results = YOLO_MODEL(image_path, conf=conf)[0]

    crops = []
    boxes = results.boxes.xyxy.cpu().numpy() if len(results.boxes) else []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        # clamp coordinates
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)

        crop = image.crop((x1, y1, x2, y2))
        base = os.path.splitext(os.path.basename(image_path))[0]
        crop_name = f"{base}_crop_{i}.jpg"
        crop_path = os.path.join(output_dir, crop_name)

        # avoid overwriting existing crops
        if not os.path.exists(crop_path):
            crop.save(crop_path)

        crops.append(crop_path)

    return crops
