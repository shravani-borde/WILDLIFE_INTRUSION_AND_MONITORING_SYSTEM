# ============================================================
# src/detector.py
# ============================================================

import torch
from ultralytics import YOLO

# ============================================================
# MODEL CONFIG
# ============================================================

MODEL_PATH = "yolov8l.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# LOAD MODEL
# ============================================================

model = YOLO(MODEL_PATH)

model.to(DEVICE)

# ============================================================
# CLASSES
# ============================================================

ANIMAL_CLASSES = {

    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "deer",
    "fox",
    "wolf",
    "lion",
    "tiger",
    "leopard",
    "cheetah",
    "rhino",
    "hippo",
    "monkey"
}

# ============================================================
# DETECTION FUNCTION
# ============================================================

def run_detection(

    frame,

    confidence_threshold=0.15
):

    detections = []

    results = model(

        frame,

        conf=confidence_threshold,

        verbose=False
    )

    for result in results:

        for box in result.boxes:

            cls_id = int(box.cls[0])

            class_name = model.names[cls_id]

            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            width = x2 - x1
            height = y2 - y1

            # =================================================
            # CLASSIFICATION
            # =================================================

            if class_name == "person":

                object_type = "human"

            elif class_name.lower() in ANIMAL_CLASSES:

                object_type = "animal"

            else:

                continue

            detections.append({

                "bbox": [x1, y1, width, height],

                "confidence": confidence,

                "class_name": class_name,

                "type": object_type
            })

    return detections