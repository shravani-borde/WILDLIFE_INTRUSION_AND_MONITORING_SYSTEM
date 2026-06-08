"""
detector.py — fixed Conv.bn fuse crash for wildlife_drone_best.pt
"""
import torch
import torch.nn as nn
from ultralytics import YOLO

ANIMAL_MODEL_PATH = "models/wildlife_drone_best.pt"
HUMAN_MODEL_PATH  = "models/yolov8m.pt"

ANIMAL_CONF = 0.30
HUMAN_CONF  = 0.40

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def _patch_model_no_fuse(yolo_model):
    """
    Patch every Conv layer that has no 'bn' attribute so that
    ultralytics fuse() won't crash with AttributeError.
    """
    for m in yolo_model.model.modules():
        if m.__class__.__name__ == "Conv":
            if not hasattr(m, "bn"):
                m.bn = None   # add dummy so fuse() check passes safely
    return yolo_model

print(f"[detector] Loading animal model: {ANIMAL_MODEL_PATH}")
animal_model = YOLO(ANIMAL_MODEL_PATH)
_patch_model_no_fuse(animal_model)
animal_model.to(DEVICE)

print(f"[detector] Loading human model: {HUMAN_MODEL_PATH}")
human_model = YOLO(HUMAN_MODEL_PATH)
human_model.to(DEVICE)

# Exclude class 7 (Human) — handled by yolov8m.pt
ANIMAL_CLASS_IDS = [k for k, v in animal_model.names.items() if v.lower() != "human"]

def run_detection(frame, animal_conf=ANIMAL_CONF, human_conf=HUMAN_CONF):
    detections = []

    try:
        for result in animal_model(frame, conf=animal_conf, verbose=False,
                           classes=ANIMAL_CLASS_IDS):
            for box in result.boxes:
                cls_id = int(box.cls[0])

                # DEBUG
                print(
                    f"DETECTED -> {animal_model.names[cls_id]} | "
                    f"conf={round(float(box.conf[0]),3)}"
                )

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                print(
                    f"Frame Animal -> "
                    f"{animal_model.names[cls_id]} "
                    f"conf={round(float(box.conf[0]),3)} "
                    f"box={w}x{h}"
                )

                detections.append({
                    "bbox": [x1, y1, w, h],
                    "center": [round(x1 + w/2, 2), round(y1 + h/2, 2)],
                    "confidence": round(float(box.conf[0]), 4),
                    "class_id": cls_id,
                    "class_name": animal_model.names[cls_id],
                    "type": "animal",
                })
    except Exception as e:
        print(f"[detector] animal model error: {e}")

    try:
        for result in human_model(frame, conf=human_conf, verbose=False, classes=[0]):
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1
                detections.append({
                    "bbox":       [x1, y1, w, h],
                    "center":     [round(x1 + w/2, 2), round(y1 + h/2, 2)],
                    "confidence": round(float(box.conf[0]), 4),
                    "class_id":   0,
                    "class_name": "person",
                    "type":       "human",
                })
    except Exception as e:
        print(f"[detector] human model error: {e}")

    return detections