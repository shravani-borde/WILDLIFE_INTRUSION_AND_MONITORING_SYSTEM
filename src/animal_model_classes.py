# Classes from wildlife_drone_best.pt
# Trained on Wild Animals dataset (Roboflow) — drone/aerial view
# Note: Human class (7) exists but we use yolov8m.pt for human detection

ANIMAL_CLASSES = {
    0:  "Bear",
    1:  "Cheetah",
    2:  "Crocodile",
    3:  "Elephant",
    4:  "Fox",
    5:  "Giraffe",
    6:  "Hedgehog",
    8:  "Leopard",
    9:  "Lion",
    10: "Lynx",
    11: "Ostrich",
    12: "Rhinoceros",
    13: "Tiger",
    14: "Zebra",
}
# class 7 = Human — handled by yolov8m.pt instead
