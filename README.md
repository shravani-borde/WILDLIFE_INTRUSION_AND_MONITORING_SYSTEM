# 🦁 AI-Based Multi-Drone Wildlife Monitoring & Intrusion Detection System

A real-time AI-powered surveillance system that uses multiple drones to monitor wildlife reserves, detect animals and humans simultaneously, and trigger automatic poacher alerts.

---

## Problem Statement

Wildlife poaching and illegal human encroachment into protected forest zones is one of the most critical threats to biodiversity conservation today. Traditional monitoring methods depend on manual ground patrols, which are slow, expensive, cover limited area, and put rangers at personal risk.

There is a clear need for an automated, scalable, and intelligent system that can monitor large forest areas continuously, identify wildlife and human presence simultaneously, and raise immediate alerts when a potential poaching situation is detected — without requiring constant human supervision.

---

## Methodology

The system follows a multi-stage pipeline:

**1. Data Collection**
Drone video feeds are captured in real time from multiple aerial sources. Each drone runs as an independent parallel thread, processing its feed continuously.

**2. Dual-Model Object Detection**
Two separate YOLOv8 models are used — one trained specifically on aerial wildlife imagery (`wildlife_drone_best.pt`) for detecting animals, and one standard COCO model (`yolov8m.pt`) restricted to the person class for detecting humans. Using two models prevents cross-class confusion that occurs when a single general-purpose model is used for both wildlife and humans.

**3. Object Tracking**
DeepSORT (Deep Simple Online and Realtime Tracking) assigns persistent track IDs to each detected object across consecutive frames, enabling the system to follow individual animals and persons over time rather than treating each frame independently.

**4. Multi-Drone Fusion**
A central engine aggregates detections from all active drones, deduplicates objects spotted by multiple drones simultaneously, and maintains a unified scene understanding across all feeds.

**5. Poacher Alert Engine**
An alert is triggered automatically when at least one confirmed animal track and at least one confirmed human track appear in the same processing cycle. The alert is logged with a timestamp, species list, human count, and drone IDs.

**6. Real-Time Dashboard**
A Streamlit-based web dashboard reads the system state and auto-refreshes every second, displaying live video feeds, detection tables, species analytics, drone fleet status, and the alert timeline.

---

## Tech Stack

**AI & Computer Vision**
- YOLOv8 (Ultralytics) — object detection
- `wildlife_drone_best.pt` — custom-trained wildlife detection model (Bear, Cheetah, Crocodile, Elephant, Fox, Giraffe, Hedgehog, Jaguar, Leopard, Lion, Lynx, Ostrich, Rhinoceros, Tiger, Zebra)
- `yolov8m.pt` (COCO) — human detection
- DeepSORT (`deep-sort-realtime`) — multi-object tracking
- PyTorch — deep learning backend

**Training**
- Platform: Google Colab (T4 GPU)
- Dataset: Wild Animals — Roboflow Universe (Joel John, CC BY 4.0)
- Base model: YOLOv8m, 80 epochs, AdamW optimizer
- Augmentations: Mosaic, flip, rotation, scale — tuned for aerial drone angles

**Backend & Pipeline**
- Python 3.11
- OpenCV — video frame extraction
- Python threading — parallel multi-drone processing
- JSON — shared state between pipeline and dashboard

**Dashboard**
- Streamlit — web-based real-time UI
- Custom CSS (glassmorphism, Inter + JetBrains Mono fonts)
- Inline SVG — detection trend sparklines

---

## Project Structure

```
wp2/
├── models/
│   ├── wildlife_drone_best.pt    ← custom wildlife model
│   └── yolov8m.pt                ← COCO human detection model
├── src/
│   ├── detector.py               ← dual-model detection pipeline
│   ├── tracker.py                ← DeepSORT tracker
│   ├── animal_model_classes.py   ← wildlife class labels
│   └── human_model_classes.py    ← human class label
├── dashboard/
│   └── dashboard.py              ← Streamlit dashboard
├── data/                         ← runtime output (JSON + frames)
├── main.py                       ← single drone entry point
├── multi_drone.py                ← multi-drone engine
└── requirements.txt
```

---

## How to Run

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Single drone:**
```bash
python main.py --source your_video.mp4
```

**Multi-drone + dashboard (two terminals):**
```bash
# Terminal 1
python multi_drone.py --sources video1.mp4 video2.mp4 --feeds wildlife urban

# Terminal 2
streamlit run dashboard/dashboard.py
```
Open browser at `http://localhost:8501`

---

## Evaluation Metrics

The system is evaluated on the following:

| Metric | Description |
|---|---|
| mAP@0.5 | Mean Average Precision at IoU threshold 0.5 — primary detection accuracy metric |
| mAP@0.5:0.95 | Mean Average Precision across IoU thresholds 0.5 to 0.95 |
| Precision | Ratio of correct detections to total detections (minimises false positives) |
| Recall | Ratio of correct detections to total actual objects (minimises missed detections) |
| FPS (Frames Per Second) | Real-time processing speed per drone feed |
| Alert Latency | Time from human entering frame to alert being raised |
| Track ID Consistency | Stability of DeepSORT track IDs across frames for the same object |
| Cross-Feed Deduplication Accuracy | Correctness of merging the same object seen by two drones |

---

## Deliverables

- Trained wildlife detection model (`wildlife_drone_best.pt`) capable of detecting 15 species from aerial drone footage
- Dual-model detection pipeline with human and animal separation
- Multi-drone surveillance engine with real-time parallel processing
- DeepSORT-based object tracking with persistent IDs
- Automatic poacher alert system with logging
- Real-time web dashboard with live feeds, detection tables, analytics, and alert center
- Complete source code with setup guide

---

## Scope

**In Scope:**
- Detection of 15 wildlife species from aerial drone footage
- Human detection in protected forest zones
- Multi-drone parallel processing (tested with 2 drones, extendable)
- Real-time alerting when human-animal proximity is detected
- Live dashboard accessible from any browser on the local network

**Out of Scope:**
- GPS coordinate mapping of detected objects
- Drone hardware control or flight path automation
- Night vision / thermal imaging
- Cloud deployment or mobile app
- Identification of individual animals (species-level only, not individual)

---

## Applications

- **Forest departments** — automated surveillance of protected reserves without manual patrol
- **Anti-poaching units** — immediate alert when humans enter wildlife zones
- **Wildlife conservation NGOs** — population monitoring and activity tracking
- **National parks** — visitor encroachment detection in restricted zones
- **Research institutions** — automated wildlife census and behavioral data collection
- **Border/perimeter security** — monitoring large areas with limited human resources

---

## Future Work

- Integrate GPS tagging to map detections onto a geographic overlay
- Add night vision / thermal camera support for 24-hour monitoring
- Expand species coverage by training on larger wildlife datasets
- Implement drone-to-drone communication for coordinated patrol paths
- Deploy to cloud (AWS / GCP) for remote access from multiple locations
- Develop a mobile app for field rangers to receive alerts on their phones
- Add individual animal re-identification to track specific animals over days
- Train a behavior classification model to detect unusual activity (e.g. running, distress)

---

## Team

- **Shravani Borde**
- **Srushti Agrawal**
- **Sandarbh Singh**
- **Venkatesh Paitwar**

---

## Note

This project is developed as a **Third Year Project (Data Science / AI domain)** and focuses on real-world problem-solving using modern deep learning techniques.
