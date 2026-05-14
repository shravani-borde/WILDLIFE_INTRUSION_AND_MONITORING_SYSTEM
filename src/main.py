"""
main.py
-------
Entry point for the drone surveillance pipeline.

Wires together:
    detector.py  ->  tracker.py  ->  shared state (JSON + frame image)

Run on a video file:
    python main.py --source video.mp4

Run on a folder of frames:
    python main.py --source data/frames/

Run on webcam (index 0):
    python main.py --source 0
"""

import argparse
import json
import os
import time
import cv2

from src.detector import run_detection
from src.tracker import get_tracker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUTPUT_JSON  = "data/detections.json"   # read by Streamlit dashboard
OUTPUT_FRAME = "data/latest_frame.jpg"  # latest annotated frame for dashboard
SHOW_WINDOW  = False                    # set True to show OpenCV window locally


# ---------------------------------------------------------------------------
# Shared state helpers
# ---------------------------------------------------------------------------

def _init_state() -> dict:
    return {
        "frame_id":      0,
        "timestamp":     None,
        "tracks":        [],
        "poacher_alert": False,
        "animal_count":  0,
        "human_count":   0,
        "alert_log":     [],        # rolling log of last 20 poacher alerts
    }


def _save_state(state: dict):
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(state, f, indent=2)


# ---------------------------------------------------------------------------
# Visualisation helper
# ---------------------------------------------------------------------------

def _draw_tracks(frame, tracks: list[dict], poacher_alert: bool):
    """Draw bboxes and labels onto the frame in-place."""
    for obj in tracks:
        x, y, w, h = [int(v) for v in obj["bbox"]]
        label = f"{obj['class_name']} #{obj['track_id']} {obj['confidence']:.2f}"
        color = (0, 80, 220) if obj["type"] == "human" else (30, 180, 30)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, max(y - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if poacher_alert:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 200), -1)
        cv2.putText(frame, "POACHER ALERT", (12, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(source):
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {source}")

    tracker  = get_tracker(max_age=30, n_init=2)
    frame_id = 0
    state    = _init_state()

    print(f"[main] Pipeline started  |  source={source}")
    print(f"[main] Press  Q  to quit (if window is open)")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[main] End of source.")
            break

        frame_id += 1

        # -- Step 1: detect ------------------------------------------------
        detections = run_detection(frame)

        # -- Step 2: track -------------------------------------------------
        result        = tracker.update(detections, frame=frame)
        tracks        = result["tracks"]
        poacher_alert = result["poacher_alert"]

        # -- Step 3: update shared state -----------------------------------
        state["frame_id"]      = frame_id
        state["timestamp"]     = time.strftime("%Y-%m-%dT%H:%M:%S")
        state["tracks"]        = tracks
        state["poacher_alert"] = poacher_alert
        state["animal_count"]  = sum(1 for t in tracks if t["type"] == "animal")
        state["human_count"]   = sum(1 for t in tracks if t["type"] == "human")

        # Append to rolling alert log (keep last 20)
        if poacher_alert:
            state["alert_log"].append({
                "frame_id":  frame_id,
                "timestamp": state["timestamp"],
                "animals":   [t["class_name"] for t in tracks if t["type"] == "animal"],
                "humans":    state["human_count"],
            })
            state["alert_log"] = state["alert_log"][-20:]

        _save_state(state)

        # -- Step 4: save annotated frame for dashboard --------------------
        vis = _draw_tracks(frame.copy(), tracks, poacher_alert)
        os.makedirs(os.path.dirname(OUTPUT_FRAME), exist_ok=True)
        cv2.imwrite(OUTPUT_FRAME, vis)

        # -- Step 5: console summary ---------------------------------------
        print(
            f"[frame {frame_id:05d}]  "
            f"animals={state['animal_count']}  "
            f"humans={state['human_count']}  "
            f"{'*** POACHER ALERT ***' if poacher_alert else 'ok'}"
        )

        # -- Step 6: optional live OpenCV window ---------------------------
        if SHOW_WINDOW:
            cv2.imshow("Drone Surveillance", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[main] Quit signal received.")
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[main] Pipeline stopped.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone surveillance pipeline")
    parser.add_argument(
        "--source", default="0",
        help="Video file, image folder, or webcam index (default: 0)",
    )
    args = parser.parse_args()
    run(args.source)
