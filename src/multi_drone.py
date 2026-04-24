"""
multi_drone.py
--------------
Simulates multiple drone feeds running in parallel.
Each drone gets its own detector + tracker instance and writes to
its own state slot in data/multi_drone_state.json.

The MultiDroneEngine then runs cross-feed deduplication and priority
ranking exactly as described in the alert engine architecture.

Run:
    python multi_drone.py --sources 0 video1.mp4 video2.mp4

Each source = one simulated drone feed.
"""

import argparse
import json
import os
import time
import threading
import cv2

from detector import run_detection
from tracker import ObjectTracker

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

STATE_FILE   = "data/multi_drone_state.json"
FRAME_DIR    = "data/drone_frames"
DEDUP_RADIUS = 80    # pixels — two detections within this distance = same target
DEDUP_SECS   = 2     # seconds — same target window across drone feeds


# ---------------------------------------------------------------------------
# Per-drone worker
# ---------------------------------------------------------------------------

class DroneWorker:
    """
    One thread per drone source.
    Runs detection + tracking on its assigned video/webcam feed.
    Writes results into a shared dict keyed by drone_id.
    """

    def __init__(self, drone_id: str, source, shared_state: dict, lock: threading.Lock):
        self.drone_id     = drone_id
        self.source       = source
        self.shared_state = shared_state
        self.lock         = lock
        self.tracker      = ObjectTracker(max_age=30, n_init=3, max_cosine_distance=0.25)
        self.running      = False
        self._thread      = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self.running = True
        self._thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            print(f"[{self.drone_id}] Cannot open source: {self.source}")
            return

        frame_dir = os.path.join(FRAME_DIR, self.drone_id)
        os.makedirs(frame_dir, exist_ok=True)
        frame_id = 0

        print(f"[{self.drone_id}] Feed started  source={self.source}")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                # Loop video sources
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_id += 1
            detections    = run_detection(frame)
            result        = self.tracker.update(detections, frame=frame)
            tracks        = result["tracks"]
            poacher_alert = result["poacher_alert"]

            # Save annotated frame
            vis = frame.copy()
            for t in tracks:
                x, y, w, h = [int(v) for v in t["bbox"]]
                color = (0, 80, 220) if t["type"] == "human" else (30, 180, 30)
                cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
                cv2.putText(vis, f"{t['class_name']} #{t['track_id']}",
                            (x, max(y-6, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
            frame_path = os.path.join(frame_dir, "latest.jpg")
            cv2.imwrite(frame_path, vis)

            # Write into shared state under this drone's key
            with self.lock:
                self.shared_state[self.drone_id] = {
                    "drone_id":      self.drone_id,
                    "frame_id":      frame_id,
                    "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "tracks":        tracks,
                    "poacher_alert": poacher_alert,
                    "animal_count":  sum(1 for t in tracks if t["type"] == "animal"),
                    "human_count":   sum(1 for t in tracks if t["type"] == "human"),
                    "frame_path":    frame_path,
                }

        cap.release()
        print(f"[{self.drone_id}] Feed stopped.")


# ---------------------------------------------------------------------------
# Multi-drone alert engine
# ---------------------------------------------------------------------------

class MultiDroneEngine:
    """
    Reads all drone states every tick and runs:
        Stage 1  collect    — gather all tracks from all drones
        Stage 2  deduplicate — merge same-target detections across feeds
        Stage 3  rank        — sort by severity + confidence
        Stage 4  alert       — fire poacher alert if animals + humans confirmed
        Stage 5  output      — write unified state to JSON for dashboard
    """

    def __init__(self, shared_state: dict, lock: threading.Lock):
        self.shared_state = shared_state
        self.lock         = lock
        self.alert_log    = []

    def tick(self):
        with self.lock:
            drone_states = dict(self.shared_state)   # snapshot

        if not drone_states:
            return

        # -- Stage 1: collect all tracks ----------------------------------
        all_events = []
        for drone_id, ds in drone_states.items():
            for track in ds["tracks"]:
                all_events.append({
                    **track,
                    "drone_id":  drone_id,
                    "timestamp": ds["timestamp"],
                })

        # -- Stage 2: cross-feed deduplication ----------------------------
        # Two events = same target if:
        #   - same class_name
        #   - centers within DEDUP_RADIUS pixels
        # Keep only one representative per group (highest confidence)
        deduplicated = []
        used = set()

        for i, ev in enumerate(all_events):
            if i in used:
                continue
            group = [ev]
            cx1, cy1 = ev["center"]
            for j, other in enumerate(all_events):
                if j <= i or j in used:
                    continue
                if other["class_name"] != ev["class_name"]:
                    continue
                cx2, cy2 = other["center"]
                dist = ((cx1 - cx2)**2 + (cy1 - cy2)**2) ** 0.5
                if dist < DEDUP_RADIUS:
                    group.append(other)
                    used.add(j)
            used.add(i)
            # Keep the detection with highest confidence from the group
            best = max(group, key=lambda e: e["confidence"])
            # Annotate how many drones saw this target
            best["confirmed_by"] = list({g["drone_id"] for g in group})
            deduplicated.append(best)

        # -- Stage 3: priority ranking ------------------------------------
        # Score = confidence * (2 if seen by multiple drones else 1)
        # Humans score higher than animals (poacher priority)
        def _score(ev):
            multi_bonus   = 1.5 if len(ev["confirmed_by"]) > 1 else 1.0
            type_bonus    = 1.2 if ev["type"] == "human" else 1.0
            return ev["confidence"] * multi_bonus * type_bonus

        deduplicated.sort(key=_score, reverse=True)

        # -- Stage 4: alert logic -----------------------------------------
        has_animal = any(e["type"] == "animal" for e in deduplicated)
        has_human  = any(e["type"] == "human"  for e in deduplicated)
        poacher_alert = has_animal and has_human

        if poacher_alert:
            entry = {
                "timestamp":   time.strftime("%Y-%m-%dT%H:%M:%S"),
                "animals":     [e["class_name"] for e in deduplicated if e["type"] == "animal"],
                "humans":      sum(1 for e in deduplicated if e["type"] == "human"),
                "drones":      list(drone_states.keys()),
            }
            self.alert_log.append(entry)
            self.alert_log = self.alert_log[-30:]   # keep last 30

        # -- Stage 5: write unified output --------------------------------
        output = {
            "timestamp":      time.strftime("%Y-%m-%dT%H:%M:%S"),
            "drone_count":    len(drone_states),
            "drone_states":   drone_states,
            "deduplicated_tracks": deduplicated,
            "animal_count":   sum(1 for e in deduplicated if e["type"] == "animal"),
            "human_count":    sum(1 for e in deduplicated if e["type"] == "human"),
            "poacher_alert":  poacher_alert,
            "alert_log":      self.alert_log,
        }

        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, "w") as f:
            json.dump(output, f, indent=2)

        # Console summary
        print(
            f"[engine]  drones={len(drone_states)}  "
            f"raw={len(all_events)}  deduped={len(deduplicated)}  "
            f"animals={output['animal_count']}  humans={output['human_count']}  "
            f"{'*** POACHER ALERT ***' if poacher_alert else 'ok'}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(sources: list):
    shared_state = {}
    lock         = threading.Lock()

    # Start one worker per source
    workers = []
    for i, src in enumerate(sources):
        drone_id = f"drone_{i+1}"
        worker   = DroneWorker(drone_id, src, shared_state, lock)
        workers.append(worker)
        worker.start()

    engine = MultiDroneEngine(shared_state, lock)

    print(f"[engine] Multi-drone engine running  |  {len(workers)} drones  |  Ctrl+C to stop")

    try:
        while True:
            engine.tick()
            time.sleep(0.5)   # engine runs at ~2 ticks per second
    except KeyboardInterrupt:
        print("\n[engine] Stopping all drones...")
        for w in workers:
            w.stop()
        print("[engine] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-drone surveillance simulation")
    parser.add_argument(
        "--sources", nargs="+", default=["0"],
        help="List of sources: webcam index, video file paths. Each = one drone.",
    )
    args = parser.parse_args()
    run(args.sources)
