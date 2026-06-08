import argparse, json, os, time
import cv2
from src.detector import run_detection
from src.tracker import get_tracker

OUTPUT_JSON  = "data/detections.json"
OUTPUT_FRAME = "data/latest_frame.jpg"

def _save(state):
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_JSON, "w") as f: json.dump(state, f, indent=2)

def _draw(frame, tracks, alert):
    for o in tracks:
        x, y, w, h = [int(v) for v in o["bbox"]]
        color = (0, 80, 220) if o["type"] == "human" else (30, 180, 30)
        cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)
        cv2.putText(frame, f"{o['class_name']} #{o['track_id']} {o['confidence']:.2f}",
            (x, max(y-8,12)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    if alert:
        cv2.rectangle(frame, (0,0), (frame.shape[1], 50), (0,0,200), -1)
        cv2.putText(frame, "⚠ POACHER ALERT", (12,36),
            cv2.FONT_HERSHEY_DUPLEX, 1.1, (255,255,255), 2)
    return frame

def run(source):
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened(): raise RuntimeError(f"Cannot open: {source}")
    tracker = get_tracker(max_age=30, n_init=2)
    state = {"frame_id":0,"timestamp":None,"tracks":[],"poacher_alert":False,
             "animal_count":0,"human_count":0,"alert_log":[]}
    fid = 0
    print(f"[main] Running | source={source} | Press Ctrl+C to stop")
    while True:
        ret, frame = cap.read()
        if not ret: break
        fid += 1
        dets   = run_detection(frame)
        result = tracker.update(dets, frame=frame)
        tracks = result["tracks"]
        alert  = result["poacher_alert"]
        state.update({"frame_id":fid, "timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
            "tracks":tracks, "poacher_alert":alert,
            "animal_count": sum(1 for t in tracks if t["type"]=="animal"),
            "human_count":  sum(1 for t in tracks if t["type"]=="human")})
        if alert:
            state["alert_log"].append({"frame_id":fid,"timestamp":state["timestamp"],
                "animals":[t["class_name"] for t in tracks if t["type"]=="animal"],
                "humans":state["human_count"]})
            state["alert_log"] = state["alert_log"][-20:]
        _save(state)
        vis = _draw(frame.copy(), tracks, alert)
        os.makedirs("data", exist_ok=True)
        cv2.imwrite(OUTPUT_FRAME, vis)
        print(f"[frame {fid:05d}] animals={state['animal_count']} humans={state['human_count']} {'*** POACHER ALERT ***' if alert else 'ok'}")
    cap.release()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0", help="video file path or 0 for webcam")
    run(p.parse_args().source)
