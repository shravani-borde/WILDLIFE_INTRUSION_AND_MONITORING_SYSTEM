import argparse, json, os, time, threading
import cv2
from src.detector import run_detection
from src.tracker import ObjectTracker

STATE_FILE = "data/multi_drone_state.json"
FRAME_DIR  = "data/drone_frames"

class DroneWorker:
    def __init__(self, drone_id, source, feed_label, shared, lock):
        self.drone_id=drone_id; self.source=source; self.feed_label=feed_label
        self.shared=shared; self.lock=lock; self.running=False
        self.tracker=ObjectTracker(max_age=30, n_init=1)
        self._thread=threading.Thread(target=self._loop, daemon=True)

    def start(self): self.running=True; self._thread.start()
    def stop(self): self.running=False

    def _loop(self):
        src = int(self.source) if str(self.source).isdigit() else self.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened(): print(f"[{self.drone_id}] Cannot open {self.source}"); return
        fdir = os.path.join(FRAME_DIR, self.drone_id); os.makedirs(fdir, exist_ok=True)
        fid = 0
        print(f"[{self.drone_id}] Started | source={self.source}")
        while self.running:
            ret, frame = cap.read()
            if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); continue
            fid += 1
            dets = run_detection(frame)
            result = self.tracker.update(dets, frame=frame)
            tracks = result["tracks"]; alert = result["poacher_alert"]
            vis = frame.copy()
            for t in tracks:
                x,y,w,h = [int(v) for v in t["bbox"]]
                color = (0,80,220) if t["type"]=="human" else (30,180,30)
                cv2.rectangle(vis,(x,y),(x+w,y+h),color,2)
                cv2.putText(vis,f"{t['class_name']} #{t['track_id']}",(x,max(y-6,12)),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,color,1)
            fp = os.path.join(fdir, "latest.jpg")
            success = cv2.imwrite(fp, vis)
            if success:
                print(f"[{self.drone_id}] Saved frame -> {fp}")
            else:
                print(f"[{self.drone_id}] ERROR saving frame")
            with self.lock:
                self.shared[self.drone_id] = {
                    "drone_id":self.drone_id,"feed_label":self.feed_label,
                    "frame_id":fid,"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "tracks":tracks,"poacher_alert":alert,"frame_path":fp,
                    "animal_count":sum(1 for t in tracks if t["type"]=="animal"),
                    "human_count": sum(1 for t in tracks if t["type"]=="human"),
                }
        cap.release()

def run_engine(shared, lock, alert_log):
    while True:
        time.sleep(0.5)
        with lock: states = dict(shared)
        if not states: continue
        all_events = []
        for did, ds in states.items():
            for t in ds["tracks"]: all_events.append({**t, "drone_id":did})
        deduped = []; used = set()
        for i, ev in enumerate(all_events):
            if i in used: continue
            grp = [ev]; cx1,cy1 = ev["center"]
            for j, oth in enumerate(all_events):
                if j<=i or j in used or oth["class_name"]!=ev["class_name"]: continue
                cx2,cy2 = oth["center"]
                if ((cx1-cx2)**2+(cy1-cy2)**2)**0.5 < 80: grp.append(oth); used.add(j)
            used.add(i)
            best = max(grp, key=lambda e:e["confidence"])
            best["confirmed_by"] = list({g["drone_id"] for g in grp})
            deduped.append(best)
        has_a = any(e["type"]=="animal" for e in deduped)
        has_h = any(e["type"]=="human"  for e in deduped)
        alert = has_a and has_h
        if alert:
            alert_log.append({"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
                "animals":[e["class_name"] for e in deduped if e["type"]=="animal"],
                "humans":sum(1 for e in deduped if e["type"]=="human"),
                "drones":list(states.keys())})
            del alert_log[:-30]
        os.makedirs("data", exist_ok=True)
        with open(STATE_FILE,"w") as f:
            json.dump({"timestamp":time.strftime("%Y-%m-%dT%H:%M:%S"),
                "drone_count":len(states),"drone_states":states,
                "deduplicated_tracks":deduped,
                "animal_count":sum(1 for e in deduped if e["type"]=="animal"),
                "human_count": sum(1 for e in deduped if e["type"]=="human"),
                "poacher_alert":alert,"alert_log":alert_log},f,indent=2)
        print(f"[engine] drones={len(states)} animals={sum(1 for e in deduped if e['type']=='animal')} humans={sum(1 for e in deduped if e['type']=='human')} {'*** POACHER ALERT ***' if alert else 'ok'}")

def run(sources, feeds):
    shared={}; lock=threading.Lock(); alert_log=[]
    while len(feeds)<len(sources): feeds.append(f"drone_{len(feeds)+1}")
    workers=[DroneWorker(f"drone_{i+1}",src,feeds[i],shared,lock) for i,src in enumerate(sources)]
    for w in workers: w.start()
    t=threading.Thread(target=run_engine,args=(shared,lock,alert_log),daemon=True); t.start()
    print(f"[main] {len(workers)} drones running | Ctrl+C to stop")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping..."); [w.stop() for w in workers]

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", default=["0"])
    p.add_argument("--feeds",   nargs="+", default=[])
    a = p.parse_args(); run(a.sources, a.feeds)
