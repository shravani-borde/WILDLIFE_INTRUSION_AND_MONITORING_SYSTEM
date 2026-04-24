"""
tracker.py
----------
DeepSORT-based object tracker for the aerial surveillance drone system.

Input  — each dict from run_detection():
    { "bbox":[x,y,w,h], "center":[cx,cy], "confidence":float,
      "class_id":int, "class_name":str, "type":str, "id":None }

Output — dict returned by update():
    {
        "tracks":        list[dict],   # confirmed tracked objects
        "poacher_alert": bool          # True when animals AND humans both confirmed
    }

BBOX FORMAT:
    deep_sort_realtime natively expects [left, top, w, h] == our [x, y, w, h].
    No xyxy conversion anywhere in this file.
"""

from deep_sort_realtime.deepsort_tracker import DeepSort
from animal_model_classes import ANIMAL_CLASSES
from human_model_classes  import HUMAN_CLASSES

# -- Class name -> type lookup sets ----------------------------------------
_ANIMAL_NAMES: set[str] = set(ANIMAL_CLASSES.values())
_HUMAN_NAMES:  set[str] = set(HUMAN_CLASSES.values())


def _resolve_type(class_name: str) -> str:
    if class_name in _ANIMAL_NAMES:
        return "animal"
    if class_name in _HUMAN_NAMES or class_name == "human":
        return "human"
    return "unknown"


class ObjectTracker:
    """
    Wraps DeepSORT for the drone surveillance pipeline.

    Tuning notes (webcam / indoor use):
        max_age=8       : drop stale tracks after 8 frames (~0.27s at 30fps)
                          was 30 — caused 0.00-confidence ghost tracks to persist
        n_init=3        : require 3 consecutive frames to confirm a track
                          was 2 — too easy at 30fps, caused duplicate confirmations
        max_cosine_dist : 0.3 stricter re-ID matching
                          was 0.4 — was matching different people to same track

    For real drone footage (slower movement, more occlusion) increase
    max_age back to 20-30 and n_init back to 2.
    """

    def __init__(
        self,
        max_age: int = 30,
        n_init: int = 3,
        max_cosine_distance: float = 0.25,
        nn_budget: int = 100,
    ):
        self._tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            nn_budget=nn_budget,
        )

        # Pre-populate type cache from ground-truth class maps
        self._class_to_type: dict[str, str] = {}
        for name in _ANIMAL_NAMES:
            self._class_to_type[name] = "animal"
        for name in _HUMAN_NAMES:
            self._class_to_type[name] = "human"
        self._class_to_type["human"] = "human"

    def update(self, detections: list[dict], frame=None) -> dict:
        """
        Run DeepSORT on one frame's detections.

        Parameters
        ----------
        detections : list[dict]   output of detector.run_detection(frame)
        frame      : np.ndarray   current BGR frame for appearance embeddings

        Returns
        -------
        dict  { "tracks": list[dict], "poacher_alert": bool }
        """

        # -- Build DeepSORT input -----------------------------------------
        # Format: List[ ( [left,top,w,h], confidence, class_name ) ]
        # bbox from detector is already [x,y,w,h] == [left,top,w,h] — no conversion
        raw_detections = []
        for det in detections:
            class_name = det["class_name"]
            self._class_to_type[class_name] = _resolve_type(class_name)
            raw_detections.append((det["bbox"], det["confidence"], class_name))

        # -- Run tracker --------------------------------------------------
        tracks = self._tracker.update_tracks(raw_detections, frame=frame)

        # -- Build output -------------------------------------------------
        tracked_objects: list[dict] = []
        has_animal = False
        has_human  = False

        for track in tracks:
            if not track.is_confirmed():
                continue

            # Skip coasting tracks with zero confidence
            # These are tracks kept alive by Kalman prediction only —
            # they appear as 0.00 confidence boxes on screen
            confidence = track.get_det_conf()
            if confidence is None or confidence == 0.0:
                continue

            track_id   = track.track_id
            class_name = track.get_det_class()

            ltwh = track.to_ltwh(orig=True)
            if ltwh is None:
                ltwh = track.to_ltwh(orig=False)

            x, y, w, h = (float(v) for v in ltwh)
            cx = round(x + w / 2, 2)
            cy = round(y + h / 2, 2)

            det_type = self._class_to_type.get(class_name, "unknown")

            if det_type == "animal":
                has_animal = True
            elif det_type == "human":
                has_human = True

            tracked_objects.append({
                "track_id":   track_id,
                "bbox":       [round(x,2), round(y,2), round(w,2), round(h,2)],
                "center":     [cx, cy],
                "class_name": class_name,
                "type":       det_type,
                "confidence": round(float(confidence), 4),
            })

        # Poacher alert only fires on tracks with live detections (conf > 0)
        poacher_alert = has_animal and has_human

        return {
            "tracks":        tracked_objects,
            "poacher_alert": poacher_alert,
        }


# -- Singleton helper ------------------------------------------------------

_default_tracker: ObjectTracker | None = None

def get_tracker(**kwargs) -> ObjectTracker:
    """Return module-level singleton. kwargs applied on first call only."""
    global _default_tracker
    if _default_tracker is None:
        _default_tracker = ObjectTracker(**kwargs)
    return _default_tracker
