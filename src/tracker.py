from deep_sort_realtime.deepsort_tracker import DeepSort
from src.animal_model_classes import ANIMAL_CLASSES
from src.human_model_classes import HUMAN_CLASSES

_ANIMAL_NAMES = set(ANIMAL_CLASSES.values())
_HUMAN_NAMES  = set(HUMAN_CLASSES.values())

def _resolve_type(class_name):
    if class_name.lower() in {n.lower() for n in _ANIMAL_NAMES}: return "animal"
    if class_name.lower() in {"human","person"}: return "human"
    return "unknown"

class ObjectTracker:
    def __init__(self, max_age=40, n_init=1, max_cosine_distance=0.4, nn_budget=100):
        self._tracker = DeepSort(max_age=max_age, n_init=n_init,
            max_cosine_distance=max_cosine_distance, nn_budget=nn_budget)
        self._type_map = {n.lower(): "animal" for n in _ANIMAL_NAMES}
        self._type_map.update({n.lower(): "human" for n in _HUMAN_NAMES})
        self._type_map["human"] = "human"
        self._type_map["person"] = "human"

    def update(self, detections, frame=None):
        raw = [(d["bbox"], d["confidence"], d["class_name"]) for d in detections]
        tracks = self._tracker.update_tracks(raw, frame=frame)
        objects, has_animal, has_human = [], False, False
        for t in tracks:
            if not t.is_confirmed(): continue
            conf = t.get_det_conf()
            if conf is None:
                conf = 0.0
            cls = t.get_det_class()

            # FIX: don't use `or` on numpy arrays — try orig first, fallback explicitly
            ltwh = t.to_ltwh(orig=True)
            if ltwh is None:
                ltwh = t.to_ltwh(orig=False)

            x, y, w, h = (float(v) for v in ltwh)
            det_type = self._type_map.get(cls.lower() if cls else "", "unknown")
            if det_type == "animal": has_animal = True
            elif det_type == "human": has_human = True
            objects.append({
                "track_id": t.track_id,
                "bbox": [round(x,2), round(y,2), round(w,2), round(h,2)],
                "center": [round(x+w/2,2), round(y+h/2,2)],
                "class_name": cls, "type": det_type,
                "confidence": round(float(conf), 4),
            })
        return {"tracks": objects, "poacher_alert": has_animal and has_human}

_default = None
def get_tracker(**kw):
    global _default
    if _default is None: _default = ObjectTracker(**kw)
    return _default