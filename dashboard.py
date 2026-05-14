# ============================================================
# dashboard/dashboard.py
# ============================================================
# Wildlife Intrusion Monitoring Dashboard
# Multi-Drone Edition
#
# Backend pipeline (Terminal 1):
#   py -3.11 multi_drone.py --sources LionVideo.mp4 0 --feeds wildlife urban
#
# This dashboard (Terminal 2):
#   py -3.11 -m streamlit run dashboard.py
#
# Reads:  data/multi_drone_state.json   (written by MultiDroneEngine.tick())
#         data/drone_frames/<id>/latest.jpg  (written atomically by DroneWorker)
# ============================================================

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import streamlit as st
from PIL import Image

# ============================================================
# CONSTANTS  — must match multi_drone.py
# ============================================================

STATE_FILE = "data/multi_drone_state.json"
FRAME_DIR  = "data/drone_frames"

# Maps drone_id → display label driven by --feeds argument order
# drone_1 = first --sources entry, drone_2 = second, etc.
FEED_LABELS = {
    "drone_1": "🌿 Wildlife Feed",
    "drone_2": "🏙️  Urban Feed",
}

# BUG FIX: was 0.1 s (10 fps) hammering the CPU.
# 1.0 s matches the engine tick rate (~2/s) and is plenty for a dashboard.
AUTO_REFRESH_SECS = 1.0

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Wildlife Intrusion Monitoring System",
    page_icon="🦁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS — WILDLIFE FOREST THEME
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] { font-family:'Syne',sans-serif; }
.stApp { background-color:#081108; color:#d8f0c8; }

[data-testid="stSidebar"] {
    background:linear-gradient(180deg,#0d1a0d,#132813);
    border-right:1px solid #214221;
}
[data-testid="stSidebar"] * { color:#b8e8a0 !important; }

/* Header */
.main-header {
    background:linear-gradient(135deg,#102510,#1a3d1a,#102510);
    border:1px solid #2f5f2f; border-radius:18px;
    padding:24px; margin-bottom:24px;
    box-shadow:0 0 30px rgba(40,120,40,0.25);
}
.main-header h1 { font-size:2.3rem; color:#89e05d; margin:0; font-weight:800; }
.main-header p  { color:#71ba52; margin-top:8px;
    font-family:'Space Mono',monospace; font-size:0.78rem; letter-spacing:2px; }

/* Metric cards */
.metric-card {
    background:#0f1d0f; border:1px solid #2d5c2d;
    border-radius:14px; padding:18px; text-align:center;
}
.metric-value { font-size:2.4rem; font-weight:800; color:#8cf060;
    font-family:'Space Mono',monospace; }
.metric-label { color:#6db54f; margin-top:6px; font-size:0.7rem;
    letter-spacing:2px; text-transform:uppercase; font-family:'Space Mono',monospace; }

/* Alert pulse */
@keyframes redpulse {
    0%,100%{ border-color:#2d5c2d; box-shadow:none; }
    50%     { border-color:#ff4c4c; box-shadow:0 0 16px rgba(255,76,76,.5); }
}
.alert-active { animation:redpulse 1.2s infinite; }

/* Banners */
.poacher-banner {
    background:#1b0c0c; border:2px solid #ff4c4c; border-radius:10px;
    padding:14px 20px; text-align:center; animation:redpulse 1.2s infinite;
    font-family:'Space Mono',monospace; font-size:.88rem; color:#ffb3b3;
    margin-bottom:16px;
}
.safe-banner {
    background:#0f1f0f; border:1px solid #2f5f2f; border-radius:10px;
    padding:10px 20px; text-align:center;
    font-family:'Space Mono',monospace; font-size:.78rem; color:#c8ffc8;
    margin-bottom:16px;
}

/* Drone feed card */
.feed-card {
    border:1px solid #2b5b2b; border-radius:14px;
    background:#0c150c; padding:10px; margin-bottom:14px;
}
.feed-header {
    font-family:'Space Mono',monospace; font-size:.65rem;
    letter-spacing:3px; text-transform:uppercase; color:#3a7a3a;
    border-bottom:1px solid #1a381a; padding-bottom:7px; margin-bottom:10px;
}

/* Live dot */
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }
.dot-live { display:inline-block; width:8px; height:8px; border-radius:50%;
    background:#8cf060; margin-right:6px; animation:blink 1.2s infinite; }
.dot-wait { display:inline-block; width:8px; height:8px; border-radius:50%;
    background:#444; margin-right:6px; }

/* Track table */
.track-table { width:100%; border-collapse:collapse; }
.track-table th {
    font-family:'Space Mono',monospace; font-size:.6rem; color:#2a6a2a;
    text-transform:uppercase; letter-spacing:1px;
    border-bottom:1px solid #2f5f2f; padding:4px 6px; text-align:left;
}
.track-table td {
    font-family:'Space Mono',monospace; font-size:.68rem;
    padding:4px 6px; border-bottom:1px solid #0f1e0f;
}
.tr-animal { color:#8cf060; }
.tr-human  { color:#ff9999; }
.tr-multi  { color:#ffe066; }

/* Alert entry */
.alert-danger {
    background:#1b0c0c; border-left:4px solid #ff4c4c;
    border-radius:8px; padding:10px; margin-bottom:8px;
    color:#ffb3b3; font-family:'Space Mono',monospace; font-size:.68rem;
}
.alert-success {
    background:#0f1f0f; border-left:4px solid #59d659;
    border-radius:8px; padding:10px; margin-bottom:8px;
    color:#c8ffc8; font-family:'Space Mono',monospace; font-size:.72rem;
}

/* Log box */
.log-box {
    background:#091109; border:1px solid #214221;
    border-radius:10px; padding:12px; height:280px;
    overflow-y:auto; font-family:'Space Mono',monospace; font-size:.7rem;
}

/* Section title */
.sec {
    font-size:.6rem; font-family:'Space Mono',monospace;
    text-transform:uppercase; letter-spacing:3px; color:#3a7a3a;
    border-bottom:1px solid #1a381a; padding-bottom:6px; margin:14px 0 10px;
}

/* Buttons */
.stButton>button {
    background:linear-gradient(135deg,#2d612d,#3f813f);
    color:#e6ffe0; border-radius:10px; border:1px solid #65b565;
    font-family:'Space Mono',monospace; transition:.2s;
}
.stButton>button:hover { border-color:#92ff92; }

/* Waiting screen */
.waiting {
    text-align:center; padding:70px 20px;
    font-family:'Space Mono',monospace; color:#2a5a2a;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================

if "system_running" not in st.session_state:
    st.session_state.system_running = False

# ============================================================
# HELPERS
# ============================================================

def load_state() -> dict | None:
    """
    Load data/multi_drone_state.json written by MultiDroneEngine.tick().
    Returns None if not present or mid-write.
    """
    try:
        if not os.path.exists(STATE_FILE):
            return None
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def load_frame(drone_id: str) -> Image.Image | None:
    """
    Load the latest annotated frame for a drone.
    BUG FIX: .copy() releases the file handle immediately so the writer
    (DroneWorker) is never blocked on Windows file locking.
    Race condition on the write side is already fixed in multi_drone.py
    via os.replace() atomic rename.
    """
    path = Path(FRAME_DIR) / drone_id / "latest.jpg"
    if path.exists():
        try:
            return Image.open(path).copy()
        except Exception:
            pass
    return None


def feed_label(drone_id: str) -> str:
    return FEED_LABELS.get(drone_id, drone_id.upper().replace("_", " "))


def fmt_ts(iso: str) -> str:
    return iso[11:19] if len(iso) >= 19 else "—"


def track_css(track: dict) -> str:
    if len(track.get("confirmed_by", [])) > 1:
        return "tr-multi"
    return "tr-human" if track.get("type") == "human" else "tr-animal"


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## 🌿 System Configuration")
    st.markdown("---")

    st.markdown('<div class="sec">How to run</div>', unsafe_allow_html=True)
    st.markdown("""
**Terminal 1 — detection engine**
```
py -3.11 multi_drone.py \\
  --sources LionVideo.mp4 0 \\
  --feeds wildlife urban
```

**Terminal 2 — this dashboard**
```
py -3.11 -m streamlit run dashboard.py
```
    """)

    st.markdown("---")
    st.markdown('<div class="sec">Dashboard Settings</div>', unsafe_allow_html=True)

    auto_refresh = st.checkbox("Auto-refresh", value=True)

    # BUG FIX: was hardcoded 0.1s (10 fps). Now user-controlled, default 1s.
    refresh_rate = st.slider(
        "Refresh interval (s)",
        min_value=0.5,
        max_value=5.0,
        value=AUTO_REFRESH_SECS,
        step=0.5,
        help="Engine writes at ~2 ticks/s. Refreshing faster than 0.5s wastes CPU."
    )

    show_raw = st.checkbox("Show per-drone raw tracks", value=True)

    st.markdown("---")
    if st.button("🔄 Force Refresh"):
        st.rerun()

    if st.button("⏹ STOP SYSTEM"):
        st.session_state.system_running = False
        st.rerun()

# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>🦁 Wildlife Intrusion Monitoring System</h1>
    <p>YOLOv8 • DeepSORT Persistent IDs • Multi-Drone Coordination • Cross-Feed Deduplication</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# LOAD STATE
# ============================================================

state = load_state()

# ============================================================
# WAITING SCREEN
# ============================================================

if state is None:
    st.markdown("""
    <div class="waiting">
      <div style="font-size:3rem">🛸</div>
      <div style="font-size:.9rem;letter-spacing:2px;margin-top:16px;">
        WAITING FOR DRONE ENGINE…
      </div>
      <div style="font-size:.72rem;color:#1e421e;margin-top:10px;">
        Start <code>multi_drone.py</code> in Terminal 1 first.<br>
        This dashboard will populate automatically.
      </div>
    </div>
    """, unsafe_allow_html=True)
    # BUG FIX: only rerun if auto_refresh is on and system is supposed to be running
    if auto_refresh:
        time.sleep(refresh_rate)
        st.rerun()
    st.stop()

# ============================================================
# UNPACK STATE  (keys written by MultiDroneEngine.tick())
# ============================================================

drone_states   : dict = state.get("drone_states",        {})
deduped_tracks : list = state.get("deduplicated_tracks", [])
alert_log      : list = state.get("alert_log",           [])
poacher_alert  : bool = state.get("poacher_alert",       False)
n_animals      : int  = state.get("animal_count",        0)
n_humans       : int  = state.get("human_count",         0)
n_drones       : int  = state.get("drone_count",         0)
last_ts        : str  = fmt_ts(state.get("timestamp",    ""))
drone_ids      : list = sorted(drone_states.keys())

# ============================================================
# POACHER / SAFE BANNER
# ============================================================

if poacher_alert:
    st.markdown("""
    <div class="poacher-banner">
      ⚠️ &nbsp; POACHER / INTRUSION ALERT
      — Animals &amp; Humans confirmed simultaneously &nbsp; ⚠️
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="safe-banner">✅ &nbsp; All clear — no active intrusion</div>',
        unsafe_allow_html=True
    )

# ============================================================
# METRICS
# ============================================================

# NOTE ON COUNTS:
# These come from deduplicated_tracks, NOT raw detections.
# DeepSORT assigns each individual a persistent track_id.
# Same animal re-entering the feed keeps its id → count stays stable.
n_unique  = len(deduped_tracks)
n_multi   = sum(1 for t in deduped_tracks if len(t.get("confirmed_by", [])) > 1)
n_alerts  = len(alert_log)

m1, m2, m3, m4, m5, m6 = st.columns(6)
for col, val, lbl, extra in [
    (m1, n_drones,  "🛸 Active Drones",       ""),
    (m2, n_animals, "🐾 Animals in Frame",     ""),
    (m3, n_humans,  "👤 Humans in Frame",      "alert-active" if n_humans else ""),
    (m4, n_unique,  "🎯 Unique IDs",           ""),
    (m5, n_multi,   "⭐ Cross-Feed Confirmed", ""),
    (m6, n_alerts,  "🚨 Alert Events",         "alert-active" if poacher_alert else ""),
]:
    col.markdown(f"""
    <div class="metric-card {extra}">
      <div class="metric-value">{val}</div>
      <div class="metric-label">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown(
    f'<div style="font-family:Space Mono,monospace;font-size:.6rem;'
    f'color:#2a5a2a;text-align:right;margin-top:6px;">'
    f'Last engine tick: {last_ts} &nbsp;|&nbsp; ~2 ticks/s</div>',
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# DUAL LIVE FEEDS
# ============================================================

st.markdown('<div class="sec">📡 Live Drone Feeds</div>', unsafe_allow_html=True)

if not drone_ids:
    st.info("No drone feeds in state yet. Is multi_drone.py running?")
else:
    feed_cols = st.columns(len(drone_ids))

    for col, drone_id in zip(feed_cols, drone_ids):
        ds      = drone_states[drone_id]
        img     = load_frame(drone_id)       # atomic read, race-condition safe
        a_cnt   = ds.get("animal_count", 0)
        h_cnt   = ds.get("human_count",  0)
        frame_n = ds.get("frame_id",     0)
        ts      = fmt_ts(ds.get("timestamp", ""))
        d_alert = ds.get("poacher_alert", False)
        lbl     = feed_label(drone_id)
        dot     = "dot-live" if img else "dot-wait"

        with col:
            st.markdown(f"""
            <div class="feed-card {'alert-active' if d_alert else ''}">
              <div class="feed-header">
                <span class="{dot}"></span>
                {lbl} &nbsp;|&nbsp; frame #{frame_n:,} &nbsp;|&nbsp; {ts}
              </div>
            """, unsafe_allow_html=True)    

            if img:
                st.image(img, width="stretch")
            else:
                st.markdown(
                    '<div style="height:180px;display:flex;align-items:center;'
                    'justify-content:center;color:#1e421e;'
                    'font-family:Space Mono,monospace;font-size:.7rem;">'
                    'Waiting for first frame…</div>',
                    unsafe_allow_html=True
                )

            st.markdown(f"""
              <div style="display:flex;gap:16px;justify-content:center;
                   margin-top:8px;font-family:Space Mono,monospace;font-size:.7rem;">
                <span style="color:#8cf060">🐾 {a_cnt} animal{'s' if a_cnt!=1 else ''}</span>
                <span style="color:#ff9999">👤 {h_cnt} human{'s' if h_cnt!=1 else ''}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ============================================================
# DEDUPLICATED TARGETS  +  ALERTS  +  LOGS
# ============================================================

col_left, col_right = st.columns([1.7, 1])

# ── LEFT: deduped table + raw tracks ─────────────────────────────────────────
with col_left:

    st.markdown('<div class="sec">🎯 Deduplicated Target List</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-family:Space Mono,monospace;font-size:.6rem;color:#2a5a2a;'
        'margin-bottom:8px;">'
        'One row = one individual. DeepSORT track_id persists across frames — '
        'same animal re-entering keeps its ID, count stays stable. '
        '⭐ = confirmed by both drones.</div>',
        unsafe_allow_html=True
    )

    if deduped_tracks:
        tbl = """<table class="track-table">
          <tr>
            <th>Track ID</th><th>Class</th><th>Type</th>
            <th>Confidence</th><th>Drone(s)</th><th>Priority</th>
          </tr>"""
        for t in deduped_tracks:
            tid    = t.get("track_id",   "?")
            cls    = t.get("class_name", "unknown")
            typ    = t.get("type",       "other")
            conf   = t.get("confidence", 0.0)
            drones = t.get("confirmed_by", [t.get("drone_id", "?")])
            multi  = len(drones) > 1
            css    = track_css(t)
            icon   = "👤" if typ == "human" else "🐾"
            badge  = ("⭐ " + " + ".join(d.replace("drone_", "D") for d in drones)
                      if multi else drones[0].replace("drone_", "D"))
            score  = round(conf * (1.5 if multi else 1.0) * (1.2 if typ == "human" else 1.0), 3)

            tbl += f"""<tr class="{css}">
              <td>#{tid}</td><td>{cls}</td>
              <td>{icon} {typ}</td><td>{conf:.3f}</td>
              <td>{badge}</td><td>{score:.3f}</td>
            </tr>"""
        tbl += "</table>"
        st.markdown(tbl, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="font-family:Space Mono,monospace;font-size:.72rem;'
            'color:#2a5a2a;padding:20px 0;">No targets in current frame.</div>',
            unsafe_allow_html=True
        )

    # Per-drone raw tracks
    if show_raw:
        st.markdown('<div class="sec">📋 Per-Drone Raw Tracks</div>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:Space Mono,monospace;font-size:.6rem;color:#2a5a2a;'
            'margin-bottom:6px;">Raw DeepSORT output before cross-feed deduplication. '
            'Ghost/coasting tracks (conf=0) are filtered by tracker.py.</div>',
            unsafe_allow_html=True
        )

        for drone_id in drone_ids:
            ds     = drone_states.get(drone_id, {})
            tracks = ds.get("tracks", [])
            lbl    = feed_label(drone_id)

            with st.expander(f"{lbl}  —  {len(tracks)} confirmed track(s)", expanded=True):
                if tracks:
                    tbl = """<table class="track-table">
                      <tr><th>ID</th><th>Class</th><th>Type</th>
                          <th>Conf</th><th>Center (x,y)</th></tr>"""
                    for t in tracks:
                        typ = t.get("type", "other")
                        css = "tr-human" if typ == "human" else "tr-animal"
                        cx, cy = t.get("center", [0, 0])
                        tbl += f"""<tr class="{css}">
                          <td>#{t.get('track_id','?')}</td>
                          <td>{t.get('class_name','?')}</td>
                          <td>{'👤' if typ=='human' else '🐾'} {typ}</td>
                          <td>{t.get('confidence',0):.3f}</td>
                          <td>{cx:.0f}, {cy:.0f}</td>
                        </tr>"""
                    tbl += "</table>"
                    st.markdown(tbl, unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<span style="font-family:Space Mono,monospace;font-size:.7rem;'
                        'color:#2a5a2a">No confirmed tracks (conf > 0) this frame.</span>',
                        unsafe_allow_html=True
                    )

# ── RIGHT: alerts + species ───────────────────────────────────────────────────
with col_right:

    st.markdown('<div class="sec">🚨 Alerts</div>', unsafe_allow_html=True)

    if alert_log:
        for entry in reversed(alert_log[-10:]):
            animals_seen = ", ".join(entry.get("animals", [])) or "unknown"
            n_hum        = entry.get("humans", "?")
            ts_e         = fmt_ts(entry.get("timestamp", ""))
            drones_e     = ", ".join(entry.get("drones", []))
            st.markdown(f"""
            <div class="alert-danger">
              🚨 <strong>{ts_e}</strong><br>
              Animals : {animals_seen}<br>
              Humans  : {n_hum} detected<br>
              Drones  : {drones_e}
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="alert-success">✅ No active alerts</div>',
            unsafe_allow_html=True
        )

    # Detection log (latest tracks across all drones)
    st.markdown('<div class="sec">📋 Detection Log</div>', unsafe_allow_html=True)
    all_tracks = []
    for ds in drone_states.values():
        for t in ds.get("tracks", []):
            all_tracks.append(t)

    log_html = ""
    for t in reversed(all_tracks[-25:]):
        typ   = t.get("type", "other")
        color = "#ff9999" if typ == "human" else "#8cf060"
        log_html += (
            f'<div style="padding:4px 0;border-bottom:1px solid #173117;color:{color}">'
            f'{t.get("class_name","?")} | '
            f'ID #{t.get("track_id","?")} | '
            f'{t.get("confidence",0):.2f}'
            f'</div>'
        )

    st.markdown(
        f'<div class="log-box">{log_html if log_html else "<span style=color:#2a5a2a>Awaiting tracks…</span>"}</div>',
        unsafe_allow_html=True
    )

    # Species counter
    st.markdown('<div class="sec">🐾 Species Seen</div>', unsafe_allow_html=True)
    species: dict = defaultdict(int)
    for ds in drone_states.values():
        for t in ds.get("tracks", []):
            if t.get("type") == "animal":
                species[t.get("class_name", "unknown")] += 1

    if species:
        max_cnt = max(species.values())
        for sp, cnt in sorted(species.items(), key=lambda x: -x[1])[:10]:
            bar = int(cnt / max_cnt * 100)
            st.markdown(f"""
            <div style="padding:4px 0;border-bottom:1px solid #0f1e0f;">
              <div style="display:flex;justify-content:space-between;
                   font-family:Space Mono,monospace;font-size:.7rem;color:#5a9a5a;">
                <span>{sp}</span><span style="color:#8cf060">{cnt}</span>
              </div>
              <div style="height:3px;background:#0f2a0f;border-radius:2px;margin-top:3px;">
                <div style="width:{bar}%;height:3px;
                     background:linear-gradient(90deg,#3a8a3a,#8cf060);
                     border-radius:2px;"></div>
              </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="font-family:Space Mono,monospace;font-size:.7rem;color:#2a5a2a">'
            'No animals in current frame.</div>',
            unsafe_allow_html=True
        )

# ============================================================
# FOOTER
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f'<div style="font-family:Space Mono,monospace;font-size:.58rem;'
    f'color:#1a3a1a;text-align:center;">'
    f'Wildlife Monitoring System &nbsp;•&nbsp; '
    f'Refresh: {"ON @ " + str(refresh_rate) + "s" if auto_refresh else "OFF"}'
    f' &nbsp;•&nbsp; {STATE_FILE}</div>',
    unsafe_allow_html=True
)

# ============================================================
# AUTO REFRESH
# BUG FIX: was time.sleep(0.1) + st.rerun() — 10fps, CPU hammering.
# Now sleeps for refresh_rate (default 1.0s) and only reruns when
# auto_refresh is enabled, not unconditionally.
# ============================================================

if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()