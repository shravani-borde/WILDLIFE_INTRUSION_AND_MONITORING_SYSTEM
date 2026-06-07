"""
dashboard/dashboard.py — Wildlife Monitoring Dashboard v4
Run: streamlit run dashboard/dashboard.py
"""
import json, os, time, base64
from collections import defaultdict, deque
from pathlib import Path
import streamlit as st

STATE_FILE = "data/multi_drone_state.json"
FRAME_DIR  = "data/drone_frames"

st.set_page_config(
    page_title="AI Wildlife Monitoring System",
    page_icon="🦁", layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
html,body,[class*="css"],.stApp{
  font-family:'Inter',sans-serif!important;
  background:#0B1120!important;
  color:#E2E8F0!important;
}
section[data-testid="stSidebar"]{display:none}
.block-container{padding:0!important;max-width:100%!important}
header[data-testid="stHeader"]{display:none}
div[data-testid="stVerticalBlock"]>div{padding:0!important}
div[data-testid="stImage"] img{border-radius:0!important;width:100%!important;display:block}

/* ── TOP NAV ── */
.topnav{
  display:flex;justify-content:space-between;align-items:center;
  padding:12px 24px;
  background:#0D1526;
  border-bottom:1px solid #1E2D4A;
}
.brand{display:flex;align-items:center;gap:14px}
.brand-icon{
  width:48px;height:48px;border-radius:10px;
  background:linear-gradient(135deg,#1B3A6B,#2563EB);
  display:flex;align-items:center;justify-content:center;font-size:1.4rem;
}
.brand-title{font-size:1.1rem;font-weight:700;color:#F1F5F9;line-height:1.2}
.brand-sub{font-size:.6rem;color:#3B82F6;letter-spacing:3px;text-transform:uppercase;margin-top:2px}
.nav-right{display:flex;align-items:center;gap:12px}
.nav-pill{
  display:flex;align-items:center;gap:7px;
  background:#111D35;border:1px solid #1E2D4A;border-radius:8px;
  padding:7px 14px;font-size:.72rem;color:#94A3B8;
}
.nav-pill .val{color:#F1F5F9;font-weight:600}
.op-dot{width:8px;height:8px;border-radius:50%;background:#22C55E;box-shadow:0 0 8px #22C55E;animation:blink 2s infinite}
.op-dot.red{background:#EF4444;box-shadow:0 0 8px #EF4444;animation:blink .8s infinite}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.4}}

/* ── KPI ROW ── */
.kpi-row{
  display:grid;grid-template-columns:repeat(6,1fr);gap:1px;
  background:#1E2D4A;border-bottom:1px solid #1E2D4A;
}
.kpi-card{
  background:#0D1526;padding:16px 20px;
  display:flex;align-items:center;gap:14px;
}
.kpi-icon{
  width:44px;height:44px;border-radius:10px;
  display:flex;align-items:center;justify-content:center;
  font-size:1.3rem;flex-shrink:0;
}
.kpi-icon.blue{background:rgba(37,99,235,.15);border:1px solid rgba(37,99,235,.3)}
.kpi-icon.green{background:rgba(34,197,94,.15);border:1px solid rgba(34,197,94,.3)}
.kpi-icon.red{background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.3)}
.kpi-icon.orange{background:rgba(249,115,22,.15);border:1px solid rgba(249,115,22,.3)}
.kpi-icon.purple{background:rgba(139,92,246,.15);border:1px solid rgba(139,92,246,.3)}
.kpi-icon.cyan{background:rgba(6,182,212,.15);border:1px solid rgba(6,182,212,.3)}
.kpi-label{font-size:.6rem;color:#64748B;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:4px}
.kpi-value{font-size:1.8rem;font-weight:700;color:#F1F5F9;line-height:1}
.kpi-sub{font-size:.62rem;margin-top:3px}
.kpi-sub.green{color:#22C55E}
.kpi-sub.red{color:#EF4444}
.kpi-sub.blue{color:#3B82F6}
.kpi-sub.orange{color:#F97316}

/* ── SECTION LABEL ── */
.sec-label{
  display:flex;align-items:center;gap:8px;
  font-size:.65rem;font-weight:600;color:#94A3B8;
  text-transform:uppercase;letter-spacing:2px;
  padding:10px 16px 8px;
  border-bottom:1px solid #1E2D4A;
}
.sec-label .live-dot{
  width:7px;height:7px;border-radius:50%;
  background:#22C55E;box-shadow:0 0 5px #22C55E;
  animation:blink 1.5s infinite;
}
.sec-label-right{margin-left:auto;font-size:.58rem;color:#3B82F6;cursor:pointer}

/* ── FEED ── */
.feed-wrap{
    background:#0D1526;
    border:1px solid #1E2D4A;
    border-radius:0;
}

.feed-topbar{
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:8px 14px;
    background:#111D35;
    border-bottom:1px solid #1E2D4A;
}

.feed-title{
    display:flex;
    align-items:center;
    gap:7px;
    font-size:.72rem;
    font-weight:600;
    color:#F1F5F9;
}

.feed-title .live-chip{
    background:#22C55E;
    color:#000;
    font-size:.5rem;
    font-weight:700;
    padding:2px 6px;
    border-radius:3px;
    letter-spacing:1px;
}

.feed-title .live-chip.alert{
    background:#EF4444;
    color:#fff;
    animation:blink .8s infinite;
}

.feed-meta-right{
    display:flex;
    align-items:center;
    gap:12px;
    font-size:.6rem;
    color:#64748B;
}

.feed-meta-right .mv{
    color:#94A3B8;
}

.feed-meta-right .sig{
    display:flex;
    align-items:center;
    gap:3px;
    color:#22C55E;
}

/* FIXED VIDEO AREA */
.feed-imgbox{
    position:relative;
    background:#060D18;
    overflow:hidden;
    width:100%;
    padding:0;
    margin:0;
    line-height:0;
}

/* MAKE IMAGE FILL CARD */
.feed-imgbox img{
    width:100%;
    height:auto;
    display:block;
    object-fit:cover;
}

/* STREAMLIT IMAGE */
.feed-imgbox > div{
    width:100%;
}

.feed-imgbox > div img{
    width:100% !important;
    height:auto !important;
    object-fit:cover;
    display:block;
}

.feed-overlay-tl{
    position:absolute;
    top:10px;
    left:10px;
    z-index:10;
}

.feed-overlay-tr{
    position:absolute;
    top:10px;
    right:10px;
    z-index:10;
}

.intrusion-badge{
    background:#EF4444;
    color:#fff;
    font-size:.52rem;
    font-weight:700;
    letter-spacing:1.5px;
    padding:3px 8px;
    border-radius:4px;
    animation:blink .7s infinite;
}

.rec-badge{
    display:flex;
    align-items:center;
    gap:4px;
    font-size:.58rem;
    color:#EF4444;
    font-family:'JetBrains Mono', monospace;
}

.rec-dot{
    width:6px;
    height:6px;
    border-radius:50%;
    background:#EF4444;
    animation:blink 1s infinite;
}

.no-signal{
    min-height:500px;
    display:flex;
    flex-direction:column;
    justify-content:center;
    align-items:center;
    color:#1E3A5F;
}

.no-signal .ns-icon{
    font-size:2.5rem;
    opacity:.3;
}

.no-signal .ns-txt{
    font-size:.6rem;
    letter-spacing:3px;
    margin-top:8px;
    text-transform:uppercase;
}

.feed-bottombar{
    display:flex;
    gap:20px;
    padding:8px 14px;
    background:#0A1628;
    border-top:1px solid #1E2D4A;
    font-size:.65rem;
    color:#64748B;
}

.feed-bottombar .fs{
    display:flex;
    align-items:center;
    gap:5px;
}

.feed-bottombar .fv{
    color:#94A3B8;
    font-weight:600;
}

.feed-bottombar .fv.red{
    color:#EF4444;
}
/* ── ALERT COMMAND CENTER ── */
.acc-wrap{background:#0D1526;border:1px solid #1E2D4A;height:100%}
.alert-card{
  margin:8px;padding:10px 12px;border-radius:8px;
  border:1px solid;cursor:pointer;transition:all .2s;
}
.alert-card.high{
  background:linear-gradient(135deg,rgba(127,29,29,.5),rgba(69,10,10,.8));
  border-color:rgba(239,68,68,.3);
}
.alert-card.medium{
  background:rgba(120,53,15,.3);border-color:rgba(249,115,22,.25);
}
.alert-card.info{
  background:rgba(30,58,138,.2);border-color:rgba(59,130,246,.2);
}
.alert-card-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:6px}
.alert-title{font-size:.7rem;font-weight:600}
.alert-title.red{color:#FCA5A5}
.alert-title.orange{color:#FED7AA}
.alert-title.blue{color:#BFDBFE}
.alert-time{font-size:.58rem;color:#475569;font-family:'JetBrains Mono',monospace}
.alert-desc{font-size:.65rem;color:#94A3B8;margin-bottom:6px}
.alert-meta{display:flex;align-items:center;gap:10px;font-size:.58rem;color:#64748B}
.alert-badge-pill{
  padding:2px 8px;border-radius:3px;font-size:.55rem;font-weight:700;letter-spacing:1px;
}
.alert-badge-pill.high{background:#7F1D1D;color:#FCA5A5}
.alert-badge-pill.medium{background:#7C2D12;color:#FED7AA}
.alert-badge-pill.info{background:#1E3A8A;color:#BFDBFE}
.alert-thumb{
  width:50px;height:36px;border-radius:4px;
  background:#1E3A8A;display:flex;align-items:center;justify-content:center;
  font-size:1rem;flex-shrink:0;
}

/* ── DETECTIONS TABLE ── */
.det-wrap{padding:0 0 8px;overflow-x:auto}
.dt{width:100%;border-collapse:collapse;font-size:.66rem}
.dt th{
  font-size:.56rem;color:#475569;text-transform:uppercase;
  letter-spacing:1.5px;padding:8px 12px;
  border-bottom:1px solid #1E2D4A;text-align:left;
  background:#0A1628;white-space:nowrap;
}
.dt td{
  padding:8px 12px;border-bottom:1px solid #111D35;
  color:#94A3B8;white-space:nowrap;
}
.dt tr:hover td{background:rgba(30,45,74,.4)}
.dt td:first-child{color:#F1F5F9;font-weight:600;font-family:'JetBrains Mono',monospace}
.type-pill{
  display:inline-block;padding:2px 8px;border-radius:4px;
  font-size:.56rem;font-weight:700;letter-spacing:1px;
}
.type-pill.animal{background:rgba(34,197,94,.15);color:#86EFAC;border:1px solid rgba(34,197,94,.2)}
.type-pill.human{background:rgba(239,68,68,.15);color:#FCA5A5;border:1px solid rgba(239,68,68,.2)}
.status-pill{
  display:inline-block;padding:2px 8px;border-radius:4px;
  font-size:.56rem;font-weight:600;
}
.status-pill.active{background:rgba(34,197,94,.1);color:#86EFAC}
.status-pill.alert{background:rgba(239,68,68,.2);color:#FCA5A5;animation:blink 1s infinite}
.conf-wrap{display:flex;align-items:center;gap:6px}
.conf-bar-bg{width:50px;height:3px;background:#1E2D4A;border-radius:2px;overflow:hidden}
.conf-bar-fill{height:100%;border-radius:2px;background:linear-gradient(90deg,#2563EB,#60A5FA)}

/* ── ANALYTICS ── */
.analytics-wrap{padding:0 16px 16px}
.sp-row{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid #111D35;font-size:.65rem}
.sp-name{width:80px;color:#CBD5E1}
.sp-track{flex:1;height:5px;background:#1E2D4A;border-radius:3px;overflow:hidden}
.sp-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,#1D4ED8,#3B82F6)}
.sp-cnt{width:22px;text-align:right;color:#475569}

/* ── FLEET ── */
.fleet-wrap{padding:0 12px 12px}
.drone-card{
  background:#111D35;border:1px solid #1E2D4A;border-radius:8px;
  padding:12px;margin-bottom:8px;
}
.drone-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px}
.drone-name{font-size:.75rem;font-weight:700;color:#F1F5F9;display:flex;align-items:center;gap:8px}
.drone-name .icon{font-size:1.1rem}
.online-badge{
  display:flex;align-items:center;gap:4px;
  font-size:.58rem;color:#22C55E;
  font-family:'JetBrains Mono',monospace;
}
.online-dot{width:5px;height:5px;border-radius:50%;background:#22C55E;box-shadow:0 0 4px #22C55E}
.drone-grid{display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:.6rem;color:#64748B}
.drone-grid span{color:#94A3B8}
.drone-status-line{margin-top:6px;font-size:.58rem;font-family:'JetBrains Mono',monospace}

/* ── TIMELINE ── */
.timeline-wrap{padding:0 12px 12px;max-height:300px;overflow-y:auto}
.tl-item{display:flex;gap:10px;padding:6px 0;border-bottom:1px solid #111D35;align-items:flex-start}
.tl-left{display:flex;flex-direction:column;align-items:center;gap:2px;flex-shrink:0}
.tl-icon{
  width:28px;height:28px;border-radius:50%;
  display:flex;align-items:center;justify-content:center;font-size:.8rem;
}
.tl-icon.red{background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.2)}
.tl-icon.orange{background:rgba(249,115,22,.15);border:1px solid rgba(249,115,22,.2)}
.tl-icon.blue{background:rgba(59,130,246,.15);border:1px solid rgba(59,130,246,.2)}
.tl-icon.green{background:rgba(34,197,94,.15);border:1px solid rgba(34,197,94,.2)}
.tl-time{font-size:.55rem;color:#3B82F6;font-family:'JetBrains Mono',monospace;white-space:nowrap}
.tl-content{flex:1}
.tl-msg{font-size:.65rem;color:#CBD5E1;line-height:1.4}
.tl-sub{font-size:.58rem;color:#475569;margin-top:1px}

/* ── TREND CHART ── */
.chart-placeholder{
  background:#0A1628;border:1px solid #1E2D4A;border-radius:6px;
  padding:16px;margin:0 16px 12px;
  font-family:'JetBrains Mono',monospace;font-size:.6rem;color:#475569;
}
.trend-area{position:relative;height:80px;margin-top:8px}
.trend-label{font-size:.58rem;color:#475569;display:flex;justify-content:space-between;margin-top:4px}

/* scrollbar */
::-webkit-scrollbar{width:4px;height:4px}
::-webkit-scrollbar-track{background:#0B1120}
::-webkit-scrollbar-thumb{background:#1E2D4A;border-radius:2px}

/* banner */
.alert-banner{
  padding:10px 24px;text-align:center;
  font-size:.72rem;letter-spacing:2px;font-weight:600;
}
.alert-banner.danger{
  background:linear-gradient(90deg,#450A0A,#7F1D1D,#450A0A);
  color:#FCA5A5;border-bottom:1px solid rgba(239,68,68,.4);
  animation:blink 1s infinite;
}
.alert-banner.safe{
  background:#0A1628;color:#22C55E;
  border-bottom:1px solid #1E2D4A;font-weight:400;
}
</style>
""", unsafe_allow_html=True)

# ── helpers ──────────────────────────────────────────────────
def load_state():
    try:
        if not os.path.exists(STATE_FILE): return None
        with open(STATE_FILE) as f: return json.load(f)
    except: return None

def load_frame_b64(did):
    p = Path(FRAME_DIR) / did / "latest.jpg"
    if p.exists():
        try:
            with open(p,"rb") as f: return base64.b64encode(f.read()).decode()
        except: pass
    return None

def fmt_ts(iso):
    return iso[11:19] if len(iso)>=19 else time.strftime("%H:%M:%S")

# ── session state ─────────────────────────────────────────────
if "timeline" not in st.session_state:
    st.session_state.timeline = deque([
        {"time":time.strftime("%H:%M:%S"),"msg":"System initialised","sub":"All zones operational","level":"blue","icon":"🛡️"},
    ], maxlen=60)
if "species_hist" not in st.session_state:
    st.session_state.species_hist = defaultdict(int)
if "start_time" not in st.session_state:
    st.session_state.start_time = time.time()
if "animal_trend" not in st.session_state:
    st.session_state.animal_trend = deque([0]*20, maxlen=20)
if "human_trend" not in st.session_state:
    st.session_state.human_trend = deque([0]*20, maxlen=20)

# ── load data ─────────────────────────────────────────────────
state = load_state()
if state is None:
    drone_states={}; deduped_tracks=[]; alert_log=[]; poacher_alert=False
    n_animals=n_humans=n_drones=0; last_ts=time.strftime("%H:%M:%S")
else:
    drone_states   = state.get("drone_states",{})
    deduped_tracks = state.get("deduplicated_tracks",[])
    alert_log      = state.get("alert_log",[])
    poacher_alert  = state.get("poacher_alert",False)
    n_animals      = state.get("animal_count",0)
    n_humans       = state.get("human_count",0)
    n_drones       = state.get("drone_count",0)
    last_ts        = fmt_ts(state.get("timestamp",""))
    st.session_state.animal_trend.append(n_animals)
    st.session_state.human_trend.append(n_humans)
    if poacher_alert and (not st.session_state.timeline or "INTRUSION" not in st.session_state.timeline[-1]["msg"]):
        animals_near = [e["class_name"] for e in deduped_tracks if e.get("type")=="animal"]
        st.session_state.timeline.append({"time":time.strftime("%H:%M:%S"),
            "msg":f"INTRUSION ALERT — Human detected near {', '.join(animals_near) or 'wildlife'}",
            "sub":f"Drones: {', '.join(drone_states.keys())}","level":"red","icon":"🚨"})
    elif n_animals>0:
        last = list(st.session_state.timeline)[-1] if st.session_state.timeline else {}
        if "detected" not in last.get("msg",""):
            sp = [e["class_name"] for e in deduped_tracks if e.get("type")=="animal"]
            if sp:
                st.session_state.timeline.append({"time":time.strftime("%H:%M:%S"),
                    "msg":f"{sp[0]} detected","sub":f"Zone {'B' if any('2' in str(t.get('drone_id','')) for t in deduped_tracks) else 'A'}",
                    "level":"green","icon":"🐾"})
    for t in deduped_tracks:
        if t.get("type")=="animal":
            st.session_state.species_hist[t.get("class_name","?")] += 1

drone_ids = sorted(drone_states.keys())
uptime = int(time.time() - st.session_state.start_time)
uptime_str = f"{uptime//3600:02d}:{(uptime%3600)//60:02d}:{uptime%60:02d}"
unique_tracks = len(deduped_tracks)

# ── TOP NAV ───────────────────────────────────────────────────
st.markdown(f"""
<div class="topnav">
  <div class="brand">
    <div class="brand-icon">🦁</div>
    <div>
      <div class="brand-title">AI-Based Multi-Drone Wildlife Monitoring<br>& Intrusion Detection System</div>
      <div class="brand-sub">YOLOv8 &nbsp;•&nbsp; DeepSORT &nbsp;•&nbsp; Real-Time Wildlife Analytics</div>
    </div>
  </div>
  <div class="nav-right">
    <div class="nav-pill">
      <div class="op-dot {'red' if poacher_alert else ''}"></div>
      <span class="val" style="color:{'#EF4444' if poacher_alert else '#22C55E'}">{'ALERT' if poacher_alert else 'OPERATIONAL'}</span>
    </div>
    <div class="nav-pill">📅 <span class="val">{time.strftime("%d %b %Y")}</span></div>
    <div class="nav-pill">🕐 <span class="val">{time.strftime("%H:%M:%S")}</span></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPI ROW ───────────────────────────────────────────────────
st.markdown(f"""
<div class="kpi-row">
  <div class="kpi-card">
    <div class="kpi-icon blue">🚁</div>
    <div>
      <div class="kpi-label">Active Drones</div>
      <div class="kpi-value">{n_drones}</div>
      <div class="kpi-sub green">● Online</div>
    </div>
  </div>
  <div class="kpi-card">
    <div class="kpi-icon green">🐾</div>
    <div>
      <div class="kpi-label">Animals Detected</div>
      <div class="kpi-value">{n_animals}</div>
      <div class="kpi-sub blue">Live count</div>
    </div>
  </div>
  <div class="kpi-card">
    <div class="kpi-icon {'red' if n_humans else 'orange'}">👤</div>
    <div>
      <div class="kpi-label">Humans Detected</div>
      <div class="kpi-value">{n_humans}</div>
      <div class="kpi-sub {'red' if n_humans else 'blue'}">{'⚠ In protected zone' if n_humans else 'No humans'}</div>
    </div>
  </div>
  <div class="kpi-card">
    <div class="kpi-icon {'red' if alert_log else 'orange'}">⚠️</div>
    <div>
      <div class="kpi-label">Poacher Alerts</div>
      <div class="kpi-value">{len(alert_log)}</div>
      <div class="kpi-sub {'red' if poacher_alert else 'blue'}">{'Active' if poacher_alert else 'Total logged'}</div>
    </div>
  </div>
  <div class="kpi-card">
    <div class="kpi-icon purple">🎯</div>
    <div>
      <div class="kpi-label">Unique Tracks</div>
      <div class="kpi-value">{unique_tracks}</div>
      <div class="kpi-sub blue">Total Active</div>
    </div>
  </div>
  <div class="kpi-card">
    <div class="kpi-icon cyan">⏱️</div>
    <div>
      <div class="kpi-label">System Uptime</div>
      <div class="kpi-value" style="font-size:1.3rem">{uptime_str}</div>
      <div class="kpi-sub green">Running</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── ALERT BANNER ─────────────────────────────────────────────
if poacher_alert:
    near = [e["class_name"] for e in deduped_tracks if e.get("type")=="animal"]
    st.markdown(f'<div class="alert-banner danger">🚨 &nbsp; POACHER ALERT — HUMAN DETECTED NEAR WILDLIFE &nbsp;|&nbsp; Species: {", ".join(near) or "Unknown"} &nbsp;|&nbsp; IMMEDIATE RESPONSE REQUIRED &nbsp; 🚨</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="alert-banner safe">✦ &nbsp; ALL ZONES SECURE — NO INTRUSION DETECTED &nbsp; ✦</div>', unsafe_allow_html=True)

# ── LIVE FEEDS + ALERT CENTER ─────────────────────────────────
f1col, f2col, acol = st.columns([1, 1, 0.72])

def render_feed(col, idx, default_label):
    did    = drone_ids[idx] if len(drone_ids) > idx else None
    ds     = drone_states.get(did,{}) if did else {}
    b64    = load_frame_b64(did) if did else None
    a_cnt  = ds.get("animal_count",0)
    h_cnt  = ds.get("human_count",0)
    fid    = ds.get("frame_id",0)
    d_alrt = ds.get("poacher_alert",False)
    label  = ds.get("feed_label", default_label) if did else default_label
    chip   = f'<span class="live-chip {"alert" if d_alrt else ""}">{"⚠ ALERT" if d_alrt else "● LIVE"}</span>'

    with col:
        st.markdown(f"""
        <div style="border:1px solid #1E2D4A;border-bottom:none">
          <div class="sec-label">
            <div class="live-dot"></div>
            LIVE FEED — DRONE {idx+1}
          </div>
          <div class="feed-topbar">
            <div class="feed-title">{chip} &nbsp; {label.upper()}</div>
            <div class="feed-meta-right">
              <span>FPS <span class="mv">{23+idx:.1f}</span></span>
              <span>FRAME <span class="mv">#{fid:,}</span></span>
              <span class="sig">▌▌▌ Strong</span>
            </div>
          </div>
          <div class="feed-imgbox">
        """, unsafe_allow_html=True)

        if b64:
            badge = '<div class="feed-overlay-tl"><div class="intrusion-badge">⚠ INTRUSION</div></div>' if d_alrt else ''
            st.markdown(f"""
            {badge}
            <div class="feed-overlay-tr"><div class="rec-badge"><div class="rec-dot"></div>REC</div></div>
            <img src="data:image/jpeg;base64,{b64}" style="width:100%;display:block"/>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="no-signal" style="padding:60px 0">
              <div class="ns-icon">📡</div>
              <div class="ns-txt">Awaiting signal</div>
              <div class="ns-txt" style="font-size:.5rem;margin-top:4px;opacity:.5">Start multi_drone.py</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
          </div>
          <div class="feed-bottombar">
            <div class="fs">🐾 Animals: <span class="fv">{a_cnt}</span></div>
            <div class="fs">👤 Humans: <span class="fv {'red' if h_cnt else ''}">{h_cnt}</span></div>
            <div class="fs">📍 Zone {'A' if idx==0 else 'B'}</div>
          </div>
        </div>""", unsafe_allow_html=True)

render_feed(f1col, 0, "Wildlife Feed")
render_feed(f2col, 1, "Secondary Feed")

with acol:
    st.markdown('<div style="border:1px solid #1E2D4A;border-bottom:none">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label"><div class="live-dot"></div>ALERT COMMAND CENTER<span class="sec-label-right">View All</span></div>', unsafe_allow_html=True)

    if alert_log:
        for entry in reversed(alert_log[-2:]):
            ts      = fmt_ts(entry.get("timestamp",""))
            animals = ", ".join(entry.get("animals",[]))
            n_hum   = entry.get("humans","?")
            drones  = entry.get("drones",[])
            zone    = "Zone B" if drones and "2" in str(drones[-1]) else "Zone A"
            st.markdown(f"""
            <div class="alert-card high">
              <div class="alert-card-top">
                <div class="alert-title red">⚠ HIGH RISK — POACHER ALERT</div>
                <div class="alert-time">{ts}</div>
              </div>
              <div class="alert-desc">Human detected near {animals or "wildlife"}</div>
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div class="alert-meta">
                  <span>🚁 {(drones[-1] if drones else "Drone").replace("_"," ").title()}</span>
                  <span>📍 {zone}</span>
                </div>
                <span class="alert-badge-pill high">HIGH</span>
              </div>
            </div>""", unsafe_allow_html=True)

    if n_animals > 0 and not poacher_alert:
        sp_list = [e["class_name"] for e in deduped_tracks if e.get("type")=="animal"]
        st.markdown(f"""
        <div class="alert-card medium">
          <div class="alert-card-top">
            <div class="alert-title orange">⚡ MEDIUM RISK</div>
            <div class="alert-time">{last_ts}</div>
          </div>
          <div class="alert-desc">Multiple animals detected — {", ".join(set(sp_list[:3])) or "wildlife"}</div>
          <div style="display:flex;justify-content:space-between;align-items:center">
            <div class="alert-meta"><span>🚁 Drone 1</span><span>📍 Zone A</span></div>
            <span class="alert-badge-pill medium">MEDIUM</span>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="alert-card info">
      <div class="alert-card-top">
        <div class="alert-title blue">🛡️ SYSTEM MESSAGE</div>
        <div class="alert-time">{time.strftime("%H:%M:%S")}</div>
      </div>
      <div class="alert-desc">{'Monitoring active — all drones online' if n_drones else 'Waiting for drone engine'}</div>
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div class="alert-meta"><span>⚙ System</span></div>
        <span class="alert-badge-pill info">INFO</span>
      </div>
    </div>
    </div>""", unsafe_allow_html=True)

# ── BOTTOM ROW ───────────────────────────────────────────────
det_col, analytics_col, fleet_col, timeline_col = st.columns([1.2, 0.9, 0.7, 0.7])

with det_col:
    st.markdown('<div style="border:1px solid #1E2D4A">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label"><div class="live-dot"></div>ACTIVE DETECTIONS<span class="sec-label-right">View All</span></div>', unsafe_allow_html=True)
    if deduped_tracks:
        rows = ""
        for t in deduped_tracks[:8]:
            typ   = t.get("type","?")
            cls   = t.get("class_name","?")
            conf  = t.get("confidence",0)
            tid   = t.get("track_id","?")
            drones= t.get("confirmed_by",[t.get("drone_id","?")])
            is_al = typ=="human" and n_animals>0
            drn   = (drones[0] if drones else "drone_1").replace("_"," ").title()
            zone  = "Zone B" if drones and "2" in str(drones[0]) else "Zone A"
            bw    = int(conf*50)
            rows += f"""<tr>
              <td>#{tid}</td><td>{cls}</td>
              <td><span class="type-pill {typ}">{typ.upper()}</span></td>
              <td><div class="conf-wrap"><div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{bw}px"></div></div><span>{conf:.2f}</span></div></td>
              <td>{drn}</td><td>{zone}</td>
              <td><span class="status-pill {'alert' if is_al else 'active'}">{'Alert' if is_al else 'Active'}</span></td>
            </tr>"""
        st.markdown(f"""
        <div class="det-wrap">
          <table class="dt">
            <thead><tr><th>TRACK ID</th><th>SPECIES</th><th>TYPE</th><th>CONFIDENCE</th><th>DRONE</th><th>ZONE</th><th>STATUS</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:24px;font-size:.68rem;color:#475569;text-align:center">No active detections</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with analytics_col:
    st.markdown('<div style="border:1px solid #1E2D4A">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label"><div class="live-dot"></div>WILDLIFE ANALYTICS</div>', unsafe_allow_html=True)

    # Species distribution
    all_sp = defaultdict(int)
    for ds_v in drone_states.values():
        for t in ds_v.get("tracks",[]):
            if t.get("type")=="animal": all_sp[t.get("class_name","?")] += 1
    for k,v in st.session_state.species_hist.items():
        all_sp[k] = max(all_sp[k], min(v,15))

    st.markdown('<div style="padding:8px 16px 4px;font-size:.6rem;color:#475569;letter-spacing:2px;text-transform:uppercase">SPECIES DISTRIBUTION</div>', unsafe_allow_html=True)
    if all_sp:
        max_c = max(all_sp.values()) or 1
        rows_sp=""
        for sp,cnt in sorted(all_sp.items(),key=lambda x:-x[1])[:6]:
            pct=int(cnt/max_c*100)
            rows_sp+=f"""<div class="sp-row">
              <div class="sp-name">{sp}</div>
              <div class="sp-track"><div class="sp-fill" style="width:{pct}%"></div></div>
              <div class="sp-cnt">{cnt}</div>
            </div>"""
        st.markdown(f'<div class="analytics-wrap">{rows_sp}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:16px;font-size:.65rem;color:#475569">Awaiting data...</div>', unsafe_allow_html=True)

    # Trend sparkline (SVG)
    at = list(st.session_state.animal_trend)
    ht = list(st.session_state.human_trend)
    max_v = max(max(at),max(ht),1)
    W,H = 240,60
    def svg_line(data,color):
        n=len(data)
        if n<2: return ""
        pts=" ".join(f"{i*(W//(n-1))},{H-int(v/max_v*(H-4))-2}" for i,v in enumerate(data))
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.5" stroke-linejoin="round"/>'
    st.markdown(f"""
    <div style="padding:0 16px 12px">
      <div style="font-size:.6rem;color:#475569;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px">DETECTION TREND</div>
      <div style="display:flex;gap:12px;font-size:.58rem;margin-bottom:6px">
        <span style="color:#3B82F6">─ Animals</span>
        <span style="color:#F97316">─ Humans</span>
      </div>
      <svg width="100%" viewBox="0 0 {W} {H}" style="background:#0A1628;border-radius:4px">
        {svg_line(at,'#3B82F6')}
        {svg_line(ht,'#F97316')}
      </svg>
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with fleet_col:
    st.markdown('<div style="border:1px solid #1E2D4A">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label"><div class="live-dot"></div>DRONE FLEET STATUS</div>', unsafe_allow_html=True)
    if drone_ids:
        st.markdown('<div class="fleet-wrap">', unsafe_allow_html=True)
        for did in drone_ids:
            ds    = drone_states[did]
            a_cnt = ds.get("animal_count",0)
            h_cnt = ds.get("human_count",0)
            fid   = ds.get("frame_id",0)
            d_alrt= ds.get("poacher_alert",False)
            label = ds.get("feed_label","Feed")
            num   = did.replace("drone_","")
            st.markdown(f"""
            <div class="drone-card">
              <div class="drone-top">
                <div class="drone-name"><span class="icon">🚁</span>DRONE {num}</div>
                <div class="online-badge"><div class="online-dot"></div>ONLINE</div>
              </div>
              <div class="drone-grid">
                <div>Frame Count</div><div><span>#{fid:,}</span></div>
                <div>Animals Tracked</div><div><span>{a_cnt}</span></div>
                <div>Humans Tracked</div><div><span style="color:{'#EF4444' if h_cnt else '#94A3B8'}">{h_cnt}</span></div>
                <div>Status</div><div><span style="color:{'#EF4444' if d_alrt else '#22C55E'}">{'Alert' if d_alrt else 'Normal'}</span></div>
              </div>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="padding:20px;font-size:.68rem;color:#475569;text-align:center">No drones online</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with timeline_col:
    st.markdown('<div style="border:1px solid #1E2D4A">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label"><div class="live-dot"></div>RECENT ALERT TIMELINE<span class="sec-label-right">View All</span></div>', unsafe_allow_html=True)
    tl_html=""
    icon_map={"red":"🚨","orange":"⚡","blue":"🛡️","green":"🐾"}
    for item in reversed(list(st.session_state.timeline)[-8:]):
        lvl = item.get("level","blue")
        ico = item.get("icon", icon_map.get(lvl,"●"))
        tl_html+=f"""<div class="tl-item">
          <div class="tl-left">
            <div class="tl-icon {lvl}">{ico}</div>
            <div class="tl-time">{item['time']}</div>
          </div>
          <div class="tl-content">
            <div class="tl-msg">{item['msg']}</div>
            <div class="tl-sub">{item.get('sub','')}</div>
          </div>
        </div>"""
    st.markdown(f'<div class="timeline-wrap">{tl_html}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding:12px 24px;background:#0D1526;border-top:1px solid #1E2D4A;
     display:flex;justify-content:space-between;align-items:center;
     font-size:.58rem;color:#475569;font-family:'JetBrains Mono',monospace">
  <div style="display:flex;align-items:center;gap:10px">
    <span style="font-size:1rem">🦁</span>
    <div>
      <div style="color:#94A3B8;font-weight:600">AI-Based Multi-Drone Wildlife Monitoring & Intrusion Detection System</div>
      <div style="margin-top:2px">YOLOv8 Wildlife Detection &nbsp;•&nbsp; YOLOv8 Human Detection &nbsp;•&nbsp; DeepSORT Tracking &nbsp;•&nbsp; Real-Time Analytics</div>
    </div>  </div>
  <div>{time.strftime("%Y-%m-%d %H:%M:%S")}</div>
</div>
""", unsafe_allow_html=True)

time.sleep(1)
st.rerun()
