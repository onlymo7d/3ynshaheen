from pathlib import Path
import json
import time
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_DIR / "Logs"
LIVE_STATUS_PATH = LOGS_DIR / "live_status.json"
LIVE_FRAME_PATH = LOGS_DIR / "live_frame.jpg"

st.set_page_config(page_title="Perimeter Detection System", page_icon="🛡️", layout="wide")

st.markdown('''
<style>
.block-container {padding-top: 1.1rem; padding-bottom: 1.5rem;}
.threat-card {padding: 1rem 1.2rem; border-radius: 0.8rem; border: 1px solid rgba(128,128,128,.25); margin-bottom: .8rem;}
.low {background: rgba(0,180,80,.12);}
.medium {background: rgba(255,190,0,.12);}
.high {background: rgba(255,120,0,.14);}
.critical {background: rgba(220,0,0,.14);}
.muted {opacity: .65; font-size: .9rem;}
</style>
''', unsafe_allow_html=True)


def read_json(path):
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return None


def read_events(path):
    if not path:
        return []

    event_path = Path(path)

    if not event_path.exists():
        return []

    events = []

    try:
        with open(
            event_path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                events.append(event)

    except OSError:
        return []

    return events


def threat_label(threat):
    return {"LOW":"🟢 LOW","MEDIUM":"🟡 MEDIUM","HIGH":"🟠 HIGH","CRITICAL":"🔴 CRITICAL"}.get(threat, f"⚪ {threat}")


def threat_class(threat):
    return {"LOW":"low","MEDIUM":"medium","HIGH":"high","CRITICAL":"critical"}.get(threat, "")


st.title("🛡️ Perimeter Detection & Classification")
st.caption("Real-time detection • tracking • threat assessment • alerts")

status = read_json(LIVE_STATUS_PATH)
if status is None:
    st.warning("Backend data is not available yet. Start live_pipeline_v1.py first.")
    st.code("python live_pipeline_v1.py", language="powershell")
    time.sleep(1)
    st.rerun()

counts = status.get("counts", {})
scene_threat = status.get("scene_threat", "UNKNOWN")
camera_connected = status.get("camera_connected", False)
storage_ok = status.get("storage_ok", False)

html = (
    f'<div class="threat-card {threat_class(scene_threat)}">'
    '<div class="muted">CURRENT SCENE THREAT</div>'
    f'<div style="font-size:2rem;font-weight:700;">{threat_label(scene_threat)}</div>'
    '</div>'
)
st.markdown(html, unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
m1.metric("FPS", f"{status.get('fps', 0):.1f}")
m2.metric("AI Latency", f"{status.get('ai_latency_ms', 0):.1f} ms")
m3.metric("Camera", "Connected" if camera_connected else "Disconnected")
m4.metric("Storage", "OK" if storage_ok else "Warning")

st.divider()
left, right = st.columns([2.1, 1])

with left:
    st.subheader("Live Detection")
    st.markdown(
        """
        <img
            src="http://127.0.0.1:5000/video_feed"
            style="
                width:100%;
                border-radius:10px;
            "
        >
        """,
        unsafe_allow_html=True
    )

with right:
    st.subheader("Live Statistics")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Agents", counts.get("agent", 0))
        st.metric("Civilian Vehicles", counts.get("civilian_vehicle", 0))
        st.metric("Drones", counts.get("drone", 0))
    with c2:
        st.metric("Non-Agents", counts.get("non_agent", 0))
        st.metric("Military Vehicles", counts.get("military_vehicle", 0))
        st.metric("Armed Persons", counts.get("armed_person", 0))

st.divider()
st.subheader("Latest HIGH / CRITICAL Alerts")
events = read_events(status.get("current_event_log"))
latest_events = list(reversed(events[-10:]))

if not latest_events:
    st.success("No HIGH or CRITICAL alerts in the current session.")
else:
    for event in latest_events:
        threat = event.get("threat_level", "UNKNOWN")
        classification = event.get("classification", "Unknown")
        timestamp = event.get("timestamp", "")
        confidence = float(event.get("confidence", 0) or 0)
        with st.expander(f"{threat_label(threat)} — {classification} — {timestamp}"):
            info_col, image_col = st.columns([1, 1.8])
            with info_col:
                st.write(f"**Event ID:** {event.get('event_id', '-')}")
                st.write(f"**Object:** {event.get('object_type', '-')}")
                st.write(f"**Track ID:** {event.get('track_id', '-')}")
                st.write(f"**Confidence:** {confidence:.2f}")
                st.write(f"**Weapon confirmed:** {'Yes' if event.get('has_weapon', False) else 'No'}")
            with image_col:
                snapshot_path = event.get("snapshot_path")
                if snapshot_path and Path(snapshot_path).exists():
                    try:
                        st.image(Path(snapshot_path).read_bytes(), use_container_width=True)
                    except OSError:
                        st.info("Snapshot is being updated...")
                else:
                    st.info("No snapshot available.")

st.caption(f"Session: {status.get('session_id', '-')} • Last backend update: {status.get('timestamp', '-')}")

