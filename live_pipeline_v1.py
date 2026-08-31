from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
import json
import math
import shutil
import time
import torch

import cv2
from ultralytics import YOLO

from flask import (
    Flask,
    Response,
    jsonify,
    render_template_string,
    send_from_directory,
)
from threading import Thread, Lock


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "Models"

PERSON_MODEL_PATH = MODELS_DIR / "person_best.pt"

# Keep this spelling because your model is currently named this.
VEHICLE_MODEL_PATH = MODELS_DIR / "vechile_best.pt"

DRONE_MODEL_PATH = MODELS_DIR / "drone_best.pt"
WEAPON_MODEL_PATH = MODELS_DIR / "weapon_best.pt"

LOGS_DIR = PROJECT_DIR / "Logs"
SNAPSHOTS_DIR = PROJECT_DIR / "Snapshots"

LOGS_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

SESSION_TIME = datetime.now().strftime("%Y%m%d_%H%M%S")

TODAY = datetime.now().strftime("%Y-%m-%d")

SESSION_LOG_DIR = LOGS_DIR / TODAY
SESSION_SNAPSHOT_DIR = SNAPSHOTS_DIR / TODAY / SESSION_TIME

SESSION_LOG_DIR.mkdir(parents=True, exist_ok=True)
SESSION_SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

EVENT_LOG_PATH = (
    SESSION_LOG_DIR
    / f"session_{SESSION_TIME}_events.jsonl"
)

# Dashboard bridge files
LIVE_STATUS_PATH = LOGS_DIR / "live_status.json"
LIVE_FRAME_PATH = LOGS_DIR / "live_frame.jpg"
LIVE_FRAME_WRITE_INTERVAL = 1
LIVE_FRAME_JPEG_QUALITY = 65
MIN_FREE_STORAGE_MB = 250



app = Flask(__name__)

latest_stream_frame = None
latest_jpeg_frame = None

stream_lock = Lock()
jpeg_lock = Lock()



# ============================================================
# CAMERA / MODEL SETTINGS
# ============================================================

PERSON_CONF = 0.45
VEHICLE_CONF = 0.99
DRONE_CONF = 0.43
WEAPON_CONF = 0.50

IMG_SIZE = 640
CAMERA_INDEX = 0

BOX_THICKNESS = 2
FONT_SCALE = 0.55

# ============================================================
# DEVICE AUTO-SELECTION
# ============================================================

if torch.cuda.is_available():
    DEVICE = 0
    print("Using NVIDIA CUDA GPU")

elif (
    hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    DEVICE = "mps"
    print("Using Apple MPS GPU")

else:
    DEVICE = "cpu"
    print("Using CPU")


# ============================================================
# PERSON CLASSIFICATION SMOOTHING
# ============================================================

CLASS_HISTORY_SIZE = 10
CLASS_REQUIRED_HITS = 7

classification_history = defaultdict(
    lambda: deque(maxlen=CLASS_HISTORY_SIZE)
)

stable_classification = {}


# ============================================================
# WEAPON CONFIRMATION
# ============================================================

WEAPON_HISTORY_SIZE = 10

# Weapon becomes confirmed after 5 positive
# associations among the latest 10 checks.
WEAPON_CONFIRM_HITS = 5

# Once confirmed, require 7 misses among the
# latest 10 checks before disarming the person.
WEAPON_RELEASE_MISSES = 7

weapon_history = defaultdict(
    lambda: deque(maxlen=WEAPON_HISTORY_SIZE)
)

confirmed_weapon_state = {}


# ============================================================
# WEAPON DISPLAY SMOOTHING
# ============================================================

WEAPON_DISPLAY_HOLD_FRAMES = 3

# This was missing from your current code.
weapon_display_memory = {}


# ============================================================
# VEHICLE / DRONE TEMPORAL CONFIRMATION
# ============================================================

OBJECT_HISTORY_SIZE = 8
OBJECT_REQUIRED_HITS = 5

object_detection_history = defaultdict(
    lambda: deque(maxlen=OBJECT_HISTORY_SIZE)
)


# ============================================================
# THREAT ENGINE
# ============================================================

THREAT_PRIORITY = {
    "LOW": 0,
    "MEDIUM": 1,
    "HIGH": 2,
    "CRITICAL": 3,
}

THREAT_COLORS = {
    "LOW": (0, 255, 0),
    "MEDIUM": (0, 255, 255),
    "HIGH": (0, 165, 255),
    "CRITICAL": (0, 0, 255),
}

# Only these levels are security events.
ALERT_THREATS = {
    "HIGH",
    "CRITICAL",
}


# ============================================================
# EVENT MANAGER SETTINGS
# ============================================================

event_counter = 0

# Example:
#
# person:4:
#     threat = CRITICAL
#     classification = non-agent
#     has_weapon = True
#     last_seen_frame = 950
#
active_event_states = {}

# Object must be missing for this many frames before
# its state is forgotten.
EVENT_DISAPPEAR_GRACE_FRAMES = 45


# ============================================================
# MODEL VALIDATION
# ============================================================

def validate_models():

    paths = [
        PERSON_MODEL_PATH,
        VEHICLE_MODEL_PATH,
        DRONE_MODEL_PATH,
        WEAPON_MODEL_PATH,
    ]

    for path in paths:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing model:\n{path}"
            )


def storage_is_ok():
    try:
        usage = shutil.disk_usage(PROJECT_DIR)
        free_mb = usage.free / (1024 * 1024)
        return free_mb >= MIN_FREE_STORAGE_MB
    except OSError:
        return False


# ============================================================
# DETECTION EXTRACTION
# ============================================================

def extract_detections(result, prefix):

    detections = []

    if result.boxes is None:
        return detections

    names = result.names

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        track_id = None

        if box.id is not None:
            track_id = int(box.id[0])

        detections.append({
            "source_model": prefix,
            "class_id": class_id,
            "class_name": names[class_id],
            "confidence": confidence,
            "bbox": [x1, y1, x2, y2],
            "track_id": track_id,

            # Updated later.
            "weapon_detected_now": False,
            "has_weapon": False,
            "threat": "LOW",
        })

    return detections


# ============================================================
# GEOMETRY
# ============================================================

def get_center(bbox):

    x1, y1, x2, y2 = bbox

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def point_inside_box(point, bbox, margin=0):

    px, py = point
    x1, y1, x2, y2 = bbox

    return (
        x1 - margin <= px <= x2 + margin
        and
        y1 - margin <= py <= y2 + margin
    )


def center_distance(bbox_a, bbox_b):

    ax, ay = get_center(bbox_a)
    bx, by = get_center(bbox_b)

    return math.sqrt(
        (ax - bx) ** 2
        +
        (ay - by) ** 2
    )


# ============================================================
# PERSON CLASSIFICATION SMOOTHING
# ============================================================

def update_stable_classification(people):

    for person in people:

        track_id = person.get("track_id")

        if track_id is None:
            continue

        raw_class = person["class_name"].lower()

        classification_history[
            track_id
        ].append(raw_class)

        history = classification_history[
            track_id
        ]

        agent_hits = history.count("agent")
        non_agent_hits = history.count("non-agent")

        # ----------------------------------------------------
        # FIRST CLASSIFICATION
        # ----------------------------------------------------

        if track_id not in stable_classification:

            stable_classification[
                track_id
            ] = raw_class

        else:

            current_stable = stable_classification[
                track_id
            ]

            # Require strong evidence before switching.
            if (
                current_stable != "agent"
                and
                agent_hits >= CLASS_REQUIRED_HITS
            ):

                stable_classification[
                    track_id
                ] = "agent"

            elif (
                current_stable != "non-agent"
                and
                non_agent_hits >= CLASS_REQUIRED_HITS
            ):

                stable_classification[
                    track_id
                ] = "non-agent"

        person["raw_class_name"] = raw_class

        person["class_name"] = (
            stable_classification[
                track_id
            ]
        )


# ============================================================
# WEAPON ↔ PERSON ASSOCIATION
# ============================================================

def associate_weapons_to_people(detections):

    people = [
        d
        for d in detections
        if (
            d["source_model"] == "person"
            and
            d.get("track_id") is not None
        )
    ]

    weapons = [
        d
        for d in detections
        if d["source_model"] == "weapon"
    ]

    associations = {}

    for weapon in weapons:

        weapon_center = get_center(
            weapon["bbox"]
        )

        candidates = []

        for person in people:

            person_bbox = person["bbox"]

            x1, y1, x2, y2 = person_bbox

            person_width = x2 - x1
            person_height = y2 - y1

            # Allows a held weapon to extend outside
            # the person's detector box.
            margin = int(
                max(
                    person_width,
                    person_height
                )
                * 0.35
            )

            if point_inside_box(
                weapon_center,
                person_bbox,
                margin=margin
            ):

                distance = center_distance(
                    weapon["bbox"],
                    person_bbox
                )

                candidates.append(
                    (
                        distance,
                        person
                    )
                )

        if not candidates:
            continue

        candidates.sort(
            key=lambda item: item[0]
        )

        _, matched_person = candidates[0]

        track_id = matched_person[
            "track_id"
        ]

        previous_weapon = associations.get(
            track_id
        )

        if (
            previous_weapon is None
            or
            weapon["confidence"]
            >
            previous_weapon["weapon_confidence"]
        ):

            associations[track_id] = {
                "weapon": True,
                "weapon_confidence":
                    weapon["confidence"],
                "weapon_bbox":
                    weapon["bbox"],
            }

    return associations


# ============================================================
# WEAPON TEMPORAL CONFIRMATION
# ============================================================

def update_weapon_confirmation(
    people,
    weapon_associations
):

    confirmed = {}

    for person in people:

        track_id = person.get(
            "track_id"
        )

        if track_id is None:
            continue

        detected_now = (
            track_id
            in
            weapon_associations
        )

        weapon_history[
            track_id
        ].append(
            1 if detected_now else 0
        )

        history = weapon_history[
            track_id
        ]

        hits = sum(history)
        misses = len(history) - hits

        current_state = (
            confirmed_weapon_state.get(
                track_id,
                False
            )
        )

        # ----------------------------------------------------
        # CURRENTLY UNARMED
        # ----------------------------------------------------

        if not current_state:

            if hits >= WEAPON_CONFIRM_HITS:

                current_state = True

        # ----------------------------------------------------
        # CURRENTLY ARMED
        # ----------------------------------------------------

        else:

            if (
                len(history)
                >=
                WEAPON_HISTORY_SIZE
                and
                misses
                >=
                WEAPON_RELEASE_MISSES
            ):

                current_state = False

        confirmed_weapon_state[
            track_id
        ] = current_state

        confirmed[
            track_id
        ] = current_state

    return confirmed


# ============================================================
# WEAPON DISPLAY SMOOTHING
# ============================================================

def smooth_weapon_display(detections):

    current_weapon_ids = set()

    for detection in detections:

        if (
            detection["source_model"]
            !=
            "weapon"
        ):

            continue

        track_id = detection.get(
            "track_id"
        )

        if track_id is None:
            continue

        current_weapon_ids.add(
            track_id
        )

        weapon_display_memory[
            track_id
        ] = {
            "detection":
                detection.copy(),

            "missing_frames":
                0,
        }

    persistent_weapons = []

    for track_id in list(
        weapon_display_memory.keys()
    ):

        memory = weapon_display_memory[
            track_id
        ]

        if track_id in current_weapon_ids:

            persistent_weapons.append(
                memory["detection"]
            )

            continue

        memory[
            "missing_frames"
        ] += 1

        if (
            memory["missing_frames"]
            <=
            WEAPON_DISPLAY_HOLD_FRAMES
        ):

            remembered_detection = (
                memory[
                    "detection"
                ].copy()
            )

            remembered_detection[
                "display_memory"
            ] = True

            persistent_weapons.append(
                remembered_detection
            )

        else:

            del weapon_display_memory[
                track_id
            ]

    return persistent_weapons


# ============================================================
# VEHICLE / DRONE TEMPORAL CONFIRMATION
# ============================================================

def update_object_confirmation(detections):

    confirmed_keys = set()
    currently_seen = set()

    for detection in detections:

        source = detection[
            "source_model"
        ]

        if source not in {
            "vehicle",
            "drone",
        }:
            continue

        track_id = detection.get(
            "track_id"
        )

        if track_id is None:
            continue

        key = (
            f"{source}:{track_id}"
        )

        currently_seen.add(key)

        object_detection_history[
            key
        ].append(1)

    # Add misses to existing histories.
    for key in list(
        object_detection_history.keys()
    ):

        if key not in currently_seen:

            object_detection_history[
                key
            ].append(0)

    # Require persistence.
    for (
        key,
        history
    ) in object_detection_history.items():

        if (
            sum(history)
            >=
            OBJECT_REQUIRED_HITS
        ):

            confirmed_keys.add(key)

    return confirmed_keys


# ============================================================
# PERSON THREAT
# ============================================================

def get_person_threat(person):

    name = person[
        "class_name"
    ].lower()

    has_weapon = person.get(
        "has_weapon",
        False
    )

    if name == "non-agent":

        if has_weapon:
            return "CRITICAL"

        return "LOW"

    if name == "agent":

        if has_weapon:
            return "HIGH"

        return "MEDIUM"

    return "LOW"


# ============================================================
# VEHICLE / DRONE THREAT
# ============================================================

def get_object_threat(
    detection,
    confirmed_objects
):

    source = detection[
        "source_model"
    ]

    name = detection[
        "class_name"
    ].lower()

    track_id = detection.get(
        "track_id"
    )

    # Raw weapons are handled through people.
    if source == "weapon":
        return "LOW"

    if track_id is None:
        return "LOW"

    key = (
        f"{source}:{track_id}"
    )

    if key not in confirmed_objects:
        return "LOW"

    if source == "drone":
        return "HIGH"

    if source == "vehicle":

        if "military" in name:
            return "HIGH"

        return "LOW"

    return "LOW"


# ============================================================
# SCENE THREAT
# ============================================================

def calculate_scene_threat(detections):

    highest = "LOW"

    for detection in detections:

        threat = detection.get(
            "threat",
            "LOW"
        )

        if (
            THREAT_PRIORITY[threat]
            >
            THREAT_PRIORITY[highest]
        ):

            highest = threat

    return highest


# ============================================================
# DRAWING
# ============================================================

def draw_detection(
    frame,
    detection
):

    x1, y1, x2, y2 = (
        detection["bbox"]
    )

    source = detection[
        "source_model"
    ]

    name = detection[
        "class_name"
    ].upper()

    confidence = detection[
        "confidence"
    ]

    threat = detection.get(
        "threat",
        "LOW"
    )

    color = THREAT_COLORS.get(
        threat,
        (255, 255, 255)
    )

    track_id = detection.get(
        "track_id"
    )

    # --------------------------------------------------------
    # PERSON
    # --------------------------------------------------------

    if source == "person":

        if track_id is not None:

            label = (
                f"ID {track_id} | "
                f"{name} "
                f"{confidence:.2f}"
            )

        else:

            label = (
                f"{name} "
                f"{confidence:.2f}"
            )

        if detection.get(
            "has_weapon",
            False
        ):

            label += " | WEAPON"

        label += (
            f" | {threat}"
        )

    # --------------------------------------------------------
    # OTHER OBJECTS
    # --------------------------------------------------------

    else:

        if track_id is not None:

            label = (
                f"ID {track_id} | "
                f"{name} "
                f"{confidence:.2f} "
                f"| {threat}"
            )

        else:

            label = (
                f"{name} "
                f"{confidence:.2f} "
                f"| {threat}"
            )

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        BOX_THICKNESS
    )

    cv2.putText(
        frame,
        label,
        (
            x1,
            max(
                20,
                y1 - 8
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        color,
        2,
        cv2.LINE_AA
    )


# ============================================================
# EVENT MANAGER
#
# IMPORTANT:
# ONLY HIGH AND CRITICAL ARE LOGGED.
# ONLY HIGH AND CRITICAL SAVE SNAPSHOTS.
# ============================================================

def next_event_id():

    global event_counter

    event_counter += 1

    return (
        f"EVT-{event_counter:06d}"
    )


def save_snapshot(
    frame,
    event_id,
    threat
):

    # Absolute protection:
    # never save LOW / MEDIUM.
    if threat not in ALERT_THREATS:
        return None

    timestamp = (
        datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    filename = (
        f"{timestamp}_"
        f"{event_id}_"
        f"{threat}.jpg"
    )

    path = (
        SESSION_SNAPSHOT_DIR
        /
        filename
    )

    success = cv2.imwrite(
        str(path),
        frame
    )

    if not success:

        print(
            "WARNING: "
            f"Snapshot failed: {path}"
        )

        return None

    return str(path)


def append_event_to_log(event):

    # Extra safety.
    #
    # Even if this function is accidentally called
    # for LOW/MEDIUM, it will refuse to log it.
    if (
        event["threat_level"]
        not in
        ALERT_THREATS
    ):

        return

    with open(
        EVENT_LOG_PATH,
        "a",
        encoding="utf-8"
    ) as file:

        file.write(
            json.dumps(
                event,
                ensure_ascii=False
            )
            +
            "\n"
        )


def build_event_key(detection):

    source = detection[
        "source_model"
    ]

    # Weapon events belong to the associated person.
    if source == "weapon":
        return None

    track_id = detection.get(
        "track_id"
    )

    if track_id is None:
        return None

    return (
        f"{source}:{track_id}"
    )


def create_event(
    detection,
    frame
):

    threat = detection.get(
        "threat",
        "LOW"
    )

    # Absolute protection:
    #
    # LOW / MEDIUM can never create event records.
    if threat not in ALERT_THREATS:
        return None

    event_id = next_event_id()

    timestamp = (
        datetime.now().isoformat(
            timespec="seconds"
        )
    )

    event = {
        "event_id":
            event_id,

        "timestamp":
            timestamp,

        "object_type":
            detection[
                "source_model"
            ],

        "track_id":
            detection.get(
                "track_id"
            ),

        "classification":
            detection[
                "class_name"
            ],

        "confidence":
            round(
                detection[
                    "confidence"
                ],
                4
            ),

        "threat_level":
            threat,

        "has_weapon":
            detection.get(
                "has_weapon",
                False
            ),

        "snapshot_path":
            None,
    }

    # High/Critical automatically get snapshot.
    event[
        "snapshot_path"
    ] = save_snapshot(
        frame,
        event_id,
        threat
    )

    append_event_to_log(
        event
    )

    return event


def process_events(
    detections,
    frame,
    frame_counter
):

    """
    Behaviour:

    LOW / MEDIUM:
        - NEVER logged
        - NEVER snapshotted
        - state is still remembered internally

    HIGH / CRITICAL:
        - logged when object first enters alert state
        - logged when HIGH <-> CRITICAL changes
        - snapshot saved
        - unchanged alert does not create repeated events

    Example:

    Person LOW
        -> nothing logged

    Person becomes CRITICAL
        -> 1 event + snapshot

    Remains CRITICAL for 30 seconds
        -> nothing else logged

    Returns LOW
        -> state changes silently

    Becomes CRITICAL later
        -> new event + snapshot
    """

    new_events = []

    for detection in detections:

        # Raw weapon detections never become
        # independent alert events.
        if (
            detection[
                "source_model"
            ]
            ==
            "weapon"
        ):

            continue

        event_key = build_event_key(
            detection
        )

        if event_key is None:
            continue

        threat = detection.get(
            "threat",
            "LOW"
        )

        classification = detection[
            "class_name"
        ]

        has_weapon = detection.get(
            "has_weapon",
            False
        )

        previous = active_event_states.get(
            event_key
        )

        # ----------------------------------------------------
        # NEW TRACKED OBJECT
        # ----------------------------------------------------

        if previous is None:

            active_event_states[
                event_key
            ] = {
                "threat":
                    threat,

                "classification":
                    classification,

                "has_weapon":
                    has_weapon,

                "last_seen_frame":
                    frame_counter,
            }

            # Log only if it enters already HIGH/CRITICAL.
            if threat in ALERT_THREATS:

                event = create_event(
                    detection,
                    frame
                )

                if event is not None:

                    new_events.append(
                        event
                    )

            continue

        # ----------------------------------------------------
        # EXISTING OBJECT
        # ----------------------------------------------------

        previous_threat = previous[
            "threat"
        ]

        previous_class = previous[
            "classification"
        ]

        previous_weapon = previous[
            "has_weapon"
        ]

        previous[
            "last_seen_frame"
        ] = frame_counter

        state_changed = (
            previous_threat != threat
            or
            previous_class != classification
            or
            previous_weapon != has_weapon
        )

        if not state_changed:
            continue

        # ----------------------------------------------------
        # SHOULD WE CREATE AN ALERT?
        # ----------------------------------------------------

        should_log = False

        # Entering HIGH / CRITICAL from a normal state.
        if (
            threat in ALERT_THREATS
            and
            previous_threat
            not in ALERT_THREATS
        ):

            should_log = True

        # HIGH -> CRITICAL or CRITICAL -> HIGH.
        elif (
            threat in ALERT_THREATS
            and
            previous_threat in ALERT_THREATS
            and
            threat != previous_threat
        ):

            should_log = True

        # If the threat remains HIGH/CRITICAL but the
        # classification changes, don't spam an event.
        #
        # The threat level is the alert state that matters.

        if should_log:

            event = create_event(
                detection,
                frame
            )

            if event is not None:

                new_events.append(
                    event
                )

        # Always update internal state,
        # even when LOW/MEDIUM is not logged.
        previous[
            "threat"
        ] = threat

        previous[
            "classification"
        ] = classification

        previous[
            "has_weapon"
        ] = has_weapon

    # --------------------------------------------------------
    # REMOVE STALE TRACKS
    # --------------------------------------------------------

    stale_keys = []

    for (
        key,
        state
    ) in active_event_states.items():

        frames_missing = (
            frame_counter
            -
            state[
                "last_seen_frame"
            ]
        )

        if (
            frames_missing
            >
            EVENT_DISAPPEAR_GRACE_FRAMES
        ):

            stale_keys.append(key)

    for key in stale_keys:

        del active_event_states[
            key
        ]

    return new_events


# ============================================================
# DASHBOARD BRIDGE
# ============================================================

def write_live_status(
    fps,
    inference_ms,
    scene_threat,
    agent_count,
    non_agent_count,
    civilian_vehicle_count,
    military_vehicle_count,
    drone_count,
    raw_weapon_count,
    confirmed_armed_people,
    camera_connected=True
):

    status = {
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "session_id": SESSION_TIME,
        "fps": round(fps, 2),
        "ai_latency_ms": round(
            inference_ms,
            2
        ),
        "scene_threat": scene_threat,
        "camera_connected": camera_connected,
        "storage_ok": storage_is_ok(),

        "counts": {
            "agent": agent_count,
            "non_agent": non_agent_count,
            "civilian_vehicle":
                civilian_vehicle_count,
            "military_vehicle":
                military_vehicle_count,
            "drone": drone_count,
            "raw_weapon":
                raw_weapon_count,
            "armed_person":
                confirmed_armed_people,
        },

        "current_event_log":
            str(EVENT_LOG_PATH),

        "current_snapshot_dir":
            str(SESSION_SNAPSHOT_DIR),

        "live_frame_path":
            str(LIVE_FRAME_PATH),
    }

    temp_path = (
        LIVE_STATUS_PATH.with_suffix(
            ".tmp"
        )
    )

    try:

        with open(
            temp_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                status,
                file,
                ensure_ascii=False,
                indent=2
            )

        # ----------------------------------------------------
        # WINDOWS-SAFE REPLACEMENT
        # ----------------------------------------------------

        replaced = False

        for attempt in range(5):

            try:

                temp_path.replace(
                    LIVE_STATUS_PATH
                )

                replaced = True
                break

            except PermissionError:

                time.sleep(0.02)

        # If Windows still has the file locked,
        # simply skip this update instead of crashing AI.
        if not replaced:

            try:
                temp_path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass

    except OSError as error:

        # Dashboard output should NEVER stop
        # the detection pipeline.
        print(
            f"[Dashboard warning] "
            f"Could not update live status: "
            f"{error}"
        )


def write_live_frame(
    frame,
    frame_counter
):

    if (
        frame_counter
        %
        LIVE_FRAME_WRITE_INTERVAL
        !=
        0
    ):
        return

    temp_path = (
        LIVE_FRAME_PATH.with_name(
            "live_frame_tmp.jpg"
        )
    )

    try:

        success = cv2.imwrite(
            str(temp_path),
            frame,
            [
                int(
                    cv2.IMWRITE_JPEG_QUALITY
                ),
                LIVE_FRAME_JPEG_QUALITY,
            ]
        )

        if not success:
            return

        replaced = False

        for attempt in range(5):

            try:

                temp_path.replace(
                    LIVE_FRAME_PATH
                )

                replaced = True
                break

            except PermissionError:

                time.sleep(0.02)

        if not replaced:

            try:
                temp_path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass

    except OSError as error:

        print(
            f"[Dashboard warning] "
            f"Could not update live frame: "
            f"{error}"
        )


@app.route("/logo")
def logo():
    return send_from_directory(
        str(PROJECT_DIR / "static"),
        "3ynshaheen_logo.png"
    )

@app.route("/video_feed")
def video_feed():

    def generate():

        last_frame = None

        while True:

            with jpeg_lock:
                frame = latest_jpeg_frame

            if frame is None:
                time.sleep(0.01)
                continue

            # Don't resend the exact same frame
            if frame is last_frame:
                time.sleep(0.005)
                continue

            last_frame = frame

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

    return Response(
        generate(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


def start_video_server():
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )


def stream_encoder():

    global latest_jpeg_frame

    while True:

        # Copy the newest AI frame quickly
        with stream_lock:

            if latest_stream_frame is None:
                frame = None
            else:
                frame = latest_stream_frame.copy()

        if frame is None:
            time.sleep(0.01)
            continue

        # Smaller dashboard stream
        frame = cv2.resize(
            frame,
            (960, 540)
        )

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [
                int(cv2.IMWRITE_JPEG_QUALITY),
                50
            ]
        )

        if not success:
            continue

        jpeg = buffer.tobytes()

        with jpeg_lock:
            latest_jpeg_frame = jpeg

        # Prevent unnecessary CPU spinning
        time.sleep(0.005)

# ============================================================
# FLASK DASHBOARD
# ============================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">

<head>

    <meta charset="UTF-8">

    <meta
        name="viewport"
        content="width=device-width, initial-scale=1.0"
    >

    <title>
        3YNSHAHEEN | عين شاهين
    </title>

    <style>

        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            background: #0b0f0c;
            color: #eeeeee;
            font-family: Arial, Helvetica, sans-serif;
        }

        .page {
            width: 96%;
            max-width: 1600px;
            margin: auto;
            padding: 20px;
        }

        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 18px;
        }

        .brand {
            display: flex;
            align-items: center;
            gap: 18px;
        }

        .brand-logo {
            width: 90px;
            height: 90px;
            object-fit: contain;
            border-radius: 10px;
        }

        .title {
            font-size: 32px;
            font-weight: 800;
            letter-spacing: 2px;
        }

        .arabic-title {
            font-size: 20px;
            font-weight: 600;
            margin-top: 2px;
            color: #c4b58a;
        }

        .subtitle {
            color: #999999;
            margin-top: 5px;
        }

        .threat {
            padding: 12px 25px;
            border-radius: 8px;
            font-size: 22px;
            font-weight: 700;
        }

        .LOW {
            background: #163c25;
            color: #54e081;
        }

        .MEDIUM {
            background: #4b4215;
            color: #ffe268;
        }

        .HIGH {
            background: #4a2810;
            color: #ff9a45;
        }

        .CRITICAL {
            background: #4d1212;
            color: #ff5151;
        }

        .metrics {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 12px;
            margin-bottom: 18px;
        }

        .metric {
            background: #151a16;
            border: 1px solid #29312b;
            border-radius: 8px;
            padding: 14px;
        }

        .metric-title {
            color: #8f9991;
            font-size: 13px;
        }

        .metric-value {
            font-size: 23px;
            font-weight: bold;
            margin-top: 7px;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 2.2fr 1fr;
            gap: 18px;
        }

        .panel {
            background: #121713;
            border: 1px solid #29312b;
            border-radius: 10px;
            padding: 15px;
        }

        .panel h2 {
            margin-top: 0;
            font-size: 20px;
        }

        .video {
            width: 100%;
            border-radius: 8px;
            display: block;
            background: black;
        }

        .counts {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }

        .count-box {
            padding: 14px;
            background: #191f1a;
            border-radius: 7px;
        }

        .count-title {
            color: #9da59e;
            font-size: 13px;
        }

        .count-value {
            margin-top: 6px;
            font-size: 26px;
            font-weight: bold;
        }

        .alerts {
            margin-top: 18px;
        }

        .alert {
            display: grid;
            grid-template-columns: 1fr 220px;
            gap: 15px;

            background: #151a16;
            border-radius: 9px;
            border: 1px solid #29312b;

            margin-bottom: 10px;
            padding: 14px;
        }

        .alert.HIGH {
            border-left: 6px solid #ff8a32;
        }

        .alert.CRITICAL {
            border-left: 6px solid #ff3f3f;
        }

        .alert-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 8px;
        }

        .alert-info {
            color: #b9c0ba;
            line-height: 1.6;
        }

        .snapshot {
            width: 100%;
            border-radius: 6px;
        }

        .status-good {
            color: #4ee27a;
        }

        .status-bad {
            color: #ff5555;
        }

        @media (
            max-width: 1000px
        ) {

            .main-grid {
                grid-template-columns: 1fr;
            }

            .metrics {
                grid-template-columns: 1fr 1fr;
            }

        }

    </style>

</head>


<body>

<div class="page">

    <div class="header">

        <div class="brand">

            <img
                src="/logo"
                class="brand-logo"
                alt="3ynshaheen Logo"
            >

            <div>

                <div class="title">
                    3YNSHAHEEN
                </div>

                <div class="arabic-title">
                    عين شاهين
                </div>

                <div class="subtitle">
                    Real-Time Perimeter Detection & Classification
                </div>

            </div>

        </div>

        <div
            id="sceneThreat"
            class="threat LOW"
        >
            LOW
        </div>

    </div>


    <div class="metrics">

        <div class="metric">

            <div class="metric-title">
                FPS
            </div>

            <div
                id="fps"
                class="metric-value"
            >
                0
            </div>

        </div>


        <div class="metric">

            <div class="metric-title">
                AI LATENCY
            </div>

            <div
                id="latency"
                class="metric-value"
            >
                0 ms
            </div>

        </div>


        <div class="metric">

            <div class="metric-title">
                CAMERA
            </div>

            <div
                id="camera"
                class="metric-value"
            >
                -
            </div>

        </div>


        <div class="metric">

            <div class="metric-title">
                STORAGE
            </div>

            <div
                id="storage"
                class="metric-value"
            >
                -
            </div>

        </div>

    </div>


    <div class="main-grid">

        <div class="panel">

            <h2>
                Live Detection
            </h2>

            <img
                class="video"
                src="/video_feed"
            >

        </div>


        <div class="panel">

            <h2>
                Live Statistics
            </h2>

            <div class="counts">

                <div class="count-box">

                    <div class="count-title">
                        Agents
                    </div>

                    <div
                        id="agents"
                        class="count-value"
                    >
                        0
                    </div>

                </div>


                <div class="count-box">

                    <div class="count-title">
                        Non-Agents
                    </div>

                    <div
                        id="nonAgents"
                        class="count-value"
                    >
                        0
                    </div>

                </div>


                <div class="count-box">

                    <div class="count-title">
                        Civilian Vehicles
                    </div>

                    <div
                        id="civilianVehicles"
                        class="count-value"
                    >
                        0
                    </div>

                </div>


                <div class="count-box">

                    <div class="count-title">
                        Military Vehicles
                    </div>

                    <div
                        id="militaryVehicles"
                        class="count-value"
                    >
                        0
                    </div>

                </div>


                <div class="count-box">

                    <div class="count-title">
                        Drones
                    </div>

                    <div
                        id="drones"
                        class="count-value"
                    >
                        0
                    </div>

                </div>


                <div class="count-box">

                    <div class="count-title">
                        Armed Persons
                    </div>

                    <div
                        id="armedPersons"
                        class="count-value"
                    >
                        0
                    </div>

                </div>

            </div>

        </div>

    </div>


    <div class="alerts">

        <div class="panel">

            <h2>
                Latest HIGH / CRITICAL Alerts
            </h2>

            <div id="alerts">

                No alerts in this session.

            </div>

        </div>

    </div>

</div>


<script>

async function updateStatus() {

    try {

        const response =
            await fetch(
                "/api/status?t="
                +
                Date.now()
            );

        const data =
            await response.json();

        document.getElementById(
            "fps"
        ).textContent =
            Number(
                data.fps || 0
            ).toFixed(1);


        document.getElementById(
            "latency"
        ).textContent =
            Number(
                data.ai_latency_ms || 0
            ).toFixed(1)
            +
            " ms";


        const camera =
            document.getElementById(
                "camera"
            );

        if (
            data.camera_connected
        ) {

            camera.textContent =
                "CONNECTED";

            camera.className =
                "metric-value status-good";

        } else {

            camera.textContent =
                "DISCONNECTED";

            camera.className =
                "metric-value status-bad";

        }


        const storage =
            document.getElementById(
                "storage"
            );

        if (
            data.storage_ok
        ) {

            storage.textContent =
                "OK";

            storage.className =
                "metric-value status-good";

        } else {

            storage.textContent =
                "WARNING";

            storage.className =
                "metric-value status-bad";

        }


        const threat =
            data.scene_threat
            ||
            "LOW";


        const threatBox =
            document.getElementById(
                "sceneThreat"
            );

        threatBox.textContent =
            threat;

        threatBox.className =
            "threat "
            +
            threat;


        const counts =
            data.counts
            ||
            {};


        document.getElementById(
            "agents"
        ).textContent =
            counts.agent
            ||
            0;


        document.getElementById(
            "nonAgents"
        ).textContent =
            counts.non_agent
            ||
            0;


        document.getElementById(
            "civilianVehicles"
        ).textContent =
            counts.civilian_vehicle
            ||
            0;


        document.getElementById(
            "militaryVehicles"
        ).textContent =
            counts.military_vehicle
            ||
            0;


        document.getElementById(
            "drones"
        ).textContent =
            counts.drone
            ||
            0;


        document.getElementById(
            "armedPersons"
        ).textContent =
            counts.armed_person
            ||
            0;

    }

    catch (
        error
    ) {

        console.log(
            error
        );

    }

}



async function updateAlerts() {

    try {

        const response =
            await fetch(
                "/api/events?t="
                +
                Date.now()
            );


        const events =
            await response.json();


        const container =
            document.getElementById(
                "alerts"
            );


        if (
            events.length
            ===
            0
        ) {

            container.innerHTML =
                "No HIGH or CRITICAL alerts in this session.";

            return;

        }


        container.innerHTML =
            "";


        events.forEach(
            event => {

                const alert =
                    document.createElement(
                        "div"
                    );


                alert.className =
                    "alert "
                    +
                    event.threat_level;


                let snapshotHtml =
                    "";


                if (
                    event.snapshot_url
                ) {

                    snapshotHtml =
                        `
                        <img
                            class="snapshot"
                            src="${event.snapshot_url}?t=${Date.now()}"
                        >
                        `;

                }


                alert.innerHTML =
                    `
                    <div>

                        <div class="alert-title">
                            ${event.threat_level}
                            —
                            ${event.classification}
                        </div>

                        <div class="alert-info">

                            Time:
                            ${event.timestamp}
                            <br>

                            Track ID:
                            ${event.track_id}
                            <br>

                            Confidence:
                            ${Number(
                                event.confidence
                                ||
                                0
                            ).toFixed(2)}
                            <br>

                            Weapon:
                            ${
                                event.has_weapon
                                ?
                                "YES"
                                :
                                "NO"
                            }

                        </div>

                    </div>

                    <div>
                        ${snapshotHtml}
                    </div>
                    `;


                container.appendChild(
                    alert
                );

            }
        );

    }

    catch (
        error
    ) {

        console.log(
            error
        );

    }

}



updateStatus();
updateAlerts();


setInterval(
    updateStatus,
    400
);


setInterval(
    updateAlerts,
    1000
);

</script>

</body>

</html>
"""


@app.route("/")
def dashboard():

    return render_template_string(
        DASHBOARD_HTML
    )


@app.route("/api/status")
def api_status():

    try:

        if not LIVE_STATUS_PATH.exists():

            return jsonify({
                "fps": 0,
                "ai_latency_ms": 0,
                "scene_threat": "LOW",
                "camera_connected": False,
                "storage_ok": False,
                "counts": {},
            })


        with open(
            LIVE_STATUS_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )


        return jsonify(
            data
        )


    except (
        OSError,
        json.JSONDecodeError
    ):

        return jsonify({
            "fps": 0,
            "ai_latency_ms": 0,
            "scene_threat": "LOW",
            "camera_connected": True,
            "storage_ok": True,
            "counts": {},
        })


@app.route("/api/events")
def api_events():

    events = []


    if not EVENT_LOG_PATH.exists():

        return jsonify(
            events
        )


    try:

        with open(
            EVENT_LOG_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue


                try:

                    event = json.loads(
                        line
                    )

                except json.JSONDecodeError:

                    continue


                if (
                    event.get(
                        "threat_level"
                    )
                    not in
                    {
                        "HIGH",
                        "CRITICAL"
                    }
                ):

                    continue


                snapshot_path = (
                    event.get(
                        "snapshot_path"
                    )
                )


                if snapshot_path:

                    snapshot_name = Path(
                        snapshot_path
                    ).name


                    event[
                        "snapshot_url"
                    ] = (
                        "/snapshot/"
                        +
                        snapshot_name
                    )

                else:

                    event[
                        "snapshot_url"
                    ] = None


                events.append(
                    event
                )


    except OSError:

        pass


    # newest first
    events = list(
        reversed(
            events[-10:]
        )
    )


    return jsonify(
        events
    )


@app.route(
    "/snapshot/<path:filename>"
)
def snapshot(filename):

    return send_from_directory(
        SESSION_SNAPSHOT_DIR,
        filename
    )

# ============================================================
# MAIN
# ============================================================

def main():

    validate_models()

    print("=" * 70)
    print(
        "PERIMETER DETECTION — LIVE PIPELINE V4"
    )
    print("=" * 70)

    print(
        "Loading models..."
    )

    person_model = YOLO(
        str(
            PERSON_MODEL_PATH
        )
    )

    vehicle_model = YOLO(
        str(
            VEHICLE_MODEL_PATH
        )
    )

    drone_model = YOLO(
        str(
            DRONE_MODEL_PATH
        )
    )

    weapon_model = YOLO(
        str(
            WEAPON_MODEL_PATH
        )
    )

    print(
        "Models loaded."
    )

    # ========================================================
    # START VIDEO STREAM SERVER
    # ========================================================
    Thread(
        target=start_video_server,
        daemon=True
    ).start()

    Thread(
        target=stream_encoder,
        daemon=True
    ).start()

    # ========================================================
    # CAMERA
    # ========================================================

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open camera "
            f"index {CAMERA_INDEX}"
        )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        1920
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        1080
    )

    frame_counter = 0

    fps = 0.0
    fps_counter = 0

    fps_timer = (
        time.perf_counter()
    )

    print(
        "\nCamera running."
    )

    print(
        "Press Q to exit."
    )

    # ========================================================
    # LIVE LOOP
    # ========================================================

    while True:

        success, frame = cap.read()

        if not success:

            print(
                "Camera frame read failed."
            )

            break

        frame_counter += 1

        inference_start = (
            time.perf_counter()
        )

        # ====================================================
        # PERSON
        # ====================================================

        person_result = (
            person_model.track(
                frame,
                imgsz=IMG_SIZE,
                conf=PERSON_CONF,
                device=DEVICE,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )[0]
        )

        # ====================================================
        # VEHICLE
        # ====================================================

        vehicle_result = (
            vehicle_model.track(
                frame,
                imgsz=IMG_SIZE,
                conf=VEHICLE_CONF,
                device=DEVICE,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )[0]
        )

        # ====================================================
        # DRONE
        # ====================================================

        drone_result = (
            drone_model.track(
                frame,
                imgsz=IMG_SIZE,
                conf=DRONE_CONF,
                device=DEVICE,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )[0]
        )

        # ====================================================
        # WEAPON
        # ====================================================

        weapon_result = (
            weapon_model.track(
                frame,
                imgsz=IMG_SIZE,
                conf=WEAPON_CONF,
                device=DEVICE,
                tracker="bytetrack.yaml",
                persist=True,
                verbose=False,
            )[0]
        )

        inference_end = (
            time.perf_counter()
        )

        inference_ms = (
            inference_end
            -
            inference_start
        ) * 1000

        # ====================================================
        # DETECTIONS
        # ====================================================

        detections = []

        detections.extend(
            extract_detections(
                person_result,
                "person"
            )
        )

        detections.extend(
            extract_detections(
                vehicle_result,
                "vehicle"
            )
        )

        detections.extend(
            extract_detections(
                drone_result,
                "drone"
            )
        )

        detections.extend(
            extract_detections(
                weapon_result,
                "weapon"
            )
        )

        # ====================================================
        # VEHICLE / DRONE CONFIRMATION
        # ====================================================

        confirmed_objects = (
            update_object_confirmation(
                detections
            )
        )

        # ====================================================
        # WEAPON VISUAL SMOOTHING
        # ====================================================

        persistent_weapons = (
            smooth_weapon_display(
                detections
            )
        )

        # ====================================================
        # WEAPON ↔ PERSON
        # ====================================================

        weapon_associations = (
            associate_weapons_to_people(
                detections
            )
        )

        people = [
            d
            for d in detections
            if (
                d[
                    "source_model"
                ]
                ==
                "person"
                and
                d.get(
                    "track_id"
                )
                is not None
            )
        ]

        # ====================================================
        # PERSON CLASS SMOOTHING
        # ====================================================

        update_stable_classification(
            people
        )

        # ====================================================
        # WEAPON CONFIRMATION
        # ====================================================

        confirmed_weapons = (
            update_weapon_confirmation(
                people,
                weapon_associations
            )
        )

        for detection in detections:

            if (
                detection[
                    "source_model"
                ]
                !=
                "person"
            ):

                continue

            track_id = detection.get(
                "track_id"
            )

            if track_id is None:
                continue

            detection[
                "weapon_detected_now"
            ] = (
                track_id
                in
                weapon_associations
            )

            detection[
                "has_weapon"
            ] = confirmed_weapons.get(
                track_id,
                False
            )

        # ====================================================
        # THREAT ASSIGNMENT
        # ====================================================

        for detection in detections:

            if (
                detection[
                    "source_model"
                ]
                ==
                "person"
            ):

                detection[
                    "threat"
                ] = get_person_threat(
                    detection
                )

            else:

                detection[
                    "threat"
                ] = get_object_threat(
                    detection,
                    confirmed_objects
                )

        # ====================================================
        # SCENE THREAT
        # ====================================================

        scene_threat = (
            calculate_scene_threat(
                detections
            )
        )

        # ====================================================
        # DRAW
        # ====================================================

        display_frame = (
            frame.copy()
        )

        # Raw weapon boxes are skipped.
        for detection in detections:

            if (
                detection[
                    "source_model"
                ]
                ==
                "weapon"
            ):

                continue

            draw_detection(
                display_frame,
                detection
            )

        # Smooth weapon boxes are drawn instead.
        for weapon_detection in persistent_weapons:

            draw_detection(
                display_frame,
                weapon_detection
            )

        # ====================================================
        # EVENT MANAGER
        # ====================================================

        new_events = process_events(
            detections,
            display_frame,
            frame_counter
        )

        for event in new_events:

            print(
                f"[ALERT] "
                f"{event['event_id']} | "
                f"{event['classification']} | "
                f"{event['threat_level']}"
            )

        # ====================================================
        # COUNTS
        # ====================================================

        agent_count = 0
        non_agent_count = 0

        civilian_vehicle_count = 0
        military_vehicle_count = 0

        drone_count = 0
        raw_weapon_count = 0

        confirmed_armed_people = 0

        for detection in detections:

            name = detection[
                "class_name"
            ].lower()

            if name == "agent":

                agent_count += 1

            elif name == "non-agent":

                non_agent_count += 1

            elif name == "civilian_vehicle":

                civilian_vehicle_count += 1

            elif name == "military_vehicle":

                military_vehicle_count += 1

            elif name == "drone":

                drone_count += 1

            elif name == "weapon":

                raw_weapon_count += 1

            if (
                detection[
                    "source_model"
                ]
                ==
                "person"
                and
                detection.get(
                    "has_weapon",
                    False
                )
            ):

                confirmed_armed_people += 1

        # ====================================================
        # FPS
        # ====================================================

        fps_counter += 1

        now = (
            time.perf_counter()
        )

        elapsed = (
            now
            -
            fps_timer
        )

        if elapsed >= 1.0:

            fps = (
                fps_counter
                /
                elapsed
            )

            fps_counter = 0
            fps_timer = now

        # ====================================================
        # OVERLAY
        # ====================================================

        cv2.putText(
            display_frame,
            f"FPS: {fps:.1f}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display_frame,
            (
                f"AI latency: "
                f"{inference_ms:.1f} ms"
            ),
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display_frame,
            (
                f"Agent: {agent_count} | "
                f"Non-agent: "
                f"{non_agent_count}"
            ),
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display_frame,
            (
                f"Civilian vehicle: "
                f"{civilian_vehicle_count} | "
                f"Military vehicle: "
                f"{military_vehicle_count}"
            ),
            (20, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            display_frame,
            (
                f"Drone: {drone_count} | "
                f"Raw weapon: "
                f"{raw_weapon_count} | "
                f"Armed persons: "
                f"{confirmed_armed_people}"
            ),
            (20, 165),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        scene_color = (
            THREAT_COLORS[
                scene_threat
            ]
        )

        cv2.putText(
            display_frame,
            (
                f"SCENE THREAT: "
                f"{scene_threat}"
            ),
            (20, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            scene_color,
            3,
        )

        # ====================================================
        # DASHBOARD OUTPUT
        # ====================================================

        if frame_counter % 4 == 0:

            write_live_status(
                fps=fps,
                inference_ms=inference_ms,
                scene_threat=scene_threat,
                agent_count=agent_count,
                non_agent_count=non_agent_count,
                civilian_vehicle_count=civilian_vehicle_count,
                military_vehicle_count=military_vehicle_count,
                drone_count=drone_count,
                raw_weapon_count=raw_weapon_count,
                confirmed_armed_people=confirmed_armed_people,
                camera_connected=True
            )

        #write_live_frame(display_frame, frame_counter)


        global latest_stream_frame

        with stream_lock:
            latest_stream_frame = display_frame

        # ====================================================
        # DISPLAY
        # ====================================================

        # No local OpenCV preview.
        # The Flask dashboard is the presentation interface.
        time.sleep(0.001)

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()
    cv2.destroyAllWindows()

    print(
        "\nPipeline stopped."
    )


if __name__ == "__main__":
    main()