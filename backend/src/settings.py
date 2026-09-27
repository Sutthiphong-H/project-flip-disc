"""Every tunable in one place. Override any of them with environment variables."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_flag(name, default="1"):
    return os.environ.get(name, default) == "1"


# --- Camera -----------------------------------------------------------------
CAMERA_INDEX = _env_int("FLIPDISC_CAMERA", 0)
CAMERA_FPS = _env_int("FLIPDISC_CAMERA_FPS", 30)

# --- Resolutions ------------------------------------------------------------
INPUT_RESOLUTION = (320, 240)    # frames grabbed from the camera
PROCESS_RESOLUTION = (160, 120)  # what U2NET actually sees
FLIPDISC_RESOLUTION = (80, 45)   # must match cols/rows in Flipdot.jsx

# --- Models -----------------------------------------------------------------
U2NET_WEIGHTS = MODELS_DIR / "u2net_human_seg.pth"
YOLO_WEIGHTS = MODELS_DIR / "yolo12n.pt"

# --- Detection / mode switching ---------------------------------------------
DETECT_INTERVAL = _env_int("FLIPDISC_DETECT_INTERVAL", 15)  # run YOLO every N frames
PERSON_CONFIDENCE_THRESHOLD = _env_float("FLIPDISC_CONFIDENCE", 0.75)
MAX_PERSON_DISTANCE_M = _env_float("FLIPDISC_MAX_DISTANCE", 15.0)
IDLE_TIMEOUT_SECONDS = _env_float("FLIPDISC_IDLE_TIMEOUT", 15.0)

# Rough monocular distance estimate. FOCAL_LENGTH_PIXELS was calibrated for the
# old ESP32-CAM -- re-measure it for the current webcam before trusting the
# metres it reports (point a person of known height at a known distance and
# solve focal = pixel_height * distance / real_height).
FOCAL_LENGTH_PIXELS = _env_float("FLIPDISC_FOCAL_LENGTH", 1422.22)
REAL_PERSON_HEIGHT_M = 1.7

# --- Idle video -------------------------------------------------------------
IDLE_VIDEO_PATH = ASSETS_DIR / os.environ.get("FLIPDISC_IDLE_VIDEO", "video.mp4")
IDLE_VIDEO_THRESHOLD = _env_int("FLIPDISC_IDLE_THRESHOLD", 75)

# --- Output -----------------------------------------------------------------
HOST = os.environ.get("FLIPDISC_HOST", "0.0.0.0")
PORT = _env_int("FLIPDISC_PORT", 5000)
EMIT_INTERVAL = 1.0 / _env_float("FLIPDISC_EMIT_FPS", 20.0)
DEBUG_WINDOWS = _env_flag("FLIPDISC_DEBUG")
