"""Every tunable in one place. Override any of them with environment variables."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / ".cache"


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_size(name, default):
    width, height = os.environ.get(name, default).lower().split("x")
    return int(width), int(height)


# --- Camera -----------------------------------------------------------------
CAMERA_INDEX = _env_int("FLIPDISC_CAMERA", 0)
CAMERA_FPS = _env_int("FLIPDISC_CAMERA_FPS", 30)

# --- Resolutions ------------------------------------------------------------
# --- Model ------------------------------------------------------------------
# "rvm" (default) or "u2net" -- see vision/models.py for the trade-off.
SEGMENTATION_MODEL = os.environ.get("FLIPDISC_MODEL", "rvm").lower()
RVM_WEIGHTS = MODELS_DIR / "rvm_mobilenetv3.pth"
U2NET_WEIGHTS = MODELS_DIR / "u2net_human_seg.pth"
# RVM's backbone runs at PROCESS_RESOLUTION * this; 0.5 of 640x360 kept the
# fingers that 512x288 at 1.0 lost, at the same ~14 ms.
RVM_DOWNSAMPLE = _env_float("FLIPDISC_RVM_DOWNSAMPLE", 0.5)

# --- Resolutions ------------------------------------------------------------
# All 16:9 like the display, so nothing gets squashed on the way down.
# PROCESS_RESOLUTION is the model's input size. Defaults are what looked best on
# this webcam: RVM at full camera size (downsampled internally, above); U2NET at
# 512x288 -- at 640x360 it picks up specks (it was trained at 320).
# 1280x720 input: this webcam switches to MJPEG there and delivers 20 fps,
# against 15 fps of YUY2 at 640x360. Scaling it down for the model is ~1 ms.
INPUT_RESOLUTION = (1280, 720)   # frames grabbed from the camera
_DEFAULT_PROCESS_SIZE = {"rvm": "640x360", "u2net": "512x288"}
PROCESS_RESOLUTION = _env_size("FLIPDISC_PROCESS_SIZE",
                               _DEFAULT_PROCESS_SIZE.get(SEGMENTATION_MODEL, "512x288"))
FLIPDISC_RESOLUTION = (80, 45)   # must match cols/rows in Flipdot.jsx

# --- People / mode switching ------------------------------------------------
MASK_THRESHOLD = _env_float("FLIPDISC_MASK_THRESHOLD", 0.5)  # model output is 0..1
MIN_PERSON_AREA = _env_float("FLIPDISC_MIN_AREA", 0.005)  # share of the frame; smaller blobs are noise
MAX_PERSON_DISTANCE_M = _env_float("FLIPDISC_MAX_DISTANCE", 15.0)

# Anti-flicker. Each disc tracks how much of it the silhouette covers (0..1),
# smoothed over frames, and only turns on above DISC_ON / off below DISC_OFF.
# Measured on the webcam: flicker on a still person drops ~96% for one frame
# (~70 ms) of extra lag. DISC_SMOOTHING=1 turns the smoothing off; 0.4 with
# 0.7/0.3 removes the rest of the flicker at two frames of lag.
DISC_SMOOTHING = _env_float("FLIPDISC_SMOOTHING", 0.5)  # weight of the newest frame
DISC_ON = _env_float("FLIPDISC_DISC_ON", 0.65)
DISC_OFF = _env_float("FLIPDISC_DISC_OFF", 0.35)
IDLE_TIMEOUT_SECONDS = _env_float("FLIPDISC_IDLE_TIMEOUT", 15.0)

# Each silhouette's distance is estimated from its height:
#     distance = REAL_PERSON_HEIGHT_M * FOCAL_LENGTH_PIXELS / height_px
# FOCAL_LENGTH_PIXELS is in pixels of a 360-px-tall frame, whatever the actual
# input and model resolutions are (the h=...px in the Camera view uses the same
# pixels), so changing resolution doesn't change the distances. The default assumes a
# ~70 degree webcam and is NOT calibrated -- stand a person of known height at
# a known distance, read their h=...px from the Camera view in the web UI, and set
# focal = height_px * distance / real_height. Only accurate when the whole
# body is in frame; someone cut off at the edge reads as further away.
FOCAL_LENGTH_PIXELS = _env_float("FLIPDISC_FOCAL_LENGTH", 460.0)
FOCAL_REFERENCE_HEIGHT = 360
REAL_PERSON_HEIGHT_M = 1.7

# --- Idle video -------------------------------------------------------------
IDLE_VIDEO_PATH = ASSETS_DIR / os.environ.get("FLIPDISC_IDLE_VIDEO", "video.mp4")
IDLE_VIDEO_THRESHOLD = _env_int("FLIPDISC_IDLE_THRESHOLD", 75)

# --- Output -----------------------------------------------------------------
HOST = os.environ.get("FLIPDISC_HOST", "0.0.0.0")
PORT = _env_int("FLIPDISC_PORT", 5000)
EMIT_INTERVAL = 1.0 / _env_float("FLIPDISC_EMIT_FPS", 30.0)
