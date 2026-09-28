"""Every tunable in one place. Override any of them with environment variables."""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ASSETS_DIR = BASE_DIR / "assets"
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR = BASE_DIR / ".cache"
# `npm run build` output, served at / so the built frontend needs no Vite.
FRONTEND_DIST = Path(os.environ.get("FLIPDISC_FRONTEND_DIST",
                                    BASE_DIR.parent / "flip-disc_frontend" / "dist"))


def _env_int(name, default):
    return int(os.environ.get(name, default))


def _env_float(name, default):
    return float(os.environ.get(name, default))


def _env_size(name, default):
    width, height = os.environ.get(name, default).lower().split("x")
    return int(width), int(height)


# --- Camera -----------------------------------------------------------------
CAMERA_INDEX = _env_int("FLIPDISC_CAMERA", 0)  # the first webcam, until cameras.json exists
# The cameras set up in the web UI (webcams and IP cameras) and which one is
# active. Per machine and holds IP camera passwords, so it is not committed.
CAMERAS_PATH = Path(os.environ.get("FLIPDISC_CAMERAS", BASE_DIR / "cameras.json"))
CAMERA_FPS = _env_int("FLIPDISC_CAMERA_FPS", 30)

# --- Resolutions ------------------------------------------------------------
# --- Model ------------------------------------------------------------------
# "auto": the NVIDIA GPU when there is one that works, otherwise the CPU (RVM
# ~42 ms a frame there instead of ~2 ms). "cuda" or "cpu" to insist on one.
DEVICE = os.environ.get("FLIPDISC_DEVICE", "auto").lower()
# "rvm" (default) or "u2net" -- see vision/models.py for the trade-off.
SEGMENTATION_MODEL = os.environ.get("FLIPDISC_MODEL", "rvm").lower()
RVM_WEIGHTS = MODELS_DIR / "rvm_mobilenetv3.pth"
U2NET_WEIGHTS = MODELS_DIR / "u2net_human_seg.pth"
# RVM's backbone runs at the process size (below) * this; its guided filter
# then refines the edges at the full process size.
RVM_DOWNSAMPLE = _env_float("FLIPDISC_RVM_DOWNSAMPLE", 0.5)

# --- Resolutions ------------------------------------------------------------
# All 16:9 like the display, so nothing gets squashed on the way down.
# The model's input size depends on the model and on the device, see
# process_resolution() below.
# Each camera has its own resolution (set in the web UI); this is the default
# for the first webcam. 1280x720 because this webcam switches to MJPEG there and
# delivers 20 fps, against 15 fps of YUY2 at 640x360. Scaling it down for the
# model is ~1 ms.
INPUT_RESOLUTION = (1280, 720)
FLIPDISC_RESOLUTION = (80, 45)   # must match cols/rows in Flipdot.jsx

# The model's input size. RVM on a GPU gets 1280x720, so its backbone sees
# 640x360: on the webcam against a dark background it kept both raised hands
# and their fingers, where 640x360 (backbone 320x180) lost a whole hand. That
# takes 4 ms as a CUDA graph. On the CPU it would be far too slow, so RVM gets
# 640x360 there (~42 ms). U2NET gets 512x288: at 640x360 it picks up specks (it
# was trained at 320). FLIPDISC_PROCESS_SIZE overrides all of these.
_DEFAULT_PROCESS_SIZE = {("rvm", True): "1280x720", ("rvm", False): "640x360",
                         ("u2net", True): "512x288", ("u2net", False): "512x288"}


def process_resolution(model, on_gpu):
    return _env_size("FLIPDISC_PROCESS_SIZE", _DEFAULT_PROCESS_SIZE[model, on_gpu])


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
# Fingers are narrower than a disc until you are about 1.5 m from the camera,
# so they never cover DISC_ON of one. Parts of the silhouette thinner than a
# disc (fingers, a thin arm) count this many times extra towards coverage; the
# body, and its edges, are unaffected. 0 turns it off.
DISC_THIN_BOOST = _env_float("FLIPDISC_THIN_BOOST", 2.0)
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

# --- Watchdog ---------------------------------------------------------------
# Frames go out in every mode (the idle clip plays even without a camera), so
# none for STALL_SECONDS means the pipeline is stuck or dead. The backend then
# exits with code 1 and run-backend.bat starts it again. STARTUP_SECONDS allows
# for the first start, which decodes the idle clip and loads the model.
STALL_SECONDS = _env_float("FLIPDISC_STALL_SECONDS", 10.0)
STARTUP_SECONDS = _env_float("FLIPDISC_STARTUP_SECONDS", 120.0)
