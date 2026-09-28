"""The capture -> segment -> publish loop, and the mode it runs in.

Two modes:
    active      someone is within MAX_PERSON_DISTANCE_M; their silhouette is shown
    idle_video  nobody in range for IDLE_TIMEOUT_SECONDS; the idle clip loops instead

Segmentation runs in both modes -- it is also what notices someone walking up.
With no picture from the camera the idle clip keeps playing (the camera
retries on its own thread) and the web UI can still switch to another one.
"""

import threading
import time

import torch

import settings
from frame_rate import FrameRateMeter
from sources.camera import Camera, first_frame
from sources.camera_store import CameraStore, hard_signature
from sources.idle_video import IdleVideo
from vision.models import load_model
from vision.segmenter import Segmenter


def select_device():
    """Return (device, use_fp16): an NVIDIA GPU if there is one that works, else the CPU.

    torch.cuda.is_available() alone isn't proof. This torch build (cu128) only
    has kernels for sm_75 (RTX 20-series) and newer; on an older card it still
    says True and then fails on the first kernel. So run one.
    """
    if settings.DEVICE not in ("auto", "cuda", "cpu"):
        raise ValueError(f"FLIPDISC_DEVICE must be auto, cuda or cpu, got {settings.DEVICE!r}")
    if settings.DEVICE == "cpu":
        return "cpu", False
    if not torch.cuda.is_available():
        reason = "no CUDA GPU found"
    else:
        try:
            name = torch.cuda.get_device_name(0)
            (torch.ones(8, device="cuda:0") * 2).sum().item()
            # Leave cudnn.benchmark off: on the RTX 5060 its first-run autotune
            # took 33 s of startup and picked kernels that were no faster.
            print(f"GPU: {name}")
            return "cuda:0", True
        except Exception as e:
            reason = f"the GPU failed a test ({e})"
    if settings.DEVICE == "cuda":
        raise RuntimeError(f"FLIPDISC_DEVICE=cuda, but {reason}")
    print(f"Running on the CPU: {reason}")
    return "cpu", False


class FlipdiscPipeline:
    def __init__(self, publish):
        """`publish(discs, status, preview)` is called for every emitted frame."""
        self.publish = publish
        self.device, self.use_fp16 = select_device()
        self.fps_meter = FrameRateMeter()

        self.mode = "idle_video"
        self.people = []

        self.cameras = CameraStore()  # the web UI edits it from the start
        self.camera = None            # the running Camera, from setup() on
        self.segmenter = None
        self.idle_video = None
        self._stopping = False  # set by stop(), even one that comes during setup()
        self._next_emit = 0.0
        self._switch_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def setup(self):
        print(f"Using device: {self.device}, FP16: {self.use_fp16}")
        # The model warms up on the segmenter's thread while the rest starts.
        try:
            self.segmenter = Segmenter(load_model(self.device, self.use_fp16)).start()
        except Exception as e:
            self._fall_back_to_cpu(e)
        self.idle_video = IdleVideo(settings.IDLE_VIDEO_PATH)

        # A camera that won't open doesn't stop the pipeline: the loop plays the
        # idle clip, the camera keeps retrying, and the web UI can pick another.
        self.camera = Camera(self.cameras.active).start()
        try:
            self.segmenter.wait_ready()
        except Exception as e:
            self._fall_back_to_cpu(e)
            self.segmenter.wait_ready()
        width, height = self.segmenter.model.size
        print(f"Segmentation model: {settings.SEGMENTATION_MODEL} at {width}x{height} on the {self.device}")

    def _fall_back_to_cpu(self, error):
        """Start the segmenter on the CPU after the GPU failed to load or warm up
        the model (out of memory, a missing kernel). Restarting wouldn't help:
        the next process would fail the same way."""
        if self.device == "cpu" or settings.DEVICE == "cuda":
            raise error
        print(f"The model failed on the GPU ({error}); running it on the CPU instead")
        self.device, self.use_fp16 = "cpu", False
        self.segmenter = Segmenter(load_model(self.device, self.use_fp16)).start()

    def teardown(self):
        if self.segmenter:
            self.segmenter.stop()
        if self.camera:
            self.camera.stop()
            print("Camera released")

    # -- main loop -----------------------------------------------------------

    def run(self):
        """Run until stop(). An error ends it too -- the camera still released --
        and app.py's watchdog then exits so the backend gets restarted: a model
        that won't load or a broken CUDA context doesn't fix itself."""
        try:
            self.setup()
            self._loop()
        finally:
            self.teardown()

    def _loop(self):
        last_person_seen = time.monotonic() - settings.IDLE_TIMEOUT_SECONDS
        mask, preview = None, None
        waiting_since = None  # oldest frame handed to the segmenter without an answer
        gap = False

        while not self._stopping:
            # Wait no longer than one emit interval, so the idle clip keeps its
            # pace when the camera is down.
            frame = self.camera.read(timeout=settings.EMIT_INTERVAL)
            now = time.monotonic()
            if frame is None:
                if not self.camera.has_signal:
                    gap = True
                    self._without_camera(now, last_person_seen)
                continue
            if gap:
                gap = False
                self.segmenter.reset()

            self.segmenter.submit(frame)
            waiting_since = waiting_since or now
            result = self.segmenter.poll()
            if result is None:
                if now - waiting_since > settings.STALL_SECONDS:
                    raise RuntimeError(f"no segmentation result for {now - waiting_since:.0f} s")
            else:
                waiting_since = None
                self.people = result.people
                mask, preview = result.flipdisc_mask, result.preview
                if self._anyone_in_range():
                    last_person_seen = now
                    if self.mode != "active":
                        print("Person in range - switching to active mode")
                        self.mode = "active"

            idle_for = now - last_person_seen
            if idle_for >= settings.IDLE_TIMEOUT_SECONDS and self.mode != "idle_video":
                print("Timeout reached - switching to idle mode")
                self.mode = "idle_video"

            if self.mode == "idle_video":
                # The camera preview stays live so the debug view shows who walks up.
                mask = self.idle_video.next_mask()

            if mask is not None and self._emit_due(now):
                self._emit(mask, preview, idle_for, now)
                mask, preview = None, None  # never re-send a stale frame

    def stop(self):
        self._stopping = True

    # -- camera --------------------------------------------------------------

    def switch_camera(self, cam):
        """Run a (possibly edited) camera entry, from a web request thread.

        A new source starts next to the old one and replaces it only once a
        frame comes through, so the display doesn't stall and a camera that
        doesn't work leaves the current one running. If only the name changed
        nothing restarts. Returns (ok, error or None).
        """
        with self._switch_lock:
            old = self.camera
            if hard_signature(cam) == hard_signature(old.cam):
                old.cam, old.label = cam, cam["name"]
                return True, None
            if cam["type"] == "webcam" and old.cam["type"] == "webcam" and cam["index"] == old.cam["index"]:
                old.stop()  # same device, new resolution: it can't be open twice
                new = Camera(cam).start()
                if first_frame(new) is None:
                    new.stop()
                    self.camera = Camera(old.cam).start()
                    return False, f"'{cam['name']}' sent no picture at that resolution; kept the old settings"
            else:
                new = Camera(cam).start()
                if first_frame(new) is None:
                    new.stop()
                    return False, f"'{cam['name']}' sent no picture; still using '{old.label}'"
                old.stop()  # frees the device for anyone else
            self.camera = new
            self.segmenter.reset()
            print(f"Switched camera: '{old.label}' -> '{new.label}'")
            return True, None

    def _without_camera(self, now, last_person_seen):
        """No picture: keep the idle clip playing on the display."""
        self.people = []
        self.mode = "idle_video"
        if self._emit_due(now):
            self._emit(self.idle_video.next_mask(), None, now - last_person_seen, now)

    # -- steps ---------------------------------------------------------------

    def _emit_due(self, now):
        if now < self._next_emit:
            return False
        # Step on a fixed grid rather than "interval since last emit": with the
        # camera at the same rate as the cap, frames arriving a millisecond
        # early would otherwise be dropped (30 -> 20 fps).
        self._next_emit = max(self._next_emit + settings.EMIT_INTERVAL,
                              now - settings.EMIT_INTERVAL)
        return True

    def _anyone_in_range(self):
        return any(p.distance_m <= settings.MAX_PERSON_DISTANCE_M for p in self.people)

    def _nearest_m(self):
        return min((p.distance_m for p in self.people), default=None)

    def _emit(self, mask, preview, idle_for, now):
        self.fps_meter.tick(now)
        nearest = self._nearest_m()
        self.publish(mask > 0, {
            "mode": self.mode,
            "fps": self.fps_meter.get_fps(),
            "people": len(self.people),
            "nearest_m": None if nearest is None else round(float(nearest), 2),
            "time_since_person": round(idle_for, 1),
            "camera_ok": self.camera.has_signal,
            "device": "GPU" if self.device.startswith("cuda") else "CPU",
        }, preview)
