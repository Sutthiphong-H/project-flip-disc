"""The capture -> segment -> publish loop, and the mode it runs in.

Two modes:
    active      someone is within MAX_PERSON_DISTANCE_M; their silhouette is shown
    idle_video  nobody in range for IDLE_TIMEOUT_SECONDS; the idle clip loops instead

Segmentation runs in both modes -- it is also what notices someone walking up.
"""

import time

import torch

import settings
from frame_rate import FrameRateMeter
from sources.idle_video import IdleVideo
from sources.webcam import Webcam
from vision.models import load_model
from vision.segmenter import Segmenter

MAX_READ_FAILURES = 30  # consecutive failed reads before reopening the camera


def select_device():
    """Return (device, use_fp16)."""
    if torch.cuda.is_available():
        return "cuda:0", True
    return "cpu", False


class FlipdiscPipeline:
    def __init__(self, publish):
        """`publish` is called with the payload dict for every emitted frame."""
        self.publish = publish
        self.device, self.use_fp16 = select_device()
        self.fps_meter = FrameRateMeter()

        self.mode = "idle_video"
        self.people = []

        self.camera = None
        self.segmenter = None
        self.idle_video = None
        self._running = False

    # -- lifecycle -----------------------------------------------------------

    def setup(self):
        print(f"Using device: {self.device}, FP16: {self.use_fp16}")
        if self.device.startswith("cuda"):
            # Leave cudnn.benchmark off: on this GPU its first-run autotune took
            # 33 s of startup and picked kernels that were no faster.
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        print(f"Segmentation model: {settings.SEGMENTATION_MODEL} "
              f"at {settings.PROCESS_RESOLUTION[0]}x{settings.PROCESS_RESOLUTION[1]}")
        self.segmenter = Segmenter(load_model(self.device, self.use_fp16)).start()
        self.idle_video = IdleVideo(settings.IDLE_VIDEO_PATH)

        self.camera = Webcam()
        opened = self.camera.open()
        self.segmenter.wait_ready()
        return opened

    def teardown(self):
        if self.segmenter:
            self.segmenter.stop()
        if self.camera:
            self.camera.release()

    # -- main loop -----------------------------------------------------------

    def run(self):
        if not self.setup():
            return

        self._running = True
        last_person_seen = time.monotonic() - settings.IDLE_TIMEOUT_SECONDS
        next_emit = 0.0
        read_failures = 0
        mask, preview = None, None

        while self._running:
            frame = self.camera.read()
            if frame is None:
                read_failures += 1
                if read_failures >= MAX_READ_FAILURES:
                    print("Camera stopped delivering frames, reopening...")
                    self.camera.release()
                    self.camera.open()
                    read_failures = 0
                time.sleep(0.01)
                continue

            read_failures = 0
            now = time.monotonic()

            self.segmenter.submit(frame)
            result = self.segmenter.poll()
            if result is not None:
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

            if mask is not None and now >= next_emit:
                # Step on a fixed grid rather than "interval since last emit":
                # with the camera at the same rate as the cap, frames arriving
                # a millisecond early would otherwise be dropped (30 -> 20 fps).
                next_emit = max(next_emit + settings.EMIT_INTERVAL, now - settings.EMIT_INTERVAL)
                self.fps_meter.tick(now)
                self._emit(mask, preview, idle_for)
                mask, preview = None, None  # never re-send a stale frame

        self.teardown()

    def stop(self):
        self._running = False

    # -- steps ---------------------------------------------------------------

    def _anyone_in_range(self):
        return any(p.distance_m <= settings.MAX_PERSON_DISTANCE_M for p in self.people)

    def _nearest_m(self):
        return min((p.distance_m for p in self.people), default=None)

    def _emit(self, mask, preview, idle_for):
        nearest = self._nearest_m()
        self.publish({
            "matrix": (mask > 0).astype(int).tolist(),
            "mode": self.mode,
            "fps": self.fps_meter.get_fps(),
            "people": len(self.people),
            "nearest_m": None if nearest is None else round(float(nearest), 2),
            "time_since_person": round(idle_for, 1),
        }, preview)
