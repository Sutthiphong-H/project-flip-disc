"""The capture -> detect -> segment -> publish loop, and the mode it runs in.

Two modes:
    active      a person is in front of the camera; their silhouette is shown
    idle_video  nobody for IDLE_TIMEOUT_SECONDS; the idle clip loops instead
"""

import time

import cv2
import numpy as np
import torch

import settings
from frame_rate import FrameRateMeter
from sources.idle_video import IdleVideo
from sources.webcam import Webcam
from vision.detector import PersonDetector
from vision.segmenter import Segmenter, load_u2net

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
        self.blackout = True
        self.person_boxes = []
        self.confidence = 0.0
        self.frame_count = 0

        self.camera = None
        self.detector = None
        self.segmenter = None
        self.idle_video = None
        self._running = False

    # -- lifecycle -----------------------------------------------------------

    def setup(self):
        print(f"Using device: {self.device}, FP16: {self.use_fp16}")
        if self.device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        self.detector = PersonDetector(self.device)
        self.segmenter = Segmenter(
            load_u2net(self.device, self.use_fp16),
            self.device,
            self.use_fp16,
            with_preview=settings.DEBUG_WINDOWS,
        ).start()
        self.idle_video = IdleVideo(settings.IDLE_VIDEO_PATH)

        self.camera = Webcam()
        if not self.camera.open():
            return False

        if settings.DEBUG_WINDOWS:
            cv2.namedWindow("Debug View", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Flip Disc Preview", cv2.WINDOW_NORMAL)
        return True

    def teardown(self):
        if self.segmenter:
            self.segmenter.stop()
        if self.camera:
            self.camera.release()
        if settings.DEBUG_WINDOWS:
            cv2.destroyAllWindows()

    # -- main loop -----------------------------------------------------------

    def run(self):
        if not self.setup():
            return

        self._running = True
        blackout_mask = np.zeros(settings.INPUT_RESOLUTION[::-1], dtype=np.uint8)
        last_person_seen = time.monotonic() - settings.IDLE_TIMEOUT_SECONDS
        last_emit = 0.0
        read_failures = 0

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
            self.frame_count += 1
            now = time.monotonic()

            if self.frame_count % settings.DETECT_INTERVAL == 0:
                in_range = self._detect(frame, blackout_mask)
                if in_range:
                    last_person_seen = now
                    if self.mode != "active":
                        print("Person detected - switching to active mode")
                        self.mode = "active"

            idle_for = now - last_person_seen
            if idle_for >= settings.IDLE_TIMEOUT_SECONDS and self.mode != "idle_video":
                print("Timeout reached - switching to idle mode")
                self.mode = "idle_video"
                self.blackout = True

            mask, preview = self._next_mask(frame, blackout_mask)
            if mask is not None and (now - last_emit) >= settings.EMIT_INTERVAL:
                last_emit = now
                self.fps_meter.tick(now)
                self._emit(mask, preview, idle_for)

            if settings.DEBUG_WINDOWS and cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.teardown()

    def stop(self):
        self._running = False

    # -- steps ---------------------------------------------------------------

    def _detect(self, frame, blackout_mask):
        """Run detection and refresh the blackout mask. Returns True if someone is in range."""
        try:
            self.person_boxes = self.detector.detect(frame)
        except Exception as e:
            print(f"Detection error: {e}")
            return False

        self.confidence = max((b.confidence for b in self.person_boxes), default=0.0)
        in_range = any(b.distance_m <= settings.MAX_PERSON_DISTANCE_M
                       for b in self.person_boxes)

        self.blackout = not in_range
        blackout_mask.fill(0 if self.blackout else 255)
        if not self.blackout:
            # Someone is close enough to drive the display; blank out anyone
            # standing further back than the cutoff so they don't bleed in.
            for box in self.person_boxes:
                if box.distance_m > settings.MAX_PERSON_DISTANCE_M:
                    cv2.rectangle(blackout_mask, (box.x1, box.y1), (box.x2, box.y2), 0, -1)
        return in_range

    def _next_mask(self, frame, blackout_mask):
        """Return (flipdisc_mask, preview) for this frame, or (None, None) if not ready."""
        if self.mode == "idle_video":
            return self.idle_video.next_mask(), None

        self.segmenter.submit(cv2.bitwise_and(frame, frame, mask=blackout_mask))
        result = self.segmenter.poll()
        if result is None:
            return None, None
        return result.flipdisc_mask, result.preview

    def _emit(self, mask, preview, idle_for):
        fps = self.fps_meter.get_fps()

        if settings.DEBUG_WINDOWS:
            self._show_debug(mask, preview, fps, idle_for)

        self.publish({
            "matrix": (mask > 0).astype(int).tolist(),
            "mode": self.mode,
            "blackout": self.blackout,
            "fps": fps,
            "confidence": round(self.confidence, 2),
            "time_since_person": round(idle_for, 1),
        })

    def _show_debug(self, mask, preview, fps, idle_for):
        width, height = settings.FLIPDISC_RESOLUTION
        cv2.imshow("Flip Disc Preview",
                   cv2.resize(mask, (width * 8, height * 8), interpolation=cv2.INTER_NEAREST))

        if preview is None:
            return

        colour = (0, 255, 0) if self.mode == "active" else (0, 0, 255)
        lines = (f"Mode: {self.mode}", f"Conf: {self.confidence:.2f}",
                 f"Idle: {int(idle_for)}s", f"FPS: {fps:.1f}")
        for row, line in enumerate(lines, start=1):
            cv2.putText(preview, line, (10, 30 * row),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

        for box in self.person_boxes:
            if box.distance_m == float("inf"):
                continue
            box_colour = (0, 255, 0) if box.distance_m <= settings.MAX_PERSON_DISTANCE_M else (0, 0, 255)
            cv2.rectangle(preview, (box.x1, box.y1), (box.x2, box.y2), box_colour, 1)
            cv2.putText(preview, f"{box.distance_m:.1f}m", (box.x1, box.y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_colour, 1)

        cv2.imshow("Debug View", preview)
