"""Reading from a camera: a USB webcam by index, or an IP camera over RTSP.

What to open comes from a camera entry of the saved list (camera_store.py).

Each Camera reads on its own thread (as Aicam-detection-Pi5 does): an IP
stream that stalls blocks for its read timeout, and reopening one waits for
its open timeout -- neither may freeze the display. The pipeline only ever
waits a fraction of a second for a new frame.
"""

import base64
import os
import threading
import time

import cv2

from settings import CAMERA_FPS
from sources.camera_store import redact, source

# DirectShow, not Media Foundation: MSMF reports 30 fps on this webcam but pads
# it with duplicate frames -- DirectShow delivered more unique frames per second.
_WEBCAM_BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY

# RTSP over TCP: UDP loses packets on a busy network and smears the picture.
# OpenCV's FFmpeg backend reads this on every open.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
_IP_TIMEOUTS = [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 8000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000]

RETRY_SECONDS = 5.0      # no frames this long: reopen the source
NO_SIGNAL_SECONDS = 1.0  # no frames this long: report the camera as down
FIRST_FRAME_SECONDS = 12.0
MAX_WEBCAM_INDEX = 6     # indices scanned by the web UI


class Camera:
    """One camera entry, read on a background thread; hands out mirrored frames
    at the entry's resolution. start() it, read() from it, stop() it -- once."""

    def __init__(self, cam, verbose=True):
        self.cam = cam
        self.source = source(cam)
        self.resolution = tuple(cam["resolution"])
        self.verbose = verbose
        self.label = cam["name"]
        self.size = None     # (width, height) the source actually delivers
        self.opened = None   # None: not tried yet; then whether the last open worked
        self._capture = None
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._taken = 0
        self._last_frame_at = time.monotonic()
        self._running = False
        self._thread = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        """Stop reading and release the device (waits for a read in progress)."""
        self._running = False
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=10)

    def read(self, timeout=0.1):
        """The newest frame not handed out yet, or None within `timeout`."""
        with self._cond:
            if self._seq == self._taken:
                self._cond.wait(timeout)
            if self._seq == self._taken:
                return None
            self._taken = self._seq
            return self._frame

    def snapshot(self):
        """The latest frame, whether or not read() has handed it out."""
        with self._cond:
            return self._frame

    @property
    def has_signal(self):
        return time.monotonic() - self._last_frame_at < NO_SIGNAL_SECONDS

    # -- capture thread -------------------------------------------------------

    def _loop(self):
        next_open = 0.0
        while self._running:
            if self._capture is None:
                if time.monotonic() < next_open:
                    time.sleep(0.1)
                    continue
                if not self._open():
                    # From the end of the attempt: a dead IP camera's open alone
                    # takes the whole open timeout.
                    next_open = time.monotonic() + RETRY_SECONDS
                    continue

            ok, frame = self._capture.read()
            now = time.monotonic()
            if ok and frame is not None:
                frame = cv2.flip(self._fit(frame), 1)
                with self._cond:
                    self._frame, self._seq, self._last_frame_at = frame, self._seq + 1, now
                    self._cond.notify_all()
            elif now - self._last_frame_at >= RETRY_SECONDS:
                if self.verbose:
                    print(f"No picture from camera '{self.label}', reopening...")
                self._release()
            else:
                time.sleep(0.01)
        self._release()

    def _open(self):
        if self.cam["type"] == "webcam":
            capture = cv2.VideoCapture(self.source, _WEBCAM_BACKEND)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            # A deep driver buffer means every read() returns a stale frame; 1
            # keeps the loop on live pixels. Not every backend honours it.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            capture = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, _IP_TIMEOUTS)

        self.opened = capture.isOpened()
        if not self.opened:
            capture.release()
            if self.verbose:
                print(f"Could not open camera '{self.label}' ({self._where()})")
            return False

        self._capture = capture
        self._last_frame_at = time.monotonic()  # give it RETRY_SECONDS to deliver
        self.size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self.verbose:
            print(f"Camera '{self.label}' ({self._where()}) open at {self.size[0]}x{self.size[1]}")
        return True

    def _where(self):
        return f"webcam {self.source}" if self.cam["type"] == "webcam" else redact(self.source)

    def _release(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _fit(self, frame):
        """Crop the middle to 16:9 rather than squash a 4:3 camera, then resize."""
        height, width = frame.shape[:2]
        target_w, target_h = self.resolution
        if width * target_h > height * target_w:
            crop = height * target_w // target_h
            x = (width - crop) // 2
            frame = frame[:, x:x + crop]
        elif width * target_h < height * target_w:
            crop = width * target_h // target_w
            y = (height - crop) // 2
            frame = frame[y:y + crop]
        if (frame.shape[1], frame.shape[0]) != self.resolution:
            frame = cv2.resize(frame, self.resolution, interpolation=cv2.INTER_AREA)
        return frame


def first_frame(camera, timeout=FIRST_FRAME_SECONDS):
    """Wait for a started camera's first frame; None if it doesn't come, or as
    soon as the source fails to open. isOpened() alone proves nothing."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        frame = camera.read(timeout=0.2)
        if frame is not None:
            return frame
        if camera.opened is False:
            return None
    return None


def thumbnail(frame):
    thumb = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
    _, jpeg = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def probe(cam):
    """Start a camera entry, wait for one frame and stop it again. For the web
    UI's scan and test buttons; returns a small JPEG thumbnail when it works."""
    camera = Camera(cam, verbose=False).start()
    try:
        frame = first_frame(camera)
        if frame is None:
            return {"ok": False, "error": "no picture from it (check the IP, port, login, "
                                          "channel and brand, and that the camera is reachable)"
                    if cam["type"] == "ip" else "no picture from it"}
        width, height = camera.size
        return {"ok": True, "width": width, "height": height, "thumbnail": thumbnail(frame)}
    finally:
        camera.stop()  # never keep hold of a device someone may pick next


def scan_webcams(in_use=None):
    """Every webcam index that delivers a frame. `in_use` is the running Camera:
    its device can't be opened a second time, so it is listed from its own
    latest frame instead."""
    busy = in_use.source if in_use is not None and in_use.cam["type"] == "webcam" else None
    found = []
    for index in range(MAX_WEBCAM_INDEX):
        if index == busy:
            frame = in_use.snapshot()
            found.append({"index": index, "in_use": True,
                          "thumbnail": thumbnail(frame) if frame is not None else None})
            continue
        # A missing index fails to open at once. The resolution asked for here
        # only affects the thumbnail.
        result = probe({"type": "webcam", "name": f"Webcam {index}", "index": index,
                        "resolution": [640, 360]})
        if result["ok"]:
            found.append({"index": index, "in_use": False, "thumbnail": result["thumbnail"]})
    return found
