"""Rolling frame-rate meter.

Ported from Aicam-detection-Pi5. Measuring 1/(now - previous) reports the gap
between two arbitrary frames, which swings wildly; averaging over a window
reports what the pipeline is actually sustaining.
"""

import threading
import time
from collections import deque


class FrameRateMeter:
    """Thread-safe rolling frame-rate meter.

    tick() once per produced frame, get_fps() from any thread. Returns 0.0
    until at least two samples are inside the window.
    """

    def __init__(self, window_s=2.0):
        self._window_s = max(0.5, float(window_s))
        self._stamps = deque()
        self._lock = threading.Lock()

    def _trim(self, now):
        cutoff = now - self._window_s
        while self._stamps and self._stamps[0] < cutoff:
            self._stamps.popleft()

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            self._stamps.append(now)
            self._trim(now)

    def get_fps(self, now=None):
        now = time.monotonic() if now is None else now
        with self._lock:
            self._trim(now)
            if len(self._stamps) < 2:
                return 0.0
            span = now - self._stamps[0]
            count = len(self._stamps) - 1
        return round(count / span, 1) if span > 0 else 0.0
