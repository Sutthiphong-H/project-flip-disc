"""Idle-loop playback.

Only the finished 80x45 masks are cached, not the decoded video. The clip is
the same every time round, so thresholding it once at startup removes a colour
convert + two resizes + a threshold from every idle frame -- and 3.6 KB per
cached frame instead of 230 KB means a long clip costs megabytes, not gigabytes.
"""

import cv2
import numpy as np

from settings import FLIPDISC_RESOLUTION, IDLE_VIDEO_THRESHOLD


class IdleVideo:
    """Loops a clip that has been pre-reduced to flip-disc masks."""

    def __init__(self, video_path, resolution=FLIPDISC_RESOLUTION,
                 threshold=IDLE_VIDEO_THRESHOLD):
        self.resolution = resolution
        self.masks = []
        self.index = 0

        capture = cv2.VideoCapture(str(video_path))
        if capture.isOpened():
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                self.masks.append(self._to_mask(frame, resolution, threshold))
            capture.release()

        if not self.masks:
            print(f"Warning: could not read idle video '{video_path}', using a blank frame")
            self.masks = [np.zeros(resolution[::-1], dtype=np.uint8)]

        print(f"Idle video: {len(self.masks)} frames cached "
              f"({len(self.masks) * self.masks[0].nbytes / 1e6:.1f} MB)")

    @staticmethod
    def _to_mask(frame, resolution, threshold):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, resolution, interpolation=cv2.INTER_AREA)
        _, mask = cv2.threshold(small, threshold, 255, cv2.THRESH_BINARY)
        return mask

    def next_mask(self):
        mask = self.masks[self.index]
        self.index = (self.index + 1) % len(self.masks)
        return mask

    @property
    def frame_count(self):
        return len(self.masks)
