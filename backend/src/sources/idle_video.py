"""Idle-loop playback.

Only the finished 80x45 masks are kept, not the decoded video. The clip is the
same every time round, so thresholding it once removes a colour convert + two
resizes + a threshold from every idle frame -- and 3.6 KB per frame instead of
230 KB means a long clip costs megabytes, not gigabytes.

Decoding the clip takes ~7 s, so the masks are also saved to CACHE_DIR and
reused until the video file changes.
"""

import cv2
import numpy as np

from settings import CACHE_DIR, FLIPDISC_RESOLUTION, IDLE_VIDEO_THRESHOLD


class IdleVideo:
    """Loops a clip that has been pre-reduced to flip-disc masks."""

    def __init__(self, video_path, resolution=FLIPDISC_RESOLUTION,
                 threshold=IDLE_VIDEO_THRESHOLD):
        self.index = 0
        self.masks = self._load(video_path, resolution, threshold)

        if len(self.masks) == 0:
            print(f"Warning: could not read idle video '{video_path}', using a blank frame")
            self.masks = np.zeros((1, resolution[1], resolution[0]), dtype=np.uint8)

        print(f"Idle video: {len(self.masks)} frames ({self.masks.nbytes / 1e6:.1f} MB)")

    @classmethod
    def _load(cls, video_path, resolution, threshold):
        if not video_path.exists():
            return np.empty(0)

        cache = CACHE_DIR / f"{video_path.name}.{resolution[0]}x{resolution[1]}.t{threshold}.npy"
        if cache.exists() and cache.stat().st_mtime >= video_path.stat().st_mtime:
            return np.load(cache)

        masks = cls._decode(video_path, resolution, threshold)
        if len(masks):
            CACHE_DIR.mkdir(exist_ok=True)
            np.save(cache, masks)
        return masks

    @classmethod
    def _decode(cls, video_path, resolution, threshold):
        masks = []
        capture = cv2.VideoCapture(str(video_path))
        while capture.isOpened():
            ok, frame = capture.read()
            if not ok:
                break
            masks.append(cls._to_mask(frame, resolution, threshold))
        capture.release()
        return np.array(masks, dtype=np.uint8)

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
