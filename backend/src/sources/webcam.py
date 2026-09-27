"""USB webcam capture."""

import os

import cv2

from settings import CAMERA_FPS, CAMERA_INDEX, INPUT_RESOLUTION

# DirectShow, not Media Foundation: MSMF reports 30 fps on this webcam but pads
# it with duplicate frames -- DirectShow delivered more unique frames per second.
_BACKEND = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY


class Webcam:
    """Opens one camera index and hands out frames at INPUT_RESOLUTION."""

    def __init__(self, index=CAMERA_INDEX, resolution=INPUT_RESOLUTION):
        self.index = index
        self.resolution = resolution
        self.capture = None

    def open(self):
        """Return True once the camera is delivering frames."""
        capture = cv2.VideoCapture(self.index, _BACKEND)
        if not capture.isOpened():
            print(f"Error: could not open camera {self.index}")
            return False

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        capture.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
        # A deep driver buffer means every read() returns a stale frame; 1 keeps
        # the loop on live pixels. Not every backend honours it.
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.capture = capture
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera {self.index} open at {width}x{height}")
        return True

    def read(self):
        """Return a mirrored frame at INPUT_RESOLUTION, or None."""
        if self.capture is None:
            return None

        ok, frame = self.capture.read()
        if not ok or frame is None:
            return None

        if (frame.shape[1], frame.shape[0]) != self.resolution:
            frame = cv2.resize(frame, self.resolution)
        return cv2.flip(frame, 1)

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None
