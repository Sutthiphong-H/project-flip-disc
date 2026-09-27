"""Publishes pipeline output to connected browsers.

The flip-disc matrix goes out over Socket.IO. The camera preview for the debug
view goes out as MJPEG on its own HTTP route, and is only JPEG-encoded while a
browser is actually watching it.
"""

import threading

import cv2

EVENT = "flipdisc_update"
JPEG_QUALITY = 70


class SocketPublisher:
    """Callable the pipeline hands each payload (and camera preview) to.

    Also keeps the last payload so a newly connected client and the /status
    endpoint can see the current state without waiting for the next frame.
    """

    def __init__(self, socketio):
        self.socketio = socketio
        self._cond = threading.Condition()
        self._latest = None
        self._preview = None
        self._preview_seq = 0

    def __call__(self, payload, preview=None):
        with self._cond:
            self._latest = payload
            if preview is not None:
                self._preview = preview
                self._preview_seq += 1
                self._cond.notify_all()
        try:
            self.socketio.emit(EVENT, payload)
        except Exception as e:
            print(f"Error sending data to clients: {e}")

    @property
    def latest(self):
        with self._cond:
            return self._latest

    def send_snapshot(self):
        """Push the current state to the client that just connected."""
        payload = self.latest
        if payload is not None:
            self.socketio.emit(EVENT, payload)

    def mjpeg(self):
        """Yield multipart JPEG parts, one per new preview, until the client leaves."""
        seen = 0
        while True:
            with self._cond:
                if not self._cond.wait_for(lambda: self._preview_seq != seen, timeout=1.0):
                    continue
                seen, frame = self._preview_seq, self._preview

            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpeg.tobytes() + b"\r\n")
