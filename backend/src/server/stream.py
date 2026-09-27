"""Publishes pipeline output to connected browsers.

The disc matrix goes out over Socket.IO as deltas: only the discs that changed
since the previous frame, as flat indices (row * cols + col) in "on" and "off".
On the webcam a moving person changes ~50 of the 3600 discs per frame (median
~20), so a frame is ~0.4 KB instead of ~11 KB for the whole matrix. A browser
that connects gets the whole matrix once, as "discs" (a string of 0s and 1s),
and applies deltas from there. Driving a physical display works the same way:
pulsing only the discs that change is what saves time and power.

The camera preview for the debug view goes out as MJPEG on its own HTTP
route, and is only rendered and JPEG-encoded while a browser is watching it.
"""

import threading

import cv2
import numpy as np

EVENT = "flipdisc_update"
JPEG_QUALITY = 70


class SocketPublisher:
    """Callable the pipeline hands each frame's discs, status and preview to.

    Also keeps the current discs and status so a newly connected client and
    the /status endpoint don't have to wait for the next frame.
    """

    def __init__(self, socketio):
        self.socketio = socketio
        # Held while updating the discs AND emitting, so a snapshot can't slip
        # between a delta and the state it was computed from.
        self._lock = threading.Lock()
        self._discs = None   # flat bool array, what every client has been sent
        self._status = {}
        self._cond = threading.Condition()
        self._preview = None
        self._preview_seq = 0

    def __call__(self, discs, status, preview=None):
        """`discs` is the rows x cols bool matrix, `status` a dict of plain
        values, `preview` a zero-argument callable returning the BGR debug frame."""
        flat = discs.ravel()
        with self._lock:
            first = self._discs is None
            if not first:
                changed = np.flatnonzero(flat != self._discs)
            self._discs = flat.copy()
            self._status = status
            if first:
                # Clients that connected before there was anything to snapshot
                # are still waiting for a whole matrix.
                payload = self._full_payload()
            else:
                payload = dict(status,
                               on=changed[flat[changed]].tolist(),
                               off=changed[~flat[changed]].tolist())
            try:
                self.socketio.emit(EVENT, payload)
            except Exception as e:
                print(f"Error sending data to clients: {e}")

        if preview is not None:
            with self._cond:
                self._preview = preview
                self._preview_seq += 1
                self._cond.notify_all()

    @property
    def status(self):
        with self._lock:
            return dict(self._status)

    def send_snapshot(self, sid):
        """Send the whole matrix to the client that just connected."""
        with self._lock:
            if self._discs is not None:  # otherwise the first frame will be a full one
                self.socketio.emit(EVENT, self._full_payload(), to=sid)

    def _full_payload(self):
        """The whole matrix as a string of 0s and 1s. Call with the lock held."""
        discs = (self._discs.astype(np.uint8) + ord("0")).tobytes().decode()
        return dict(self._status, discs=discs)

    def mjpeg(self):
        """Yield multipart JPEG parts, one per new preview, until the client leaves."""
        seen = 0
        while True:
            with self._cond:
                if not self._cond.wait_for(lambda: self._preview_seq != seen, timeout=1.0):
                    continue
                seen, render = self._preview_seq, self._preview

            frame = render()
            ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                       + jpeg.tobytes() + b"\r\n")
