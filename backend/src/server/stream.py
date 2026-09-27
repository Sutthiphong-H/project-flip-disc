"""Publishes pipeline output to connected browsers over Socket.IO."""

import threading

EVENT = "flipdisc_update"


class SocketPublisher:
    """Callable the pipeline hands each payload to.

    Also keeps the last payload so a newly connected client and the /status
    endpoint can see the current state without waiting for the next frame.
    """

    def __init__(self, socketio):
        self.socketio = socketio
        self._lock = threading.Lock()
        self._latest = None

    def __call__(self, payload):
        with self._lock:
            self._latest = payload
        try:
            self.socketio.emit(EVENT, payload)
        except Exception as e:
            print(f"Error sending data to clients: {e}")

    @property
    def latest(self):
        with self._lock:
            return self._latest

    def send_snapshot(self):
        """Push the current state to the client that just connected."""
        payload = self.latest
        if payload is not None:
            self.socketio.emit(EVENT, payload)
