"""Flask + Socket.IO server the React frontend talks to."""

from flask import Flask, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO

import settings
from server.stream import SocketPublisher


def create_server():
    """Return (app, socketio, publisher) with routes and handlers registered."""
    app = Flask(__name__)
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    publisher = SocketPublisher(socketio)

    @app.route("/status")
    def status():
        latest = publisher.latest or {}
        return jsonify({
            "status": "running",
            "camera": settings.CAMERA_INDEX,
            "resolution": list(settings.FLIPDISC_RESOLUTION),
            "mode": latest.get("mode"),
            "fps": latest.get("fps", 0.0),
            "blackout": latest.get("blackout"),
        })

    @socketio.on("connect")
    def on_connect():
        # Don't leave a fresh tab on random noise until the next frame lands.
        publisher.send_snapshot()

    return app, socketio, publisher


def run(app, socketio):
    socketio.run(app, host=settings.HOST, port=settings.PORT,
                 debug=False, use_reloader=False)
