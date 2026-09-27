"""Flask + Socket.IO server the React frontend talks to -- and, once built with
`npm run build`, the server that hands out the frontend itself."""

import mimetypes

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

import settings
from server.stream import SocketPublisher
from sources.camera import probe, scan_webcams, thumbnail
from sources.camera_store import hard_signature, validate


def create_server():
    """Return (app, socketio, publisher) with routes and handlers registered."""
    app = Flask(__name__, static_folder=None)
    CORS(app)
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
    publisher = SocketPublisher(socketio)

    @app.route("/status")
    def status():
        latest = publisher.status
        return jsonify({
            "status": "running",
            "camera_ok": latest.get("camera_ok"),
            "resolution": list(settings.FLIPDISC_RESOLUTION),
            "mode": latest.get("mode"),
            "fps": latest.get("fps", 0.0),
            "people": latest.get("people"),
            "nearest_m": latest.get("nearest_m"),
        })

    @app.route("/debug.mjpg")
    def debug_stream():
        # Camera frame, non-silhouette dimmed, a distance box per person.
        return Response(publisher.mjpeg(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")

    # Windows takes these from the registry, where .js can be text/plain, and
    # browsers refuse to run a module script served as that.
    mimetypes.add_type("text/javascript", ".js")
    mimetypes.add_type("text/css", ".css")

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def frontend(path):
        # /status, /debug.mjpg and /socket.io are matched before this.
        dist = settings.FRONTEND_DIST
        if not (dist / "index.html").is_file():
            return ("Frontend not built: run `npm run build` in flip-disc_frontend "
                    f"(looked in {dist}).", 404)
        if path and (dist / path).is_file():
            return send_from_directory(dist, path)
        return send_from_directory(dist, "index.html")  # client-side routes

    @socketio.on("connect")
    def on_connect():
        # Updates are deltas, so a fresh tab needs the whole matrix first.
        publisher.send_snapshot(request.sid)

    return app, socketio, publisher


def add_camera_routes(app, pipeline):
    """The web UI's camera manager: list, add, edit, delete, test and pick the
    active camera. Needs the running pipeline, so it is added after
    create_server(). Passwords never leave the backend."""
    store = pipeline.cameras

    def fail(message, status=400):
        return jsonify(ok=False, error=message), status

    def listing():
        camera = pipeline.camera
        return dict(store.listing(), signal=bool(camera and camera.has_signal))

    def starting():
        return fail("the backend is still starting, try again in a moment", 503)

    @app.get("/cameras")
    def cameras_list():
        return jsonify(listing())

    @app.get("/cameras/scan")
    def cameras_scan():
        return jsonify(webcams=scan_webcams(in_use=pipeline.camera))  # None while starting: fine

    @app.post("/cameras/test")
    def cameras_test():
        data = request.get_json(silent=True) or {}
        try:
            cam = validate(data, existing=store.get(data.get("id")))
        except ValueError as e:
            return fail(str(e))
        running = pipeline.camera
        if running is None:
            return starting()
        if hard_signature(cam) == hard_signature(running.cam):
            frame = running.snapshot()  # the device is busy: it's the running camera
            return jsonify(ok=True, in_use=True, thumbnail=thumbnail(frame) if frame is not None else None)
        return jsonify(probe(cam))

    @app.post("/cameras")
    def cameras_add():
        try:
            cam = validate(request.get_json(silent=True))
        except ValueError as e:
            return fail(str(e))
        store.add(cam)
        return jsonify(ok=True, **listing())

    @app.put("/cameras/<int:cam_id>")
    def cameras_edit(cam_id):
        existing = store.get(cam_id)
        if existing is None:
            return fail("no such camera", 404)
        try:
            cam = validate(request.get_json(silent=True), existing=existing)
        except ValueError as e:
            return fail(str(e))
        if cam_id == store.active_id:
            # Editing the running camera: only keep settings that give a picture.
            if pipeline.camera is None:
                return starting()
            ok, error = pipeline.switch_camera(cam)
            if not ok:
                return fail(error, 422)
        store.update(cam)
        return jsonify(ok=True, **listing())

    @app.delete("/cameras/<int:cam_id>")
    def cameras_delete(cam_id):
        try:
            store.remove(cam_id)
        except ValueError as e:
            return fail(str(e), 409)
        return jsonify(ok=True, **listing())

    @app.post("/cameras/active")
    def cameras_activate():
        cam = store.get((request.get_json(silent=True) or {}).get("id"))
        if cam is None:
            return fail("no such camera", 404)
        if pipeline.camera is None:
            return starting()
        ok, error = pipeline.switch_camera(cam)
        if not ok:
            return fail(error, 422)
        store.set_active(cam["id"])
        return jsonify(ok=True, **listing())


def run(app, socketio):
    socketio.run(app, host=settings.HOST, port=settings.PORT,
                 debug=False, use_reloader=False)
