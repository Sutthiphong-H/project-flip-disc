"""Flip-disc backend entry point.

    python src/app.py

Webcam -> RVM (or U2NET) segmentation -> people beyond MAX_PERSON_DISTANCE_M dropped ->
80x45 binary matrix, streamed to the React frontend over Socket.IO. With nobody
in range the idle clip loops instead.
"""

from threading import Thread

import settings
from pipeline import FlipdiscPipeline
from server.api import add_camera_routes, create_server, run


def main():
    app, socketio, publisher = create_server()
    if (settings.FRONTEND_DIST / "index.html").is_file():
        print(f"Serving the built frontend at http://localhost:{settings.PORT}/")
    else:
        print(f"No built frontend in {settings.FRONTEND_DIST} -- use `npm run dev`, "
              "or `npm run build` to have it served from here")
    pipeline = FlipdiscPipeline(publish=publisher)
    add_camera_routes(app, pipeline)

    loop = Thread(target=pipeline.run, daemon=True)
    loop.start()
    try:
        run(app, socketio)
    except KeyboardInterrupt:
        pass
    finally:
        # Wait for the loop to release the camera. At 1280x720 (where the webcam
        # sends MJPEG) a process that exits with it still open hangs on exit.
        print("Shutting down...")
        pipeline.stop()
        loop.join(timeout=3)
        if loop.is_alive():
            print("Warning: the pipeline didn't stop within 3 s; the camera may still be open")
        else:
            print("Backend stopped cleanly")


if __name__ == "__main__":
    main()
