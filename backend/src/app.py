"""Flip-disc backend entry point.

    python src/app.py

Webcam -> RVM (or U2NET) segmentation -> people beyond MAX_PERSON_DISTANCE_M dropped ->
80x45 binary matrix, streamed to the React frontend over Socket.IO. With nobody
in range the idle clip loops instead.
"""

from threading import Thread

from pipeline import FlipdiscPipeline
from server.api import create_server, run


def main():
    app, socketio, publisher = create_server()
    pipeline = FlipdiscPipeline(publish=publisher)

    loop = Thread(target=pipeline.run, daemon=True)
    loop.start()
    try:
        run(app, socketio)
    except KeyboardInterrupt:
        pass
    finally:
        # Wait for the loop to release the camera. At 1280x720 (where the webcam
        # sends MJPEG) a process that exits with it still open hangs on exit.
        pipeline.stop()
        loop.join(timeout=3)


if __name__ == "__main__":
    main()
