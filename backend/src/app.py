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

    Thread(target=pipeline.run, daemon=True).start()
    try:
        run(app, socketio)
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()


if __name__ == "__main__":
    main()
