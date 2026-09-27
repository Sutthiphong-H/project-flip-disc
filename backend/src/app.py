"""Flip-disc backend entry point.

    python src/app.py

Webcam -> YOLO person detection -> U2NET segmentation -> 80x45 binary matrix,
streamed to the React frontend over Socket.IO. With nobody in front of the
camera the idle clip loops instead.
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
