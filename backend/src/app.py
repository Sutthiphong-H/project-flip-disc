"""Flip-disc backend entry point.

    python src/app.py        (or run-backend.bat, which restarts it after a failure)

Webcam -> RVM (or U2NET) segmentation -> people beyond MAX_PERSON_DISTANCE_M dropped ->
80x45 binary matrix, streamed to the React frontend over Socket.IO. With nobody
in range the idle clip loops instead.
"""

import ctypes
import os
import sys
import time
from threading import Event, Thread

import settings
from pipeline import FlipdiscPipeline
from server.api import add_camera_routes, create_server, run


def exit_now(pipeline):
    """End the process with code 1, even with the pipeline thread hung."""
    # Release the camera if its own thread still responds -- the pipeline may
    # be stuck and never get to it.
    camera = pipeline.camera
    if camera is not None:
        release = Thread(target=camera.stop, daemon=True)
        release.start()
        release.join(timeout=3)
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":
        # os._exit() hung here (killed after 90 s) whenever the pipeline was
        # stuck; TerminateProcess skips the DLL teardown that waits.
        kernel32 = ctypes.windll.kernel32
        kernel32.TerminateProcess(kernel32.GetCurrentProcess(), 1)
    os._exit(1)


def watchdog(loop, publisher, pipeline, stopping):
    """Exit with code 1 once the pipeline has died or stopped sending frames.

    None of the causes recover in place -- a model that won't load, a CUDA
    context broken by a GPU driver reset, a hung thread -- but a fresh process
    usually does, and run-backend.bat starts one. A clean stop exits with 0.
    """
    started = time.monotonic()
    while not stopping.wait(1.0):
        age = publisher.frame_age
        if not loop.is_alive():
            problem = "the pipeline stopped (error above)"
        elif age is None and time.monotonic() - started > settings.STARTUP_SECONDS:
            problem = f"no frame within {settings.STARTUP_SECONDS:.0f} s of starting"
        elif age is not None and age > settings.STALL_SECONDS:
            problem = f"no frame for {age:.0f} s"
        else:
            continue
        print(f"Fatal: {problem} -- exiting so the backend can be restarted", flush=True)
        exit_now(pipeline)


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
    stopping = Event()
    Thread(target=watchdog, args=(loop, publisher, pipeline, stopping), daemon=True).start()
    try:
        run(app, socketio)
    except KeyboardInterrupt:
        pass
    finally:
        stopping.set()
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
