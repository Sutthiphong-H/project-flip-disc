"""Silhouette segmentation on a worker thread, split into people by distance."""

import traceback
from collections import namedtuple
from functools import partial
from queue import Empty, Queue
from threading import Event, Thread

import cv2
import numpy as np
import torch

from settings import (
    DISC_OFF,
    DISC_ON,
    DISC_SMOOTHING,
    FLIPDISC_RESOLUTION,
    FOCAL_LENGTH_PIXELS,
    FOCAL_REFERENCE_HEIGHT,
    MASK_THRESHOLD,
    MAX_PERSON_DISTANCE_M,
    MIN_PERSON_AREA,
    PROCESS_RESOLUTION,
    REAL_PERSON_HEIGHT_M,
)

# A one-off bad frame is skipped. This many in a row is an error that sticks --
# after a GPU driver reset the CUDA context stays broken until the process
# restarts -- so the worker gives up and the pipeline fails (~1.5 s of frames).
MAX_CONSECUTIVE_ERRORS = 30

#: One silhouette, in PROCESS_RESOLUTION pixels.
Person = namedtuple("Person", "x y w h distance_m")

#: `people` is everyone found, in range or not. `preview` is a zero-argument
#: callable that renders the camera frame with everything outside the kept mask
#: dimmed and a box per person -- only called while someone watches the web
#: debug view.
SegmentResult = namedtuple("SegmentResult", "flipdisc_mask people preview")


def split_people(mask):
    """Return (mask keeping only people within MAX_PERSON_DISTANCE_M, all people)."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = MIN_PERSON_AREA * mask.size
    to_focal_px = FOCAL_REFERENCE_HEIGHT / mask.shape[0]
    kept = np.zeros_like(mask)
    people = []
    for i in range(1, count):  # 0 is the background
        x, y, w, h, area = stats[i]
        if area < min_area:
            continue
        distance = REAL_PERSON_HEIGHT_M * FOCAL_LENGTH_PIXELS / (h * to_focal_px)
        person = Person(x, y, w, h, distance)
        people.append(person)
        if person.distance_m <= MAX_PERSON_DISTANCE_M:
            kept[labels == i] = 255
    return kept, people


class Segmenter:
    """Latest-frame-wins segmentation worker.

    Input frames may be dropped freely -- a frame the pipeline could not keep up
    with is stale by definition, and holding it back only adds latency.
    """

    def __init__(self, model):
        """`model` is one of vision.models: BGR frame -> 0..1 person probability map."""
        self.model = model
        self._input = Queue(maxsize=1)
        self._output = Queue(maxsize=1)
        self._stopped = False
        self._reset = False
        self._ready = Event()
        self.error = None  # why the worker stopped, if it failed
        # Anti-flicker state, one entry per disc. Only the worker thread touches it.
        self._coverage = np.zeros(FLIPDISC_RESOLUTION[::-1], np.float32)
        self._discs = np.zeros(FLIPDISC_RESOLUTION[::-1], bool)

    def start(self):
        """Start the worker, which warms the model up first. See wait_ready()."""
        Thread(target=self._run, daemon=True).start()
        return self

    def wait_ready(self):
        """Block until the warm-up is done (RVM: ~2.5 s, overlapped with camera open).
        Raises if the warm-up failed."""
        self._ready.wait()
        if self.error is not None:
            raise RuntimeError(f"segmentation model failed to start: {self.error}")

    def submit(self, frame):
        """Hand a frame to the worker, replacing any frame still waiting."""
        if self._input.full():
            try:
                self._input.get_nowait()
            except Empty:
                pass
        self._input.put(frame)

    def poll(self):
        """Return the newest SegmentResult, or None if nothing is ready. Raises
        once the worker has given up."""
        if self.error is not None:
            raise RuntimeError(f"segmentation stopped: {self.error}")
        try:
            return self._output.get_nowait()
        except Empty:
            return None

    def reset(self):
        """Drop what the model carried over from earlier frames (RVM's recurrent
        state). After a camera switch or a gap it describes another scene."""
        self._reset = True

    def stop(self):
        self._stopped = True

    def _run(self):
        try:
            with torch.no_grad():
                self.model.warm_up()  # on this thread -- see vision/models.py
                self._ready.set()
                self._work()
        except Exception as e:
            traceback.print_exc()
            self.error = e
        finally:
            self._ready.set()  # a failed warm-up must not leave wait_ready() hanging

    def _work(self):
        errors = 0
        while not self._stopped:
            try:
                frame = self._input.get(timeout=0.1)
            except Empty:
                continue

            if self._reset:
                self._reset = False
                self.model.reset()

            try:
                result = self._segment(frame)
                errors = 0
            except Exception as e:
                errors += 1
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    raise
                print(f"Segmentation error: {e}")
                continue

            if self._output.full():
                try:
                    self._output.get_nowait()
                except Empty:
                    pass
            self._output.put(result)

    def _segment(self, frame):
        # Use the probability as-is. Don't min-max stretch it: on an empty
        # scene that blows background noise up into a full-frame silhouette.
        prob = self.model(frame)
        mask = (prob > MASK_THRESHOLD).to(torch.uint8).mul_(255).cpu().numpy()
        mask, people = split_people(mask)

        return SegmentResult(self._to_discs(mask), people, partial(self._preview, frame, mask, people))

    def _to_discs(self, mask):
        """Downscale to the display without discs on the silhouette edge flickering.

        An edge disc is about half covered, so a plain 50% threshold flips it on
        and off with every frame's noise. Smoothing the coverage over time and
        using separate on/off thresholds holds it steady.
        """
        coverage = cv2.resize(mask, FLIPDISC_RESOLUTION, interpolation=cv2.INTER_AREA)
        self._coverage += DISC_SMOOTHING * (coverage / np.float32(255) - self._coverage)
        self._discs = (self._coverage > DISC_ON) | (self._discs & (self._coverage >= DISC_OFF))
        return self._discs.astype(np.uint8) * 255

    @staticmethod
    def _preview(frame, mask, people):
        full = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        # The scene dimmed, the silhouette at full brightness. (cv2 rather than
        # numpy boolean indexing: 1 ms instead of 23 ms at 1280x720.)
        preview = cv2.convertScaleAbs(frame, alpha=1 / 3)
        cv2.copyTo(frame, full, preview)
        scale = frame.shape[1] / PROCESS_RESOLUTION[0]
        to_focal_px = FOCAL_REFERENCE_HEIGHT / mask.shape[0]
        for p in people:
            colour = (0, 255, 0) if p.distance_m <= MAX_PERSON_DISTANCE_M else (0, 0, 255)
            x, y, w, h = (int(v * scale) for v in (p.x, p.y, p.w, p.h))
            cv2.rectangle(preview, (x, y), (x + w, y + h), colour, 1)
            # h in the same pixels FOCAL_LENGTH_PIXELS uses, for calibrating it
            cv2.putText(preview, f"{p.distance_m:.1f}m h={int(p.h * to_focal_px)}px", (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        return preview
