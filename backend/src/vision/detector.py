"""Person detection with YOLO, plus a rough distance estimate per box."""

from collections import namedtuple

import torch
from ultralytics import YOLO

from settings import (
    FOCAL_LENGTH_PIXELS,
    PERSON_CONFIDENCE_THRESHOLD,
    REAL_PERSON_HEIGHT_M,
    YOLO_WEIGHTS,
)

PERSON_CLASS_ID = 0

#: One detected person. `distance_m` is a monocular estimate from box height.
PersonBox = namedtuple("PersonBox", "x1 y1 x2 y2 confidence distance_m")


class PersonDetector:
    def __init__(self, device, weights=YOLO_WEIGHTS,
                 confidence_threshold=PERSON_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(str(weights))
        if device != "cpu":
            self.model.to(device)

    @torch.no_grad()
    def detect(self, frame):
        """Return the list of PersonBox found in `frame` (may be empty)."""
        if frame.mean() < 20:  # lens capped or lights off
            return []

        results = self.model(frame, classes=[PERSON_CLASS_ID],
                             conf=self.confidence_threshold, verbose=False)[0]
        if results.boxes is None:
            return []

        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            height = y2 - y1
            distance = ((REAL_PERSON_HEIGHT_M * FOCAL_LENGTH_PIXELS) / height
                        if height > 0 else float("inf"))
            boxes.append(PersonBox(x1, y1, x2, y2, float(box.conf.item()), distance))
        return boxes
