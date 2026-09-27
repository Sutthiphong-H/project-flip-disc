"""U2NET silhouette segmentation on a worker thread."""

from collections import namedtuple
from queue import Empty, Queue
from threading import Thread

import cv2
import torch

from settings import FLIPDISC_RESOLUTION, PROCESS_RESOLUTION, U2NET_WEIGHTS
from vision.u2net import U2NET

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

#: `preview` is the masked BGR frame for the debug window, or None.
SegmentResult = namedtuple("SegmentResult", "flipdisc_mask preview")


def load_u2net(device, use_fp16, weights=U2NET_WEIGHTS):
    model = U2NET(in_ch=3, out_ch=1)
    model.load_state_dict(torch.load(str(weights), map_location=device))
    model.to(device).eval()
    if use_fp16:
        model = model.half()
    return model


class Segmenter:
    """Latest-frame-wins segmentation worker.

    Input frames may be dropped freely -- a frame the pipeline could not keep up
    with is stale by definition, and holding it back only adds latency.
    """

    def __init__(self, model, device, use_fp16, with_preview=False):
        self.model = model
        self.device = device
        self.use_fp16 = use_fp16
        self.with_preview = with_preview
        self._mean = MEAN.to(device)
        self._std = STD.to(device)
        self._input = Queue(maxsize=1)
        self._output = Queue(maxsize=1)
        self._stopped = False

    def start(self):
        Thread(target=self._run, daemon=True).start()
        return self

    def submit(self, frame):
        """Hand a frame to the worker, replacing any frame still waiting."""
        if self._input.full():
            try:
                self._input.get_nowait()
            except Empty:
                pass
        self._input.put(frame)

    def poll(self):
        """Return the newest SegmentResult, or None if nothing is ready."""
        try:
            return self._output.get_nowait()
        except Empty:
            return None

    def stop(self):
        self._stopped = True

    def _run(self):
        with torch.no_grad():
            while not self._stopped:
                try:
                    frame = self._input.get(timeout=0.1)
                except Empty:
                    continue

                try:
                    result = self._segment(frame)
                except Exception as e:
                    print(f"Segmentation error: {e}")
                    continue

                if self._output.full():
                    try:
                        self._output.get_nowait()
                    except Empty:
                        pass
                self._output.put(result)

    def _segment(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, PROCESS_RESOLUTION, interpolation=cv2.INTER_AREA)

        tensor = torch.from_numpy(resized).to(self.device)
        tensor = tensor.permute(2, 0, 1).float().div_(255.0).sub_(self._mean).div_(self._std)
        tensor = tensor.unsqueeze(0)
        if self.use_fp16:
            tensor = tensor.half()

        pred = self.model(tensor)[0].squeeze().float()
        # Normalise on the device, before the one transfer back to the host.
        lo, hi = pred.min(), pred.max()
        if hi > lo:
            pred = (pred - lo) / (hi - lo)
        mask = (pred * 255).to(torch.uint8).cpu().numpy()

        small = cv2.resize(mask, FLIPDISC_RESOLUTION, interpolation=cv2.INTER_AREA)
        _, flipdisc_mask = cv2.threshold(small, 127, 255, cv2.THRESH_BINARY)

        preview = None
        if self.with_preview:
            full = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
            preview = cv2.bitwise_and(frame, cv2.cvtColor(full, cv2.COLOR_GRAY2BGR))

        return SegmentResult(flipdisc_mask, preview)
