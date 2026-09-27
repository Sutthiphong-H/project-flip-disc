"""The silhouette models behind one call: BGR frame -> person probability map.

    rvm    Robust Video Matting, MobileNetV3 (default). Recurrent -- it carries
           state from frame to frame. Keeps raised arms, hands and fingers that
           U2NET drops, including against backlight. GPL-3.0, see vision/rvm/.
    u2net  U2NET human segmentation. Stateless, Apache-2.0. Misses raised arms.

Both return a (H, W) tensor of 0..1 on the model's device at PROCESS_RESOLUTION,
and have a warm_up() that must run on the thread that will call them: cuDNN and
cuBLAS set up per thread, and RVM's first call on a fresh thread took 2.5 s.
"""

import cv2
import torch

from settings import (
    PROCESS_RESOLUTION,
    RVM_DOWNSAMPLE,
    RVM_WEIGHTS,
    SEGMENTATION_MODEL,
    U2NET_WEIGHTS,
)
from vision.rvm import MattingNetwork
from vision.u2net import U2NET

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_model(device, use_fp16, name=SEGMENTATION_MODEL):
    models = {"rvm": RobustVideoMatting, "u2net": U2Net}
    if name not in models:
        raise ValueError(f"FLIPDISC_MODEL must be one of {sorted(models)}, got {name!r}")
    return models[name](device, use_fp16)


def _to_tensor(frame, device, dtype):
    """BGR uint8 frame -> 1x3xHxW RGB tensor in 0..1 at PROCESS_RESOLUTION."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if (rgb.shape[1], rgb.shape[0]) != PROCESS_RESOLUTION:
        rgb = cv2.resize(rgb, PROCESS_RESOLUTION, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(rgb).to(device).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(dtype).div_(255.0)


class RobustVideoMatting:
    """~14 ms at 640x360 on the RTX 5060. Its backbone sees PROCESS_RESOLUTION
    scaled by RVM_DOWNSAMPLE (320x180 by default, where it does best); a guided
    filter then refines the edges back up to full size."""

    def __init__(self, device, use_fp16):
        self.device = device
        self.dtype = torch.float16 if use_fp16 else torch.float32
        model = MattingNetwork("mobilenetv3")
        model.load_state_dict(torch.load(str(RVM_WEIGHTS), map_location="cpu"))
        self.model = model.to(device, self.dtype).eval()
        self._state = [None] * 4

    def warm_up(self):
        blank = torch.zeros(1, 3, PROCESS_RESOLUTION[1], PROCESS_RESOLUTION[0],
                            device=self.device, dtype=self.dtype)
        with torch.no_grad():
            state = self.model(blank, downsample_ratio=RVM_DOWNSAMPLE)[2:]
            self.model(blank, *state, downsample_ratio=RVM_DOWNSAMPLE)  # the with-state path too

    def __call__(self, frame):
        src = _to_tensor(frame, self.device, self.dtype)
        _, alpha, *self._state = self.model(src, *self._state, downsample_ratio=RVM_DOWNSAMPLE)
        return alpha[0, 0]


class U2Net:
    """Recorded as one CUDA graph on GPU. Eager U2NET is launch-bound -- hundreds
    of small kernels, ~27 ms at any input size on the RTX 5060. Replayed as a
    graph it is ~5 ms at 320x180, ~11 ms at 512x288, ~16 ms at 640x360."""

    def __init__(self, device, use_fp16):
        self.device = device
        self.dtype = torch.float16 if use_fp16 else torch.float32
        model = U2NET(in_ch=3, out_ch=1)
        model.load_state_dict(torch.load(str(U2NET_WEIGHTS), map_location="cpu"))
        # channels_last is the layout fp16 tensor-core convolutions want: ~20% faster.
        self.model = model.to(device, self.dtype, memory_format=torch.channels_last).eval()
        self._mean = MEAN.to(device, self.dtype)
        self._std = STD.to(device, self.dtype)
        self._graph = None

    def warm_up(self):
        """On GPU, record the forward pass as a CUDA graph. Input size is fixed after this."""
        if not self.device.startswith("cuda"):
            return
        width, height = PROCESS_RESOLUTION
        self._static_in = torch.zeros(1, 3, height, width, device=self.device, dtype=self.dtype)
        self._static_in = self._static_in.contiguous(memory_format=torch.channels_last)

        with torch.no_grad():
            # Capture needs the allocator and cuDNN already warmed up, on a side stream.
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):
                    self.model(self._static_in)
            torch.cuda.current_stream().wait_stream(side)

            self._graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph):
                self._static_out = self.model(self._static_in)[0]

    def __call__(self, frame):
        """With a graph, the result is only valid until the next call."""
        src = _to_tensor(frame, self.device, self.dtype).sub_(self._mean).div_(self._std)
        if self._graph is None:
            return self.model(src)[0][0, 0]
        self._static_in.copy_(src)
        self._graph.replay()
        return self._static_out[0, 0]
