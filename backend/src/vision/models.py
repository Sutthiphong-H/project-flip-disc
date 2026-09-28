"""The silhouette models behind one call: BGR frame -> person probability map.

    rvm    Robust Video Matting, MobileNetV3 (default). Recurrent -- it carries
           state from frame to frame. Keeps raised arms, hands and fingers that
           U2NET drops, including against backlight. GPL-3.0, see vision/rvm/.
    u2net  U2NET human segmentation. Stateless, Apache-2.0. Misses raised arms.

Both return a (H, W) tensor of 0..1 on the model's device at their `size`,
and have a warm_up() that must run on the thread that will call them: cuDNN and
cuBLAS set up per thread, and RVM's first call on a fresh thread took 2.5 s.
reset() forgets anything carried over from earlier frames.

On a GPU both run as a recorded CUDA graph: they are launch-bound (hundreds of
small kernels), and replaying the graph skips the per-kernel launch cost. If
recording fails they run as plain PyTorch instead. The CPU runs them as they
are: RVM at ~42 ms is still close to the webcam's 20 fps, U2NET at ~0.9 s isn't.
"""

import cv2
import torch

from settings import (
    RVM_DOWNSAMPLE,
    RVM_WEIGHTS,
    SEGMENTATION_MODEL,
    U2NET_WEIGHTS,
    process_resolution,
)
from vision.rvm import MattingNetwork
from vision.u2net import U2NET

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_model(device, use_fp16, name=SEGMENTATION_MODEL):
    models = {"rvm": RobustVideoMatting, "u2net": U2Net}
    if name not in models:
        raise ValueError(f"FLIPDISC_MODEL must be one of {sorted(models)}, got {name!r}")
    if name == "u2net" and device == "cpu":
        print("Warning: U2NET on the CPU takes ~0.9 s per frame; FLIPDISC_MODEL=rvm runs at ~42 ms")
    return models[name](device, use_fp16, process_resolution(name, device != "cpu"))


def _record(step, static_in, runs):
    """Record `step(static_in)` as a CUDA graph; returns (graph, output).
    `runs` warms the allocator and cuDNN up first, on a side stream, as capture needs."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(runs):
            step(static_in)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = step(static_in)
    return graph, out


def _to_tensor(frame, size, device, dtype):
    """BGR uint8 frame -> 1x3xHxW RGB tensor in 0..1 at `size` (width, height)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if (rgb.shape[1], rgb.shape[0]) != size:
        rgb = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(rgb).to(device).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(dtype).div_(255.0)


class RobustVideoMatting:
    """On the RTX 5060 as a CUDA graph: 4 ms at 1280x720 (the GPU default); at
    640x360 2.3 ms, against 15.5 ms without a graph, with the same output.
    ~42 ms on the CPU at 640x360. Its backbone sees the input scaled by
    RVM_DOWNSAMPLE; a guided filter then refines the edges back up to full size.

    The graph carries the recurrent state too: each replay reads the four
    state buffers and writes the new state back into them."""

    def __init__(self, device, use_fp16, size):
        self.device = device
        self.size = size  # input (width, height)
        self.dtype = torch.float16 if use_fp16 else torch.float32
        model = MattingNetwork("mobilenetv3")
        model.load_state_dict(torch.load(str(RVM_WEIGHTS), map_location="cpu"))
        self.model = model.to(device, self.dtype).eval()
        self._state = [None] * 4  # used without a graph
        self._graph = None

    def warm_up(self):
        blank = torch.zeros(1, 3, self.size[1], self.size[0], device=self.device, dtype=self.dtype)
        with torch.no_grad():
            state = self.model(blank, downsample_ratio=RVM_DOWNSAMPLE)[2:]
            if self.device.startswith("cuda"):
                try:
                    self._record(blank, state)
                    return
                except Exception as e:
                    self._graph = None
                    print(f"Couldn't record RVM as a CUDA graph ({e}); running it without one")
            self.model(blank, *state, downsample_ratio=RVM_DOWNSAMPLE)  # the with-state path too

    def _record(self, blank, state):
        self._src = blank.clone()
        self._rec = [s.clone() for s in state]

        def step(src):
            _, alpha, *new = self.model(src, *self._rec, downsample_ratio=RVM_DOWNSAMPLE)
            for buffer, value in zip(self._rec, new):
                buffer.copy_(value)
            return alpha

        self._graph, self._alpha = _record(step, self._src, runs=3)
        self.reset()  # the warm-up runs left state behind

    def reset(self):
        if self._graph is None:
            self._state = [None] * 4
        else:
            for buffer in self._rec:
                buffer.zero_()  # what RVM starts from when it is given no state

    def __call__(self, frame):
        """With a graph, the result is only valid until the next call."""
        src = _to_tensor(frame, self.size, self.device, self.dtype)
        if self._graph is None:
            _, alpha, *self._state = self.model(src, *self._state, downsample_ratio=RVM_DOWNSAMPLE)
        else:
            self._src.copy_(src)
            self._graph.replay()
            alpha = self._alpha
        alpha = alpha[0, 0]
        # The state is fed back every frame in fp16: one NaN or overflow in it
        # would turn every later frame into NaN -- an empty mask for good.
        if not torch.isfinite(alpha).all():
            print("RVM output was not finite; resetting its recurrent state")
            self.reset()
            return torch.zeros_like(alpha)
        return alpha


class U2Net:
    """Without a graph ~27 ms at any input size on the RTX 5060; as a graph
    ~5 ms at 320x180, ~11 ms at 512x288, ~16 ms at 640x360."""

    def __init__(self, device, use_fp16, size):
        self.device = device
        self.size = size  # input (width, height)
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
        width, height = self.size
        blank = torch.zeros(1, 3, height, width, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            if self.device.startswith("cuda"):
                self._static_in = blank.contiguous(memory_format=torch.channels_last)
                try:
                    self._graph, self._static_out = _record(lambda x: self.model(x)[0],
                                                            self._static_in, runs=3)
                    return
                except Exception as e:
                    self._graph = None
                    print(f"Couldn't record U2NET as a CUDA graph ({e}); running it without one")
            self.model(blank)  # sets cuDNN / oneDNN up on this thread

    def reset(self):
        """Stateless: nothing to forget."""

    def __call__(self, frame):
        """With a graph, the result is only valid until the next call."""
        src = _to_tensor(frame, self.size, self.device, self.dtype).sub_(self._mean).div_(self._std)
        if self._graph is None:
            return self.model(src)[0][0, 0]
        self._static_in.copy_(src)
        self._graph.replay()
        return self._static_out[0, 0]
