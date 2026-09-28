# Robust Video Matting, vendored from github.com/PeterL1n/RobustVideoMatting
# (model/ at master). GPL-3.0 -- see LICENSE in this directory. One change:
# mobilenetv3.py normalizes with buffers so the model can run as a CUDA graph.
from .model import MattingNetwork