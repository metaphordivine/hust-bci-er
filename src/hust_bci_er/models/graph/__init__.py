"""Graph model implementations."""

from .dgcnn import DGCNN, build_dgcnn
from .lggnet import LGGNet, build_lggnet

__all__ = ["DGCNN", "LGGNet", "build_dgcnn", "build_lggnet"]

