# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from .ade20k import ADE20K
from .coco_captions import CocoCaptions
from .image_net import ImageNet
from .image_net_22k import ImageNet22k
from .nyu import NYU
from .jump_cellpainting import JUMPCellPainting
from .jump_cellpainting_multiview import JUMPCellPaintingMultiView

# S3 streaming version (optional, requires boto3)
try:
    from .jump_cellpainting_s3 import JUMPS3Dataset
except ImportError:
    JUMPS3Dataset = None
