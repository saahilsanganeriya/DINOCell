"""
DINOCell: Cell Segmentation with DINOv3
========================================

A powerful cell segmentation framework using DINOv3 Vision Transformer.
"""

from .model import DINOCell, create_dinocell_model
from .pipeline import DINOCellPipeline
from .preprocessing import apply_clahe, preprocess_for_dinov3

__version__ = "0.1.0"

__all__ = [
    'DINOCell',
    'create_dinocell_model',
    'DINOCellPipeline',
    'apply_clahe',
    'preprocess_for_dinov3',
]
