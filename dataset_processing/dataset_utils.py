"""
Dataset Utilities for DINOCell
===============================

Shared utilities for dataset processing.
"""

import numpy as np
import cv2

try:
    from scipy.ndimage import distance_transform_edt
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using OpenCV for distance transform")


def create_distance_map(mask):
    """
    Create normalized distance map from segmentation mask.
    
    Parameters
    ----------
    mask : numpy.ndarray
        Segmentation mask where each cell has a unique label (H, W)
    
    Returns
    -------
    numpy.ndarray
        Normalized distance map (H, W) float32 in [0, 1] range
    """
    # Create binary mask
    binary_mask = (mask > 0).astype(np.uint8)
    
    # Compute distance transform
    if SCIPY_AVAILABLE:
        dist_map = distance_transform_edt(binary_mask)
    else:
        dist_map = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5).astype(np.float32)
    
    # Normalize to [0, 1]
    if dist_map.max() > 0:
        dist_map = dist_map / dist_map.max()
    
    return dist_map.astype(np.float32)


def resize_with_padding(img, target_size=512, is_mask=False):
    """
    Resize image to target size while maintaining aspect ratio.
    Pads with zeros to make square.
    
    Parameters
    ----------
    img : numpy.ndarray
        Input image
    target_size : int
        Target size for both dimensions
    is_mask : bool
        Whether this is a mask (uses INTER_NEAREST)
    
    Returns
    -------
    numpy.ndarray
        Resized and padded image (target_size, target_size)
    """
    h, w = img.shape[:2]
    
    # Calculate scaling
    if h > w:
        scale = target_size / h
        new_h = target_size
        new_w = int(w * scale)
    else:
        scale = target_size / w
        new_w = target_size
        new_h = int(h * scale)
    
    # Resize
    interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    
    # Pad
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    img_padded = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, 
                                    cv2.BORDER_CONSTANT, value=0)
    
    return img_padded



