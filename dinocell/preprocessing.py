"""
DINOCell Preprocessing Utilities
=================================

Image preprocessing functions for DINOCell, including:
- CLAHE contrast enhancement
- DINOv3-specific normalization
- Image augmentation for training
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import Optional, Tuple


# DINOv3 uses ImageNet normalization (from official repo)
DINOV3_MEAN = (0.485, 0.456, 0.406)
DINOV3_STD = (0.229, 0.224, 0.225)


def apply_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    Parameters
    ----------
    img : numpy.ndarray
        Input grayscale image (H, W)
    clip_limit : float
        Threshold for contrast limiting (default: 3.0)
    tile_grid_size : tuple
        Size of grid for histogram equalization (default: (8, 8))
    
    Returns
    -------
    numpy.ndarray
        CLAHE-enhanced image (H, W) uint8
    """
    if img.dtype != np.uint8:
        # Normalize to 0-255 range
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_clahe = clahe.apply(img)
    
    return img_clahe


def preprocess_for_dinov3(img, mean=DINOV3_MEAN, std=DINOV3_STD):
    """
    Preprocess image for DINOv3 inference.
    
    Converts grayscale to RGB, normalizes using ImageNet statistics.
    
    Parameters
    ----------
    img : numpy.ndarray
        Input image (H, W) uint8
    mean : tuple
        Mean values for normalization (default: ImageNet mean)
    std : tuple
        Std values for normalization (default: ImageNet std)
    
    Returns
    -------
    torch.Tensor
        Preprocessed image tensor (1, 3, H, W)
    """
    # Convert to RGB if grayscale
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2RGB)
    
    # Convert BGR to RGB if needed
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert to float and normalize to [0, 1]
    img = img.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    normalize = transforms.Normalize(mean=mean, std=std)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)
    img_tensor = normalize(img_tensor)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)  # (1, 3, H, W)
    
    return img_tensor


def preprocess_training_image(img):
    """
    Preprocess image for training (CLAHE + conversion to BGR).
    
    Parameters
    ----------
    img : numpy.ndarray
        Input image (H, W) or (H, W, C)
    
    Returns
    -------
    numpy.ndarray
        Preprocessed image (H, W, 3) uint8 in BGR format
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()
    
    # Normalize to 0-255
    img_norm = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply CLAHE
    img_clahe = apply_clahe(img_norm)
    
    # Convert to BGR (3 channels)
    img_bgr = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2BGR)
    
    return img_bgr


def augment_image_and_distmap(img, dist_map, flip_prob=0.5, rotate_prob=0.5,
                               scale_range=(0.8, 1.2), brightness_range=(0.95, 1.05),
                               invert_prob=0.5):
    """
    Apply data augmentation to image and distance map.
    
    Implements SAMCell's augmentation strategy:
    - Random horizontal flip
    - Random rotation (-180 to 180 degrees)
    - Random scale (0.8 to 1.2x)
    - Random brightness (0.95 to 1.05x)
    - Random inversion
    
    Parameters
    ----------
    img : numpy.ndarray
        Input image (H, W, 3)
    dist_map : numpy.ndarray
        Distance map (H, W)
    flip_prob : float
        Probability of horizontal flip
    rotate_prob : float
        Probability of rotation
    scale_range : tuple
        Range for random scaling
    brightness_range : tuple
        Range for brightness adjustment
    invert_prob : float
        Probability of image inversion
    
    Returns
    -------
    tuple
        (augmented_img, augmented_dist_map)
    """
    # Random horizontal flip
    if np.random.rand() < flip_prob:
        img = cv2.flip(img, 1)
        dist_map = cv2.flip(dist_map, 1)
    
    # Random rotation
    if np.random.rand() < rotate_prob:
        angle = np.random.uniform(-180, 180)
        h, w = img.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        dist_map = cv2.warpAffine(dist_map, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    
    # Random scale
    scale_factor = np.random.uniform(scale_range[0], scale_range[1])
    h, w = img.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dist_map = cv2.resize(dist_map, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Random crop back to original size
    if scale_factor > 1.0:
        # Crop from center
        start_h = (new_h - h) // 2
        start_w = (new_w - w) // 2
        img = img[start_h:start_h+h, start_w:start_w+w]
        dist_map = dist_map[start_h:start_h+h, start_w:start_w+w]
    elif scale_factor < 1.0:
        # Pad to original size
        pad_h = (h - new_h) // 2
        pad_w = (w - new_w) // 2
        img = cv2.copyMakeBorder(img, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, 
                                cv2.BORDER_REFLECT)
        dist_map = cv2.copyMakeBorder(dist_map, pad_h, h-new_h-pad_h, pad_w, w-new_w-pad_w, 
                                     cv2.BORDER_CONSTANT, value=0)
    
    # Random brightness
    brightness_factor = np.random.uniform(brightness_range[0], brightness_range[1])
    img = np.clip(img * brightness_factor, 0, 255).astype(np.uint8)
    
    # Random inversion
    if np.random.rand() < invert_prob:
        img = 255 - img
    
    return img, dist_map


def random_crop_256(img, dist_map, crop_size=256):
    """
    Extract random 256x256 crop from image and distance map.
    
    Parameters
    ----------
    img : numpy.ndarray
        Input image (H, W, 3)
    dist_map : numpy.ndarray
        Distance map (H, W)
    crop_size : int
        Size of crop (default: 256)
    
    Returns
    -------
    tuple
        (cropped_img, cropped_dist_map)
    """
    h, w = img.shape[:2]
    
    if h < crop_size or w < crop_size:
        # Pad if image is too small
        pad_h = max(0, crop_size - h)
        pad_w = max(0, crop_size - w)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
        dist_map = cv2.copyMakeBorder(dist_map, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        h, w = img.shape[:2]
    
    # Random crop position
    top = np.random.randint(0, h - crop_size + 1)
    left = np.random.randint(0, w - crop_size + 1)
    
    img_crop = img[top:top+crop_size, left:left+crop_size]
    dist_map_crop = dist_map[top:top+crop_size, left:left+crop_size]
    
    return img_crop, dist_map_crop



