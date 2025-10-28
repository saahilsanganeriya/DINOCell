"""
DINOCell Dataset Classes
=========================

PyTorch Dataset classes for training DINOCell on cell segmentation data.
Compatible with pre-processed SAMCell dataset format.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from .preprocessing import preprocess_for_dinov3, random_crop_256, augment_image_and_distmap
import logging

logger = logging.getLogger(__name__)


class DINOCellDataset(Dataset):
    """
    Dataset for training DINOCell.
    
    Loads preprocessed images and distance maps from .npy files.
    Applies augmentation and random cropping during training.
    
    Compatible with SAMCell's dataset format:
    - imgs.npy: (N, H, W, 3) float32 array
    - dist_maps.npy: (N, H, W) float32 array
    """
    
    def __init__(self, imgs_path, dist_maps_path, augment=True, crop_size=256):
        """
        Initialize DINOCell dataset.
        
        Parameters
        ----------
        imgs_path : str
            Path to imgs.npy file
        dist_maps_path : str
            Path to dist_maps.npy file
        augment : bool
            Whether to apply data augmentation (default: True)
        crop_size : int
            Size of random crops (default: 256)
        """
        super().__init__()
        
        logger.info(f"Loading dataset from {imgs_path}")
        self.imgs = np.load(imgs_path)
        self.dist_maps = np.load(dist_maps_path)
        
        assert len(self.imgs) == len(self.dist_maps), \
            f"Mismatch: {len(self.imgs)} images vs {len(self.dist_maps)} distance maps"
        
        self.augment = augment
        self.crop_size = crop_size
        
        logger.info(f"Loaded {len(self.imgs)} samples")
        logger.info(f"Image shape: {self.imgs.shape}")
        logger.info(f"Distance map shape: {self.dist_maps.shape}")
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        """
        Get a training sample.
        
        Returns
        -------
        dict
            'pixel_values': (3, 256, 256) tensor, normalized for DINOv3
            'ground_truth_dist_map': (256, 256) tensor, distance map
        """
        # Get image and distance map
        img = self.imgs[idx].copy()  # (H, W, 3)
        dist_map = self.dist_maps[idx].copy()  # (H, W)
        
        # Ensure image is uint8
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Apply augmentation if enabled
        if self.augment:
            img, dist_map = augment_image_and_distmap(img, dist_map)
        
        # Random crop to training size
        img_crop, dist_map_crop = random_crop_256(img, dist_map, self.crop_size)
        
        # Convert image to grayscale for preprocessing
        if len(img_crop.shape) == 3:
            img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_crop
        
        # Preprocess for DINOv3
        pixel_values = preprocess_for_dinov3(img_gray).squeeze(0)  # (3, 256, 256)
        
        # Convert distance map to tensor
        ground_truth_dist_map = torch.from_numpy(dist_map_crop).float()  # (256, 256)
        
        return {
            'pixel_values': pixel_values,
            'ground_truth_dist_map': ground_truth_dist_map
        }


class DINOCellUnlabeledDataset(Dataset):
    """
    Dataset for self-supervised pretraining on unlabeled cell images.
    
    For DINOv3 pretraining using self-supervised learning objectives.
    """
    
    def __init__(self, imgs_path, crop_size=224, augment=True):
        """
        Initialize unlabeled dataset.
        
        Parameters
        ----------
        imgs_path : str
            Path to imgs.npy file or directory of images
        crop_size : int
            Size of crops for pretraining (default: 224 for DINOv3)
        augment : bool
            Whether to apply augmentation (default: True)
        """
        super().__init__()
        
        # Load images
        if imgs_path.endswith('.npy'):
            self.imgs = np.load(imgs_path)
        else:
            # Load from directory
            import os
            from pathlib import Path
            img_dir = Path(imgs_path)
            img_files = sorted(list(img_dir.glob('*.png')) + list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.tif')))
            self.imgs = []
            for img_file in img_files:
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    self.imgs.append(img)
            self.imgs = np.array(self.imgs)
        
        self.crop_size = crop_size
        self.augment = augment
        
        logger.info(f"Loaded {len(self.imgs)} unlabeled images for pretraining")
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        """
        Get an unlabeled image sample.
        
        Returns
        -------
        dict
            'pixel_values': (3, crop_size, crop_size) tensor
        """
        img = self.imgs[idx].copy()
        
        # Ensure uint8
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        # Ensure grayscale
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply augmentation (without distance map)
        if self.augment:
            # Random flip
            if np.random.rand() < 0.5:
                img = cv2.flip(img, 1)
            
            # Random rotation
            if np.random.rand() < 0.5:
                angle = np.random.uniform(-180, 180)
                h, w = img.shape
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            
            # Random brightness
            brightness = np.random.uniform(0.95, 1.05)
            img = np.clip(img * brightness, 0, 255).astype(np.uint8)
            
            # Random inversion
            if np.random.rand() < 0.5:
                img = 255 - img
        
        # Random crop
        h, w = img.shape
        if h < self.crop_size or w < self.crop_size:
            # Pad if too small
            pad_h = max(0, self.crop_size - h)
            pad_w = max(0, self.crop_size - w)
            img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)
            h, w = img.shape
        
        top = np.random.randint(0, h - self.crop_size + 1)
        left = np.random.randint(0, w - self.crop_size + 1)
        img_crop = img[top:top+self.crop_size, left:left+self.crop_size]
        
        # Preprocess for DINOv3
        pixel_values = preprocess_for_dinov3(img_crop).squeeze(0)
        
        return {'pixel_values': pixel_values}



