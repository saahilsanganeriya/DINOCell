# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
JUMP Cell Painting Multi-View Dataset
======================================

Advanced dataset loader that returns individual channels for multi-view consistency learning.

Key Innovation:
- Returns list of channel images instead of averaging
- Allows DINO to learn "same cell" across different channels
- Global crop 1 = averaged channels
- Global crop 2 = random single channel  
- DINO loss enforces: avg_features ≈ single_channel_features
- Result: Channel-invariant cell representations!

This is superior to simple averaging because it explicitly teaches the model
that ch1, ch2, ch3, ch4, ch5 all show the SAME cell.
"""

import os
from pathlib import Path
from typing import Any, Callable, Optional, List
import numpy as np
from PIL import Image
import cv2
import logging

from .decoders import Decoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")


class JUMPMultiViewDecoder(Decoder):
    """
    Decoder that returns individual channel images for multi-view learning.
    """
    
    def __init__(self, image_paths: list, apply_clahe: bool = True):
        """
        Parameters
        ----------
        image_paths : list
            List of paths to channel images (sorted: ch1, ch2, ..., ch8)
        apply_clahe : bool
            Whether to apply CLAHE preprocessing
        """
        self._image_paths = image_paths
        self._apply_clahe = apply_clahe
    
    def decode(self) -> List[Image.Image]:
        """
        Load all channels and return as list of PIL Images.
        
        Returns
        -------
        list of PIL.Image
            List of channel images (grayscale converted to RGB)
        """
        channel_images = []
        
        # Load first 5 channels (fluorescent)
        for i, path in enumerate(self._image_paths[:5]):  # Only use first 5 fluorescent
            try:
                # Load as grayscale
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                
                if img is None:
                    logger.warning(f"Could not load {path}")
                    continue
                
                # Apply CLAHE if requested
                if self._apply_clahe:
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    img = clahe.apply(img)
                
                # Normalize
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # Convert to RGB (required for DINOv3)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                
                # Convert to PIL
                img_pil = Image.fromarray(img_rgb)
                channel_images.append(img_pil)
                
            except Exception as e:
                logger.warning(f"Error loading {path}: {e}")
                continue
        
        # Fallback: if no channels loaded, return blank
        if len(channel_images) == 0:
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            channel_images = [Image.fromarray(blank)]
        
        # Return list of channel images
        return channel_images


class JUMPCellPaintingMultiView(ExtendedVisionDataset):
    """
    JUMP Cell Painting dataset for multi-view self-supervised learning.
    
    Returns individual channels as a list, allowing augmentation to create
    different views from different channels.
    
    Usage with MultiChannelDataAugmentationDINO:
    - Global crop 1: Uses averaged channels
    - Global crop 2: Uses random single channel
    - Local crops: Use random channels
    - DINO loss enforces consistency
    """
    
    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        apply_clahe: bool = True,
        batches: Optional[list] = None,
        max_samples: Optional[int] = None,  # For testing with subset
    ) -> None:
        """
        Initialize JUMP multi-view dataset.
        
        Parameters
        ----------
        root : str
            Path to JUMP data root
        apply_clahe : bool
            Whether to apply CLAHE preprocessing
        batches : list, optional
            Which batches to include
        max_samples : int, optional
            Limit number of samples (for testing)
        """
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=JUMPMultiViewDecoder,
            target_decoder=lambda x: x
        )
        
        self.apply_clahe = apply_clahe
        self.root = Path(root)
        
        # Default batches
        if batches is None:
            batches = [
                '2020_11_04_CPJUMP1',
                '2020_11_18_CPJUMP1_TimepointDay1',
                '2020_11_19_TimepointDay4',
                '2020_12_02_CPJUMP1_2WeeksTimePoint',
                '2020_12_07_CPJUMP1_4WeeksTimePoint',
                '2020_12_08_CPJUMP1_Bleaching'
            ]
        
        logger.info(f"Discovering JUMP Cell Painting images (Multi-View mode)...")
        logger.info(f"Root: {root}")
        logger.info(f"Batches: {batches}")
        logger.info(f"CLAHE: {apply_clahe}")
        
        # Discover all field images
        self._discover_images(batches, max_samples)
        
        logger.info(f"Discovered {len(self.fields)} fields of view")
        logger.info(f"Mode: Multi-View (returns {self.channels_per_field} channels per field)")
    
    def _discover_images(self, batches, max_samples):
        """Discover all field images and group by channels."""
        self.fields = []
        self.channels_per_field = 0
        
        for batch in batches:
            batch_path = self.root / batch / 'images'
            
            if not batch_path.exists():
                logger.warning(f"Batch not found: {batch_path}")
                continue
            
            logger.info(f"Scanning batch: {batch}")
            batch_fields = 0
            
            # Find all plate directories
            for plate_dir in batch_path.iterdir():
                if not plate_dir.is_dir():
                    continue
                
                images_dir = plate_dir / 'Images'
                if not images_dir.exists():
                    continue
                
                # Group images by field (row, col, field)
                fields_dict = {}
                for img_file in images_dir.glob('*.tiff'):
                    # Parse: r01c01f01p01-ch1sk1fk1fl1.tiff
                    name = img_file.stem
                    parts = name.split('-')
                    if len(parts) < 2:
                        continue
                    
                    # Extract field key (r01c01f01)
                    well_field = parts[0]
                    field_key = well_field[:9]  # r01c01f01
                    
                    if field_key not in fields_dict:
                        fields_dict[field_key] = []
                    
                    fields_dict[field_key].append(str(img_file))
                
                # Add fields with sufficient channels
                for field_key, channel_paths in fields_dict.items():
                    if len(channel_paths) >= 5:  # Need at least 5 channels
                        # Sort to ensure consistent channel order
                        sorted_paths = sorted(channel_paths)
                        self.fields.append(sorted_paths)
                        batch_fields += 1
                        
                        if self.channels_per_field == 0:
                            self.channels_per_field = len(sorted_paths)
                        
                        # Limit samples if requested (for testing)
                        if max_samples and len(self.fields) >= max_samples:
                            logger.info(f"Reached max_samples limit: {max_samples}")
                            return
            
            logger.info(f"  Found {batch_fields} fields in {batch}")
    
    def get_image_data(self, index: int) -> Any:
        """Return channel paths for decoder."""
        return self.fields[index]
    
    def get_target(self, index: int) -> Any:
        """Return dummy target (no labels for SSL)."""
        return 0
    
    def __len__(self) -> int:
        return len(self.fields)
    
    def get_image_decoder(self, index: int) -> Decoder:
        """Create decoder for this sample."""
        return JUMPMultiViewDecoder(self.fields[index], self.apply_clahe)


# For compatibility with data loader registration
def create_jump_multiview_dataset(root, split=None, **kwargs):
    """Factory function for JUMP multi-view dataset."""
    return JUMPCellPaintingMultiView(root=root, **kwargs)

