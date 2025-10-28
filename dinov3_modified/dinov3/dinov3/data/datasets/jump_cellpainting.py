# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
JUMP Cell Painting Dataset for DINOv3 Pretraining
==================================================

Custom dataset loader for JUMP Cell Painting Consortium data.

Images: ~3M fields across 6 batches
Channels per field:
  - 5 fluorescent channels (ch01-ch05)
  - 3 brightfield z-planes (ch06-ch08)

For DINOv3 pretraining, we convert multi-channel images to RGB by averaging
fluorescent channels or using specific channels.

Directory structure:
  root/
    ├── 2020_11_04_CPJUMP1/
    │   └── images/
    │       └── BR00117010__*/
    │           └── Images/
    │               ├── r01c01f01p01-ch1sk1fk1fl1.tiff
    │               ├── r01c01f01p01-ch2sk1fk1fl1.tiff
    │               └── ...
    └── ...other batches...
"""

import os
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
from PIL import Image
import cv2
import logging

from .decoders import Decoder, ImageDataDecoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")


class JUMPImageDecoder(Decoder):
    """Decoder for JUMP Cell Painting multi-channel images."""
    
    def __init__(self, image_paths: list, channel_mode='average'):
        """
        Parameters
        ----------
        image_paths : list
            List of paths to channel images
        channel_mode : str
            How to convert to RGB: 'average', 'brightfield', 'channel1', etc.
        """
        self._image_paths = image_paths
        self._channel_mode = channel_mode
    
    def decode(self) -> Image:
        """Load and convert multi-channel image to RGB."""
        # Load all channels
        channels = []
        for path in self._image_paths:
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    channels.append(img)
            except Exception as e:
                logger.warning(f"Could not load {path}: {e}")
                continue
        
        if len(channels) == 0:
            # Return blank image if nothing loaded
            return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Convert based on mode
        if self._channel_mode == 'average':
            # Average all available channels
            img_avg = np.mean(channels, axis=0).astype(np.uint8)
        elif self._channel_mode == 'brightfield':
            # Use middle brightfield plane (ch07, index 6)
            if len(channels) > 6:
                img_avg = channels[6]
            else:
                img_avg = channels[0]
        elif self._channel_mode.startswith('channel'):
            # Use specific channel
            ch_idx = int(self._channel_mode.replace('channel', '')) - 1
            if ch_idx < len(channels):
                img_avg = channels[ch_idx]
            else:
                img_avg = channels[0]
        else:
            # Default to average
            img_avg = np.mean(channels, axis=0).astype(np.uint8)
        
        # Apply CLAHE for cell enhancement (same as DINOCell)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img_avg)
        
        # Convert to RGB
        img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image
        return Image.fromarray(img_rgb)


class JUMPCellPainting(ExtendedVisionDataset):
    """
    JUMP Cell Painting dataset for DINOv3 self-supervised pretraining.
    
    Automatically discovers all plate/well/field images in the dataset.
    Converts multi-channel microscopy to RGB for DINOv3.
    """
    
    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        channel_mode: str = 'average',  # 'average', 'brightfield', 'channel1', etc.
        batches: Optional[list] = None,  # Which batches to include
    ) -> None:
        """
        Initialize JUMP Cell Painting dataset.
        
        Parameters
        ----------
        root : str
            Path to JUMP data root (contains batch folders)
        channel_mode : str
            How to convert multi-channel to RGB:
            - 'average': Average all 5 fluorescent channels
            - 'brightfield': Use middle brightfield plane
            - 'channel1', 'channel2', etc.: Use specific channel
        batches : list, optional
            Which batches to include (default: all)
        """
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=JUMPImageDecoder,
            target_decoder=lambda x: x  # No targets for SSL
        )
        
        self.channel_mode = channel_mode
        self.root = Path(root)
        
        # Default batches if not specified
        if batches is None:
            batches = [
                '2020_11_04_CPJUMP1',
                '2020_11_18_CPJUMP1_TimepointDay1',
                '2020_11_19_TimepointDay4',
                '2020_12_02_CPJUMP1_2WeeksTimePoint',
                '2020_12_07_CPJUMP1_4WeeksTimePoint',
                '2020_12_08_CPJUMP1_Bleaching'
            ]
        
        logger.info(f"Discovering JUMP Cell Painting images...")
        logger.info(f"Root: {root}")
        logger.info(f"Batches: {batches}")
        logger.info(f"Channel mode: {channel_mode}")
        
        # Discover all field images
        self._discover_images(batches)
        
        logger.info(f"Discovered {len(self.fields)} fields of view")
        logger.info(f"Total training samples: {len(self.fields)}")
    
    def _discover_images(self, batches):
        """Discover all field images across batches."""
        self.fields = []
        
        for batch in batches:
            batch_path = self.root / batch / 'images'
            
            if not batch_path.exists():
                logger.warning(f"Batch path not found: {batch_path}")
                continue
            
            logger.info(f"Scanning batch: {batch}")
            
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
                    # Parse filename: r01c01f01p01-ch1sk1fk1fl1.tiff
                    name = img_file.stem
                    parts = name.split('-')
                    if len(parts) < 2:
                        continue
                    
                    # Extract row, col, field
                    well_field = parts[0]  # e.g., 'r01c01f01p01'
                    channel = parts[1].split('sk')[0]  # e.g., 'ch1'
                    
                    # Create field key (unique per field of view)
                    field_key = well_field[:10]  # 'r01c01f01' (row, col, field)
                    
                    if field_key not in fields_dict:
                        fields_dict[field_key] = []
                    
                    fields_dict[field_key].append(str(img_file))
                
                # Add complete fields (have multiple channels)
                for field_key, channel_paths in fields_dict.items():
                    if len(channel_paths) >= 5:  # At least 5 channels
                        self.fields.append(sorted(channel_paths))
            
            logger.info(f"  Found {len(self.fields)} fields in {batch}")
    
    def get_image_data(self, index: int) -> Any:
        """Get image data for decoder."""
        return self.fields[index]
    
    def get_target(self, index: int) -> Any:
        """Return dummy target (no labels for SSL)."""
        return 0  # Dummy target for self-supervised learning
    
    def __len__(self) -> int:
        return len(self.fields)


# For compatibility, create a factory function
def create_jump_dataset(root, split=None, **kwargs):
    """
    Factory function for creating JUMP dataset.
    
    Parameters
    ----------
    root : str
        Path to JUMP data
    split : str
        Ignored (no splits in JUMP, all used for training)
    **kwargs : dict
        Additional arguments passed to JUMPCellPainting
    
    Returns
    -------
    JUMPCellPainting
        Dataset instance
    """
    return JUMPCellPainting(root=root, **kwargs)


