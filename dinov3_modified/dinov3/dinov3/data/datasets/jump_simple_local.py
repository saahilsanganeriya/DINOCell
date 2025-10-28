# Simple local JUMP dataset for testing
# Loads images from example_images/ directory

import os
import glob
import logging
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np
from PIL import Image
import cv2

from .decoders import Decoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")


class JUMPSimpleDecoder(Decoder):
    """Decoder for local JUMP images."""
    
    def __init__(self, channel_paths: list, apply_clahe=True):
        self.channel_paths = channel_paths
        self.apply_clahe = apply_clahe
    
    def decode(self):
        """Load and return list of channel images (multi-view mode)."""
        channels = []
        
        for path in self.channel_paths[:5]:  # First 5 channels (fluorescent)
            # Load TIFF
            img = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                logger.warning(f"Failed to load {path}")
                continue
            
            # Normalize to 0-255
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Apply CLAHE
            if self.apply_clahe:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                img = clahe.apply(img)
            
            # Convert to PIL RGB
            img_rgb = np.stack([img] * 3, axis=-1)
            pil_img = Image.fromarray(img_rgb)
            channels.append(pil_img)
        
        return channels  # Return list for multi-view augmentation


class JUMPSimpleLocal(ExtendedVisionDataset):
    """Simple local JUMP dataset for testing with example_images/"""
    
    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        Load JUMP images from simple directory structure.
        
        Expected structure:
            root/
                compound1/
                    field1-ch1.tiff
                    field1-ch2.tiff
                    ...
                compound2/
                    ...
        """
        # Create a simple decoder for dummy targets
        class DummyDecoder:
            def __init__(self, data):
                self.data = data
            def decode(self):
                return self.data
        
        super().__init__(
            root=root,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=JUMPSimpleDecoder,
            target_decoder=DummyDecoder
        )
        
        logger.info(f"Loading JUMP images from: {root}")
        
        # Discover all fields
        self.fields = []
        root_path = Path(root)
        
        # Find all TIFF files
        tiff_files = list(root_path.glob("**/*.tiff"))
        logger.info(f"Found {len(tiff_files)} TIFF files")
        
        # Group by field (r##c##f##p##)
        fields_dict = {}
        for tiff_path in tiff_files:
            filename = tiff_path.name
            # Extract field ID from filename like r01c21f05p01-ch1sk1fk1fl1.tiff
            parts = filename.split('-')
            if len(parts) < 2:
                continue
            
            field_id = parts[0]  # r01c21f05p01
            
            if field_id not in fields_dict:
                fields_dict[field_id] = {}
            
            # Extract channel number
            ch_part = parts[1]  # ch1sk1fk1fl1.tiff
            if 'ch' in ch_part:
                ch_num = int(ch_part.split('ch')[1][0])  # Get first digit after 'ch'
                fields_dict[field_id][ch_num] = str(tiff_path)
        
        # Create field list (need channels 1-5 for fluorescent)
        for field_id, channels in fields_dict.items():
            # Need at least channels 1-5
            if all(ch in channels for ch in [1, 2, 3, 4, 5]):
                channel_paths = [channels[i] for i in range(1, 9) if i in channels]
                self.fields.append(channel_paths)
        
        logger.info(f"Discovered {len(self.fields)} complete fields (with 5+ channels)")
        
        if len(self.fields) == 0:
            logger.warning("No fields found! Check directory structure")
    
    def get_image_data(self, index: int) -> Any:
        """Return channel paths for decoder."""
        return {'channel_paths': self.fields[index], 'apply_clahe': True}
    
    def get_target(self, index: int) -> Any:
        """Return dummy target."""
        return 0
    
    def __len__(self) -> int:
        return len(self.fields)

