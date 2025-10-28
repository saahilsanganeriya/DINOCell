# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
JUMP Cell Painting S3 Streaming Dataset
========================================

Stream JUMP images directly from AWS S3 without downloading locally!

AWS S3 Bucket: s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/
Access: Public (no credentials needed with --no-sign-request)

Key Benefits:
- No local storage needed (saves ~500GB)
- Start training immediately
- Auto-caching of recently used images
- Progressive loading during training

Requirements:
    pip install boto3 smart_open pillow

Usage in config:
    train.dataset_path: JUMPS3:bucket=cellpainting-gallery:prefix=cpg0000-jump-pilot/source_4/images
"""

import os
import io
import logging
from pathlib import Path
from typing import Any, Callable, Optional, List
from functools import lru_cache
import numpy as np
from PIL import Image
import cv2

from .decoders import Decoder
from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov3")

# Try to import S3 dependencies
try:
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from smart_open import open as smart_open
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    logger.warning("boto3 or smart_open not available. Install with: pip install boto3 smart-open")


class S3ImageCache:
    """LRU cache for S3 images to reduce redundant downloads."""
    
    def __init__(self, cache_size=1000):
        self.cache_size = cache_size
        self._cache = {}
        self._access_order = []
    
    def get(self, key):
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            return self._cache[key]
        return None
    
    def put(self, key, value):
        if key in self._cache:
            self._access_order.remove(key)
        elif len(self._cache) >= self.cache_size:
            # Remove least recently used
            lru_key = self._access_order.pop(0)
            del self._cache[lru_key]
        
        self._cache[key] = value
        self._access_order.append(key)


class JUMPS3Decoder(Decoder):
    """Decoder that streams images from S3."""
    
    # Shared cache across all instances
    _image_cache = S3ImageCache(cache_size=1000)
    
    def __init__(self, s3_paths: list, s3_client, channel_mode='average', apply_clahe=True):
        """
        Parameters
        ----------
        s3_paths : list
            List of S3 keys for channel images
        s3_client : boto3.client
            S3 client for downloading
        channel_mode : str
            'average', 'multiview', 'brightfield', 'channel1', etc.
        apply_clahe : bool
            Apply CLAHE preprocessing
        """
        self._s3_paths = s3_paths
        self._s3_client = s3_client
        self._channel_mode = channel_mode
        self._apply_clahe = apply_clahe
    
    def _load_image_from_s3(self, s3_key, bucket='cellpainting-gallery'):
        """Load image from S3 with caching."""
        # Check cache first
        cached = self._image_cache.get(s3_key)
        if cached is not None:
            return cached
        
        try:
            # Download from S3
            response = self._s3_client.get_object(Bucket=bucket, Key=s3_key)
            img_data = response['Body'].read()
            
            # Decode image
            img_array = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
            
            # Cache it
            self._image_cache.put(s3_key, img)
            
            return img
            
        except Exception as e:
            logger.warning(f"Failed to load {s3_key}: {e}")
            return None
    
    def decode(self):
        """Load and process multi-channel images from S3."""
        # Load channels
        channels = []
        for s3_path in self._s3_paths[:5]:  # First 5 fluorescent channels
            img = self._load_image_from_s3(s3_path)
            if img is not None:
                # Apply CLAHE
                if self._apply_clahe:
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    img = clahe.apply(img)
                
                # Normalize
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                channels.append(img)
        
        # Return based on mode
        if self._channel_mode == 'multiview':
            # Return list of channel images for multi-view learning
            channel_pils = []
            for img in channels:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                channel_pils.append(Image.fromarray(img_rgb))
            return channel_pils if len(channel_pils) > 0 else [Image.new('RGB', (224, 224))]
        
        else:
            # Average channels
            if len(channels) == 0:
                return Image.new('RGB', (224, 224))
            
            img_avg = np.mean(channels, axis=0).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_avg, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(img_rgb)


class JUMPS3Dataset(ExtendedVisionDataset):
    """
    JUMP Cell Painting dataset streaming from AWS S3.
    
    No local storage required! Images streamed on-demand with LRU caching.
    
    S3 Structure:
        s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/
            ├── 2020_11_04_CPJUMP1/
            │   └── images/
            │       └── BR00117010__*/
            │           └── Images/
            │               ├── r01c01f01p01-ch1sk1fk1fl1.tiff
            │               └── ...
            └── ...
    
    Parameters
    ----------
    bucket : str
        S3 bucket name (default: 'cellpainting-gallery')
    prefix : str
        S3 prefix (default: 'cpg0000-jump-pilot/source_4/images')
    channel_mode : str
        'average', 'multiview', 'brightfield', etc.
    batches : list, optional
        Which batches to include
    max_samples : int, optional
        Limit samples for testing
    cache_size : int
        Number of images to cache (default: 1000)
    """
    
    def __init__(
        self,
        *,
        bucket: str = 'cellpainting-gallery',
        prefix: str = 'cpg0000-jump-pilot/source_4/images',
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        channel_mode: str = 'average',
        batches: Optional[list] = None,
        max_samples: Optional[int] = None,
        cache_size: int = 1000,
    ) -> None:
        """Initialize S3 streaming dataset."""
        
        if not S3_AVAILABLE:
            raise ImportError("boto3 and smart-open required for S3 streaming. Install with: pip install boto3 smart-open")
        
        # Create a simple decoder for dummy targets
        class DummyDecoder:
            def __init__(self, data):
                self.data = data
            def decode(self):
                return self.data
        
        super().__init__(
            root=None,
            transforms=transforms,
            transform=transform,
            target_transform=target_transform,
            image_decoder=JUMPS3Decoder,
            target_decoder=DummyDecoder
        )
        
        self.bucket = bucket
        self.prefix = prefix
        self.channel_mode = channel_mode
        
        # Initialize S3 client (no credentials needed - public bucket)
        self.s3_client = boto3.client(
            's3',
            config=Config(signature_version=UNSIGNED)  # No auth required
        )
        
        # Update cache size
        JUMPS3Decoder._image_cache = S3ImageCache(cache_size=cache_size)
        
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
        
        logger.info(f"Discovering JUMP images from S3...")
        logger.info(f"Bucket: s3://{bucket}")
        logger.info(f"Prefix: {prefix}")
        logger.info(f"Batches: {batches}")
        logger.info(f"Channel mode: {channel_mode}")
        logger.info(f"Cache size: {cache_size} images")
        
        # Discover all fields
        self._discover_s3_images(batches, max_samples)
        
        logger.info(f"Discovered {len(self.fields)} fields")
        logger.info(f"Streaming enabled - no local storage needed!")
    
    def _discover_s3_images(self, batches, max_samples):
        """Discover all field images in S3 without downloading."""
        self.fields = []
        
        for batch in batches:
            logger.info(f"Scanning S3 batch: {batch}")
            
            # List objects in this batch
            batch_prefix = f"{self.prefix}/{batch}/images/"
            
            try:
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=self.bucket, Prefix=batch_prefix)
                
                # Group by field
                fields_dict = {}
                for page in pages:
                    if 'Contents' not in page:
                        continue
                    
                    for obj in page['Contents']:
                        key = obj['Key']
                        
                        # Parse filename
                        if not key.endswith('.tiff'):
                            continue
                        
                        # Extract field identifier
                        # Example: .../r01c01f01p01-ch1sk1fk1fl1.tiff
                        filename = Path(key).name
                        parts = filename.split('-')
                        if len(parts) < 2:
                            continue
                        
                        well_field = parts[0]
                        field_key = well_field[:9]  # r01c01f01
                        
                        if field_key not in fields_dict:
                            fields_dict[field_key] = []
                        
                        fields_dict[field_key].append(key)
                        
                        # Limit if requested
                        if max_samples and len(fields_dict) >= max_samples:
                            break
                    
                    if max_samples and len(fields_dict) >= max_samples:
                        break
                
                # Add complete fields
                for field_key, s3_keys in fields_dict.items():
                    if len(s3_keys) >= 5:  # At least 5 channels
                        self.fields.append(sorted(s3_keys))
                
                logger.info(f"  Found {len(fields_dict)} fields in {batch}")
                
            except Exception as e:
                logger.error(f"Error scanning batch {batch}: {e}")
                continue
    
    def get_image_data(self, index: int) -> Any:
        """Return S3 paths and client for decoder."""
        return {'s3_paths': self.fields[index], 's3_client': self.s3_client, 'channel_mode': self.channel_mode, 'apply_clahe': True}
    
    def get_target(self, index: int) -> Any:
        """Return dummy target."""
        return 0
    
    def __len__(self) -> int:
        return len(self.fields)


# For compatibility with data loader
def create_jumps3_dataset(bucket=None, prefix=None, **kwargs):
    """Factory function for S3 streaming dataset."""
    return JUMPS3Dataset(bucket=bucket or 'cellpainting-gallery',
                        prefix=prefix or 'cpg0000-jump-pilot/source_4/images',
                        **kwargs)

