# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Multi-Channel Data Augmentation for DINOv3
===========================================

Custom augmentation for multi-channel microscopy images (e.g., JUMP Cell Painting).

Strategy:
- Treat different channels as different "views" of the same cell
- Global crop 1: Average of all channels
- Global crop 2: Random single channel
- Local crops: Random channels
- DINO loss enforces consistency: model learns "same cell regardless of channel"

This is similar to multi-view contrastive learning but using DINO's framework.
"""

import logging
import random
import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2

from dinov3.data.transforms import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, GaussianBlur, make_normalize_transform

logger = logging.getLogger("dinov3")


class MultiChannelDataAugmentationDINO(object):
    """
    Data augmentation for multi-channel microscopy with channel-consistency learning.
    
    For datasets where each sample has multiple channels showing the same content
    (e.g., JUMP Cell Painting: 5 fluorescent + 3 brightfield).
    
    Key idea: Enforce consistency between different channel views via DINO loss.
    """
    
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
        num_channels=5,  # Number of channels to use
        channel_selection_mode='random',  # 'random', 'sequential', 'all'
        share_color_jitter=False,
        horizontal_flips=True,
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.num_channels = num_channels
        self.channel_selection_mode = channel_selection_mode
        self.share_color_jitter = share_color_jitter
        self.mean = mean
        self.std = std
        
        logger.info("###################################")
        logger.info("Using Multi-Channel data augmentation:")
        logger.info(f"num_channels: {num_channels}")
        logger.info(f"channel_selection_mode: {channel_selection_mode}")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info(f"horizontal flips: {horizontal_flips}")
        logger.info("###################################")
        
        # Geometric augmentations
        self.geometric_augmentation_global = v2.Compose([
            v2.RandomResizedCrop(
                global_crops_size,
                scale=global_crops_scale,
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
        ])
        
        self.geometric_augmentation_local = v2.Compose([
            v2.RandomResizedCrop(
                local_crops_size,
                scale=local_crops_scale,
                interpolation=v2.InterpolationMode.BICUBIC,
            ),
            v2.RandomHorizontalFlip(p=0.5 if horizontal_flips else 0.0),
        ])
        
        # Color augmentations (adapted for microscopy)
        color_jittering = v2.Compose([
            v2.RandomApply(
                [v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1, hue=0.05)],
                p=0.8,
            ),
            v2.RandomGrayscale(p=0.1),  # Lower prob for microscopy
        ])
        
        global_transfo1_extra = GaussianBlur(p=1.0)
        global_transfo2_extra = v2.Compose([
            GaussianBlur(p=0.1),
            v2.RandomSolarize(threshold=128, p=0.2),
        ])
        local_transfo_extra = GaussianBlur(p=0.5)
        
        # Normalization
        self.normalize = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            make_normalize_transform(mean=mean, std=std),
        ])
        
        if self.share_color_jitter:
            self.color_jittering = color_jittering
            self.global_transfo1 = v2.Compose([global_transfo1_extra, self.normalize])
            self.global_transfo2 = v2.Compose([global_transfo2_extra, self.normalize])
            self.local_transfo = v2.Compose([local_transfo_extra, self.normalize])
        else:
            self.global_transfo1 = v2.Compose([color_jittering, global_transfo1_extra, self.normalize])
            self.global_transfo2 = v2.Compose([color_jittering, global_transfo2_extra, self.normalize])
            self.local_transfo = v2.Compose([color_jittering, local_transfo_extra, self.normalize])
    
    def _select_channel(self, image_channels, mode='random'):
        """
        Select which channel(s) to use.
        
        Parameters
        ----------
        image_channels : PIL.Image or list of PIL.Image
            If list: multiple channel images
            If single: already averaged/selected
        mode : str
            'random': Random single channel
            'average': Average all channels
            'sequential': Cycle through channels
        
        Returns
        -------
        PIL.Image
            Selected/combined channel
        """
        # If already a single image, return it
        if not isinstance(image_channels, list):
            return image_channels
        
        # If list of channels, select based on mode
        if mode == 'random':
            idx = random.randint(0, len(image_channels) - 1)
            return image_channels[idx]
        elif mode == 'average':
            # Average channels
            arrays = [np.array(img) for img in image_channels]
            avg = np.mean(arrays, axis=0).astype(np.uint8)
            from PIL import Image
            return Image.fromarray(avg)
        elif mode == 'sequential':
            # Use modulo to cycle through
            idx = random.randint(0, len(image_channels) - 1)
            return image_channels[idx]
        else:
            return image_channels[0]
    
    def __call__(self, image_channels):
        """
        Apply multi-channel augmentation.
        
        Parameters
        ----------
        image_channels : list of PIL.Image or PIL.Image
            If list: [ch1, ch2, ch3, ch4, ch5] - one per channel
            If single: already processed image
        
        Returns
        -------
        dict
            'global_crops': [crop1, crop2] where crop1=avg channels, crop2=random channel
            'local_crops': [crop1, ..., crop8] with random channels
            'global_crops_teacher': Same as global_crops
        """
        output = {}
        
        # GLOBAL CROP 1: Average of all channels
        # This provides a complete view combining all channel information
        if isinstance(image_channels, list):
            image_avg = self._select_channel(image_channels, mode='average')
        else:
            image_avg = image_channels
        
        im1_base = self.geometric_augmentation_global(image_avg)
        global_crop_1 = self.global_transfo1(im1_base)
        
        # GLOBAL CROP 2: Random single channel
        # This provides a channel-specific view
        if isinstance(image_channels, list):
            image_single = self._select_channel(image_channels, mode='random')
        else:
            image_single = image_channels
        
        im2_base = self.geometric_augmentation_global(image_single)
        global_crop_2 = self.global_transfo2(im2_base)
        
        output["global_crops"] = [global_crop_1, global_crop_2]
        
        # For teacher, use the same crops (no additional augmentation)
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]
        
        # LOCAL CROPS: Random channels for diversity
        local_crops = []
        for _ in range(self.local_crops_number):
            # Each local crop uses a random channel (or average)
            if isinstance(image_channels, list):
                if random.random() < 0.3:  # 30% chance to use average
                    image_local = self._select_channel(image_channels, mode='average')
                else:  # 70% chance to use random single channel
                    image_local = self._select_channel(image_channels, mode='random')
            else:
                image_local = image_channels
            
            local_crop = self.local_transfo(self.geometric_augmentation_local(image_local))
            local_crops.append(local_crop)
        
        output["local_crops"] = local_crops
        output["offsets"] = ()  # No offsets for standard crops
        
        return output


def create_multichannel_augmentation(cfg):
    """
    Factory function to create multi-channel augmentation from config.
    
    Parameters
    ----------
    cfg : OmegaConf
        Configuration object with crops settings
    
    Returns
    -------
    MultiChannelDataAugmentationDINO
        Augmentation instance
    """
    return MultiChannelDataAugmentationDINO(
        global_crops_scale=cfg.crops.global_crops_scale,
        local_crops_scale=cfg.crops.local_crops_scale,
        local_crops_number=cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        num_channels=getattr(cfg.crops, 'num_channels', 5),
        channel_selection_mode=getattr(cfg.crops, 'channel_selection_mode', 'random'),
        share_color_jitter=cfg.crops.share_color_jitter,
        horizontal_flips=cfg.crops.horizontal_flips,
        mean=cfg.crops.rgb_mean,
        std=cfg.crops.rgb_std,
    )

