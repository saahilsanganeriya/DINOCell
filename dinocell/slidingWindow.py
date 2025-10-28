"""
Sliding Window Helper for DINOCell
===================================

Adapted from SAMCell's sliding window implementation.
Handles splitting images into overlapping patches and recombining predictions.
"""

import cv2
import numpy as np


class SlidingWindowHelper:
    """
    Helper class for sliding window inference with smooth blending.
    
    Uses cosine falloff for overlapping regions to avoid edge artifacts.
    """
    
    def __init__(self, crop_size: int, overlap_size: int):
        """
        Initialize sliding window helper.
        
        Parameters
        ----------
        crop_size : int
            Size of each crop (default: 256)
        overlap_size : int
            Overlap between adjacent crops (default: 32)
        """
        self.crop_size = crop_size
        self.overlap_size = overlap_size
        self._create_blending_mask()
    
    def _create_blending_mask(self):
        """Create cosine blending mask for smooth transitions."""
        mask = np.ones((self.crop_size, self.crop_size), dtype=np.float32)
        
        # Cosine falloff for edges
        for i in range(self.overlap_size):
            # Weight increases from edge to center
            weight = 0.5 * (1 - np.cos(np.pi * i / self.overlap_size))
            
            # Apply to all four edges
            mask[i, :] *= weight  # Top
            mask[-i-1, :] *= weight  # Bottom
            mask[:, i] *= weight  # Left
            mask[:, -i-1] *= weight  # Right
        
        self.blending_mask = mask
    
    def separate_into_crops(self, img):
        """
        Separate image into overlapping crops.
        
        Parameters
        ----------
        img : numpy.ndarray
            Input image (H, W)
        
        Returns
        -------
        tuple
            (cropped_images, orig_regions, crop_unique_region)
            - cropped_images: List of image crops
            - orig_regions: List of (x, y, w, h) tuples in original image
            - crop_unique_region: (x, y, w, h) of non-overlapping region in each crop
        """
        orig_height, orig_width = img.shape
        
        # Handle small images
        if orig_height < self.crop_size or orig_width < self.crop_size:
            new_crop_size = min(orig_height, orig_width)
            if new_crop_size < 2 * self.overlap_size:
                return [img], [(0, 0, orig_width, orig_height)], (0, 0, orig_width, orig_height)
            new_overlap_size = self.overlap_size
        else:
            new_crop_size = self.crop_size
            new_overlap_size = self.overlap_size
        
        # Calculate stride
        stride = new_crop_size - 2 * new_overlap_size
        
        # Mirror borders
        img_mirrored = cv2.copyMakeBorder(
            img, new_overlap_size, new_overlap_size, 
            new_overlap_size, new_overlap_size, 
            cv2.BORDER_REFLECT
        )
        
        mirrored_height, mirrored_width = img_mirrored.shape
        
        # Calculate number of crops
        num_crops_y = max(1, int(np.ceil(orig_height / stride)))
        num_crops_x = max(1, int(np.ceil(orig_width / stride)))
        
        cropped_images = []
        orig_regions = []
        crop_unique_region = (new_overlap_size, new_overlap_size, 
                             new_crop_size - 2 * new_overlap_size, 
                             new_crop_size - 2 * new_overlap_size)
        
        for y_idx in range(num_crops_y):
            for x_idx in range(num_crops_x):
                # Calculate crop position
                if y_idx < num_crops_y - 1:
                    y_start = y_idx * stride + new_overlap_size
                else:
                    y_start = mirrored_height - new_crop_size
                
                if x_idx < num_crops_x - 1:
                    x_start = x_idx * stride + new_overlap_size
                else:
                    x_start = mirrored_width - new_crop_size
                
                # Extract crop
                y_end = y_start + new_crop_size
                x_end = x_start + new_crop_size
                crop = img_mirrored[y_start:y_end, x_start:x_end]
                
                # Calculate corresponding region in original image
                orig_x = max(0, x_start - new_overlap_size)
                orig_y = max(0, y_start - new_overlap_size)
                orig_w = min(new_crop_size, orig_width - orig_x)
                orig_h = min(new_crop_size, orig_height - orig_y)
                
                # Clamp to valid range
                orig_x = min(orig_x, orig_width)
                orig_y = min(orig_y, orig_height)
                orig_w = max(0, min(orig_w, orig_width - orig_x))
                orig_h = max(0, min(orig_h, orig_height - orig_y))
                
                if orig_w > 0 and orig_h > 0:
                    cropped_images.append(crop)
                    orig_regions.append((orig_x, orig_y, orig_w, orig_h))
        
        return cropped_images, orig_regions, crop_unique_region
    
    def combine_crops(self, orig_size, cropped_images, orig_regions, crop_unique_region, 
                     sam_outputs=None, dist_maps=None):
        """
        Combine overlapping crops with smooth blending.
        
        Parameters
        ----------
        orig_size : tuple
            Original image size (H, W)
        cropped_images : list
            List of image crops
        orig_regions : list
            List of (x, y, w, h) tuples
        crop_unique_region : tuple
            Unused (kept for compatibility)
        sam_outputs : list, optional
            Distance map predictions for each crop
        dist_maps : list, optional
            Alias for sam_outputs (for compatibility)
        
        Returns
        -------
        numpy.ndarray
            Combined distance map (H, W)
        """
        # For backward compatibility
        if sam_outputs is None and dist_maps is not None:
            sam_outputs = dist_maps
        
        # Initialize output and weights
        output_img = np.zeros(orig_size, dtype=np.float32)
        weight_map = np.zeros(orig_size, dtype=np.float32)
        
        if not cropped_images:
            return output_img
        
        # Get crop size
        crop_h, crop_w = cropped_images[0].shape
        
        # Create blending mask if needed
        if (crop_h != self.crop_size or crop_w != self.crop_size):
            temp_crop_size = self.crop_size
            self.crop_size = max(crop_h, crop_w)
            self._create_blending_mask()
            blending_mask = self.blending_mask
            self.crop_size = temp_crop_size
        else:
            blending_mask = self.blending_mask
        
        # Ensure mask matches crop size
        if blending_mask.shape != (crop_h, crop_w):
            blending_mask = cv2.resize(blending_mask, (crop_w, crop_h))
        
        # Blend crops
        for i, (crop, region) in enumerate(zip(cropped_images, orig_regions)):
            x, y, w, h = region
            
            if w <= 0 or h <= 0:
                continue
            
            # Use distance map output if provided
            if sam_outputs is not None:
                output = sam_outputs[i]
                if output.shape != crop.shape:
                    output = cv2.resize(output, (crop.shape[1], crop.shape[0]), 
                                      interpolation=cv2.INTER_LINEAR)
                crop_to_use = output
            else:
                crop_to_use = crop
            
            # Calculate mask region
            mask_h = min(h, blending_mask.shape[0])
            mask_w = min(w, blending_mask.shape[1])
            
            y_end = min(y + mask_h, orig_size[0])
            x_end = min(x + mask_w, orig_size[1])
            mask_h = y_end - y
            mask_w = x_end - x
            
            if mask_h <= 0 or mask_w <= 0:
                continue
            
            # Extract regions
            crop_region = crop_to_use[:mask_h, :mask_w]
            mask_region = blending_mask[:mask_h, :mask_w]
            
            # Apply blending
            output_img[y:y_end, x:x_end] += crop_region * mask_region
            weight_map[y:y_end, x:x_end] += mask_region
        
        # Normalize
        mask = weight_map > 0.0001
        output_img[mask] /= weight_map[mask]
        
        # Fill any remaining gaps with nearest neighbor
        if np.any(~mask):
            valid_mask = np.zeros(orig_size, dtype=np.uint8)
            valid_mask[mask] = 1
            
            dist, indices = cv2.distanceTransformWithLabels(
                1 - valid_mask, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
            
            h, w = orig_size
            coords_y, coords_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            nearest_y = coords_y.flatten()[indices.flatten() - 1].reshape(h, w)
            nearest_x = coords_x.flatten()[indices.flatten() - 1].reshape(h, w)
            
            for y in range(h):
                for x in range(w):
                    if not mask[y, x]:
                        ny, nx = nearest_y[y, x], nearest_x[y, x]
                        if 0 <= ny < h and 0 <= nx < w:
                            output_img[y, x] = output_img[ny, nx]
        
        return output_img
    
    # Backward compatibility
    def seperate_into_crops(self, img):
        return self.separate_into_crops(img)



