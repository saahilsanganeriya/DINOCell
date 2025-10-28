"""
DINOCell Inference Pipeline
============================

Sliding window inference pipeline with watershed post-processing for cell segmentation.

Based on SAMCell's approach but adapted for DINOCell:
1. CLAHE preprocessing
2. Sliding window with 256x256 patches (32px overlap)
3. Distance map prediction using DINOCell
4. Watershed post-processing to extract cell masks

Compatible with SAMCell's interface for easy comparison.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from skimage.segmentation import watershed
from tqdm import tqdm
import logging
import traceback
from typing import Optional, Tuple, List, Dict

from .slidingWindow import SlidingWindowHelper
from .preprocessing import preprocess_for_dinov3, apply_clahe

logger = logging.getLogger(__name__)


class DINOCellPipeline:
    """
    DINOCell inference pipeline with sliding window and watershed post-processing.
    
    Compatible interface with SAMCell's SlidingWindowPipeline for easy comparison.
    """
    
    def __init__(self, model, device, crop_size=256, cells_max=0.47, cell_fill=0.09):
        """
        Initialize the DINOCell pipeline.
        
        Parameters
        ----------
        model : DINOCell
            DINOCell model instance
        device : str or torch.device
            Device to run inference on ('cuda' or 'cpu')
        crop_size : int
            Size of sliding window patches (default: 256)
        cells_max : float
            Cell peak threshold for watershed (default: 0.47)
        cell_fill : float
            Cell fill threshold for watershed (default: 0.09)
        """
        try:
            logger.info("Initializing DINOCellPipeline")
            
            if hasattr(model, 'get_model'):
                self.model = model.get_model()
            else:
                self.model = model
            
            self.device = device
            self.crop_size = crop_size
            self.sigmoid = nn.Sigmoid()
            self.sliding_window_helper = SlidingWindowHelper(crop_size, 32)
            
            # Default thresholds (same as SAMCell)
            self.cells_max_threshold = cells_max
            self.cell_fill_threshold = cell_fill
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("DINOCellPipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing pipeline: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _preprocess(self, img):
        """
        Preprocess image for DINOv3.
        
        Parameters
        ----------
        img : numpy.ndarray
            Input grayscale image (H, W)
        
        Returns
        -------
        torch.Tensor
            Preprocessed image tensor (1, 3, H, W)
        """
        try:
            if img is None or img.size == 0:
                raise ValueError("Input image is empty")
            
            # Apply CLAHE
            img_clahe = apply_clahe(img)
            
            # Preprocess for DINOv3 (normalization, to tensor, etc.)
            img_tensor = preprocess_for_dinov3(img_clahe)
            
            return img_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def get_model_prediction(self, image):
        """
        Get distance map prediction for a single 256x256 patch.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input patch (256, 256) or (256, 256, 3)
        
        Returns
        -------
        numpy.ndarray
            Predicted distance map (256, 256)
        """
        try:
            # Ensure grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Preprocess
            image_tensor = self._preprocess(image)  # (1, 3, 256, 256)
            
            # Forward pass
            with torch.no_grad():
                dist_map_pred = self.model(image_tensor)  # (1, 1, 256, 256)
            
            # Apply sigmoid to get values in [0, 1]
            dist_map = self.sigmoid(dist_map_pred)[0, 0]  # (256, 256)
            
            return dist_map.cpu().numpy()
            
        except Exception as e:
            logger.error(f"Error in get_model_prediction: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def predict_on_full_img(self, image_orig):
        """
        Predict distance map on full image using sliding window.
        
        Parameters
        ----------
        image_orig : numpy.ndarray
            Original image (H, W)
        
        Returns
        -------
        numpy.ndarray
            Predicted distance map (H, W)
        """
        try:
            orig_shape = image_orig.shape
            
            # Split into overlapping crops
            crops, orig_regions, crop_unique_region = self.sliding_window_helper.separate_into_crops(image_orig)
            
            logger.info(f"Processing {len(crops)} crops...")
            
            # Predict on each crop
            dist_maps = []
            for crop in tqdm(crops, desc="Predicting", disable=len(crops) < 10):
                dist_map = self.get_model_prediction(crop)
                dist_maps.append(dist_map)
            
            # Combine crops with blending
            cell_dist_map = self.sliding_window_helper.combine_crops(
                orig_shape, crops, orig_regions, crop_unique_region, dist_maps
            )
            
            return cell_dist_map
            
        except Exception as e:
            logger.error(f"Error in predict_on_full_img: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def cells_from_dist_map(self, dist_map, cells_max_threshold=None, cell_fill_threshold=None):
        """
        Extract cell masks from distance map using watershed algorithm.
        
        Parameters
        ----------
        dist_map : numpy.ndarray
            Predicted distance map (H, W)
        cells_max_threshold : float, optional
            Threshold for cell peaks (default: self.cells_max_threshold)
        cell_fill_threshold : float, optional
            Threshold for cell boundaries (default: self.cell_fill_threshold)
        
        Returns
        -------
        numpy.ndarray
            Labeled cell masks (H, W) int32
        """
        try:
            # Use provided thresholds or defaults
            cells_max_threshold = cells_max_threshold if cells_max_threshold is not None else self.cells_max_threshold
            cell_fill_threshold = cell_fill_threshold if cell_fill_threshold is not None else self.cell_fill_threshold
            
            # Apply thresholds
            cells_max = dist_map > cells_max_threshold
            cell_fill = dist_map > cell_fill_threshold
            
            logger.info(f"cells_max threshold: {cells_max_threshold}, sum: {np.sum(cells_max)}")
            logger.info(f"cell_fill threshold: {cell_fill_threshold}, sum: {np.sum(cell_fill)}")
            
            # Convert to binary masks
            cells_max = cells_max.astype(np.uint8)
            cell_fill = cell_fill.astype(np.uint8)
            
            # Find contours for cell centers
            try:
                contours, _ = cv2.findContours(cells_max, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            except:
                _, contours, _ = cv2.findContours(cells_max, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            logger.info(f"Found {len(contours)} cell centers")
            
            # Create marker image for watershed
            markers = np.zeros(dist_map.shape, dtype=np.int32)
            
            for i, contour in enumerate(contours):
                if contour is None or len(contour) < 3:
                    continue
                
                try:
                    # Calculate centroid
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    else:
                        # Fallback to contour center
                        cX = int(np.mean([p[0][0] for p in contour]))
                        cY = int(np.mean([p[0][1] for p in contour]))
                    
                    # Clamp to valid range
                    cY = min(max(0, cY), markers.shape[0] - 1)
                    cX = min(max(0, cX), markers.shape[1] - 1)
                    
                    # Set marker
                    markers[cY, cX] = i + 1
                    
                except Exception as e:
                    logger.error(f"Error processing contour {i}: {str(e)}")
                    continue
            
            # Apply watershed
            if np.max(markers) == 0:
                logger.warning("No cell centers found - returning empty segmentation")
                return np.zeros(dist_map.shape, dtype=np.int32)
            
            labels = watershed(-dist_map, markers, mask=cell_fill).astype(np.int32)
            
            num_cells = len(np.unique(labels)) - 1  # -1 for background
            logger.info(f"Watershed segmentation complete. Found {num_cells} cells.")
            
            return labels
            
        except Exception as e:
            logger.error(f"Error in cells_from_dist_map: {str(e)}")
            logger.error(traceback.format_exc())
            return np.zeros(dist_map.shape, dtype=np.int32)
    
    def run(self, image, return_dist_map=False, cells_max=None, cell_fill=None):
        """
        Run the DINOCell pipeline on an input image.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input grayscale image (H, W)
        return_dist_map : bool
            Whether to also return the distance map (default: False)
        cells_max : float, optional
            Override default cells_max threshold
        cell_fill : float, optional
            Override default cell_fill threshold
        
        Returns
        -------
        numpy.ndarray
            Labeled cell masks (H, W)
        numpy.ndarray, optional
            Distance map if return_dist_map=True
        """
        try:
            # Make copy to avoid modifying original
            image = image.copy()
            
            # Ensure grayscale
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Predict distance map
            logger.info(f"Running DINOCell on image of shape {image.shape}")
            dist_map = self.predict_on_full_img(image)
            
            # Update thresholds if provided
            self.cells_max_threshold = cells_max if cells_max is not None else self.cells_max_threshold
            self.cell_fill_threshold = cell_fill if cell_fill is not None else self.cell_fill_threshold
            
            # Extract cells using watershed
            labels = self.cells_from_dist_map(dist_map, self.cells_max_threshold, self.cell_fill_threshold)
            
            # Log results
            num_cells = len(np.unique(labels)) - 1
            logger.info(f"Segmentation complete. Found {num_cells} cells.")
            
            if return_dist_map:
                return labels, dist_map
            else:
                return labels
                
        except Exception as e:
            logger.error(f"Error in run: {str(e)}")
            logger.error(traceback.format_exc())
            if return_dist_map:
                return np.zeros(image.shape[:2], dtype=np.int32), np.zeros(image.shape[:2], dtype=np.float32)
            else:
                return np.zeros(image.shape[:2], dtype=np.int32)
    
    def run_batch_thresholds(self, image, cells_max_values, cell_fill_values):
        """
        Run inference with multiple threshold combinations for parameter tuning.
        
        Parameters
        ----------
        image : numpy.ndarray
            Input image (H, W)
        cells_max_values : List[float]
            List of cells_max threshold values to try
        cell_fill_values : List[float]
            List of cell_fill threshold values to try
        
        Returns
        -------
        Dict[Tuple[float, float], numpy.ndarray]
            Dictionary mapping (cells_max, cell_fill) tuples to segmentation labels
        """
        try:
            image = image.copy()
            
            # Get distance map once
            logger.info(f"Running batch threshold analysis with {len(cells_max_values)} x {len(cell_fill_values)} combinations")
            dist_map = self.predict_on_full_img(image)
            
            # Process all threshold combinations
            results = {}
            
            for cells_max in cells_max_values:
                # Compute cell centers for this cells_max value
                cells_max_mask = (dist_map > cells_max).astype(np.uint8)
                
                try:
                    contours, _ = cv2.findContours(cells_max_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                except:
                    _, contours, _ = cv2.findContours(cells_max_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Create markers
                base_markers = np.zeros(dist_map.shape, dtype=np.int32)
                for i, contour in enumerate(contours):
                    if contour is None or len(contour) < 3:
                        continue
                    
                    try:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                        else:
                            cX = int(np.mean([p[0][0] for p in contour]))
                            cY = int(np.mean([p[0][1] for p in contour]))
                        
                        cY = min(max(0, cY), base_markers.shape[0] - 1)
                        cX = min(max(0, cX), base_markers.shape[1] - 1)
                        base_markers[cY, cX] = i + 1
                    except:
                        continue
                
                # For each cell_fill value, apply watershed
                for cell_fill in cell_fill_values:
                    cell_fill_mask = (dist_map > cell_fill).astype(np.uint8)
                    
                    if np.max(base_markers) == 0:
                        labels = np.zeros(dist_map.shape, dtype=np.int32)
                    else:
                        labels = watershed(-dist_map, base_markers, mask=cell_fill_mask).astype(np.int32)
                    
                    results[(cells_max, cell_fill)] = labels
                    
                    num_cells = len(np.unique(labels)) - 1
                    logger.info(f"Combination cells_max={cells_max:.2f}, cell_fill={cell_fill:.2f}: {num_cells} cells")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in run_batch_thresholds: {str(e)}")
            logger.error(traceback.format_exc())
            return {}


# Alias for compatibility with SAMCell interface
SlidingWindowPipeline = DINOCellPipeline
