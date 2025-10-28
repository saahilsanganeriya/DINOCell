"""
Evaluation Utilities for DINOCell
==================================

Utilities for evaluating cell segmentation using Cell Tracking Challenge metrics.
Adapted from SAMCell's evaluation utilities.
"""

import os
import subprocess
import numpy as np
from pathlib import Path
import logging
import cv2

logger = logging.getLogger(__name__)


def create_evaluation_structure(base_name):
    """
    Create directory structure for Cell Tracking Challenge evaluation.
    
    Parameters
    ----------
    base_name : str
        Base name for evaluation directories
    
    Returns
    -------
    tuple
        (gt_seg_dir, gt_tra_dir, res_dir) paths
    """
    base_path = Path(base_name)
    
    # Create directories
    gt_seg_dir = base_path / '01_GT' / 'SEG'
    gt_tra_dir = base_path / '01_GT' / 'TRA'
    res_dir = base_path / '01_RES'
    
    gt_seg_dir.mkdir(parents=True, exist_ok=True)
    gt_tra_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    
    return str(gt_seg_dir), str(gt_tra_dir), str(res_dir)


def save_predictions_ctc_format(ground_truth, predictions, gt_seg_dir, gt_tra_dir, res_dir):
    """
    Save predictions in Cell Tracking Challenge format.
    
    Parameters
    ----------
    ground_truth : numpy.ndarray
        Ground truth masks (N, H, W)
    predictions : numpy.ndarray
        Predicted masks (N, H, W)
    gt_seg_dir : str
        Ground truth SEG directory
    gt_tra_dir : str
        Ground truth TRA directory
    res_dir : str
        Results directory
    """
    # Save ground truth
    for i, gt_mask in enumerate(ground_truth):
        # SEG format
        seg_path = Path(gt_seg_dir) / f'man_seg{i:04d}.tif'
        cv2.imwrite(str(seg_path), gt_mask.astype(np.uint16))
        
        # TRA format (same as SEG for instance segmentation)
        tra_path = Path(gt_tra_dir) / f'man_track{i:04d}.tif'
        cv2.imwrite(str(tra_path), gt_mask.astype(np.uint16))
    
    # Save predictions
    for i, pred_mask in enumerate(predictions):
        res_path = Path(res_dir) / f'mask{i:04d}.tif'
        cv2.imwrite(str(res_path), pred_mask.astype(np.uint16))


def run_evaluation_ctc(metric_name, eval_base, binary_path, sequence='01', num_digits='04'):
    """
    Run Cell Tracking Challenge evaluation binary.
    
    Parameters
    ----------
    metric_name : str
        Metric to compute ('SEGMeasure' or 'DETMeasure')
    eval_base : str
        Base path for evaluation structure
    binary_path : str
        Path to CTC evaluation binaries
    sequence : str
        Sequence number (default: '01')
    num_digits : str
        Number of digits for filenames (default: '04')
    
    Returns
    -------
    float or None
        Metric value, or None if evaluation failed
    """
    try:
        if binary_path is None:
            logger.warning('Binary path not provided, skipping CTC evaluation')
            return None
        
        binary_path = Path(binary_path)
        if not binary_path.exists():
            logger.warning(f'Binary path not found: {binary_path}')
            return None
        
        # Find binary
        binary_file = binary_path / metric_name
        if not binary_file.exists():
            # Try with .exe extension (Windows)
            binary_file = binary_path / f'{metric_name}.exe'
            if not binary_file.exists():
                logger.warning(f'Binary not found: {binary_file}')
                return None
        
        # Run evaluation
        cmd = [str(binary_file), str(eval_base), sequence, num_digits]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Parse output
        output = result.stdout
        for line in output.split('\n'):
            if metric_name in line and ':' in line:
                try:
                    value_str = line.split(':')[-1].strip()
                    return float(value_str)
                except:
                    pass
        
        logger.warning(f'Could not parse {metric_name} output')
        return None
        
    except Exception as e:
        logger.error(f'Error running {metric_name}: {e}')
        return None


def threshold_grid_search(model, dataset_path, cells_max_range, cell_fill_range, 
                         binary_path, device='cuda'):
    """
    Perform grid search over threshold parameters.
    
    Parameters
    ----------
    model : DINOCell
        Model to evaluate
    dataset_path : str
        Path to dataset
    cells_max_range : tuple
        (min, max, step) for cells_max values
    cell_fill_range : tuple
        (min, max, step) for cell_fill values
    binary_path : str
        Path to CTC binaries
    device : str
        Device to use
    
    Returns
    -------
    pd.DataFrame
        DataFrame with results for each threshold combination
    """
    import pandas as pd
    
    cells_max_values = np.arange(*cells_max_range)
    cell_fill_values = np.arange(*cell_fill_range)
    
    results = []
    
    for cells_max in cells_max_values:
        for cell_fill in cell_fill_values:
            logger.info(f'Testing cells_max={cells_max:.2f}, cell_fill={cell_fill:.2f}')
            
            metrics = evaluate_on_dataset(
                model, dataset_path, cells_max, cell_fill, binary_path, device
            )
            
            if metrics:
                metrics['cells_max'] = cells_max
                metrics['cell_fill'] = cell_fill
                results.append(metrics)
    
    return pd.DataFrame(results)


if __name__ == '__main__':
    main()



