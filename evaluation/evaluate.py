"""
DINOCell Evaluation Script
===========================

Evaluate DINOCell models using Cell Tracking Challenge metrics (SEG, DET, OP_CSB).

Compatible with SAMCell's evaluation approach and metrics.

Usage:
    python evaluate.py --model ../checkpoints/dinocell/best_model.pt \\
                      --dataset ../datasets/PBL_HEK \\
                      --cells-max 0.47 --cell-fill 0.09
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
import argparse
import logging
from tqdm import tqdm
import pandas as pd
import os
import shutil
import uuid

from dinocell.model import create_dinocell_model
from dinocell.pipeline import DINOCellPipeline
from evaluation_utils import (
    run_evaluation_ctc,
    create_evaluation_structure,
    save_predictions_ctc_format
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_on_dataset(model, dataset_path, cells_max, cell_fill, binary_path=None, device='cuda'):
    """
    Evaluate model on a dataset using Cell Tracking Challenge metrics.
    
    Parameters
    ----------
    model : DINOCell
        Model to evaluate
    dataset_path : str or Path
        Path to dataset folder containing imgs.npy and anns.npy
    cells_max : float
        Cell peak threshold
    cell_fill : float
        Cell fill threshold
    binary_path : str, optional
        Path to CTC evaluation binaries
    device : str
        Device to use
    
    Returns
    -------
    dict
        Dictionary with 'seg', 'det', 'csb' metrics
    """
    dataset_path = Path(dataset_path)
    
    # Load data
    imgs_path = dataset_path / 'imgs.npy'
    anns_path = dataset_path / 'anns.npy'
    
    if not imgs_path.exists():
        logger.error(f'Images not found: {imgs_path}')
        return None
    
    if not anns_path.exists():
        logger.error(f'Annotations not found: {anns_path}')
        return None
    
    logger.info(f'Loading dataset from {dataset_path}')
    imgs = np.load(imgs_path)
    anns = np.load(anns_path)
    
    # Handle different image formats
    if len(imgs.shape) == 4:
        # Convert to grayscale if multi-channel
        if imgs.shape[3] > 1:
            imgs = imgs[:, :, :, 0]
        else:
            imgs = imgs[:, :, :, 0]
    
    logger.info(f'Loaded {len(imgs)} images')
    
    # Create pipeline
    pipeline = DINOCellPipeline(
        model=model,
        device=device,
        crop_size=256,
        cells_max=cells_max,
        cell_fill=cell_fill
    )
    
    # Generate predictions
    logger.info('Generating predictions...')
    predictions = []
    for img in tqdm(imgs):
        # Ensure grayscale
        if len(img.shape) == 3:
            img = img[:, :, 0]
        
        # Ensure uint8
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        
        label = pipeline.run(img)
        predictions.append(label)
    
    predictions = np.array(predictions)
    
    # Evaluate using CTC metrics if binary available
    if binary_path and Path(binary_path).exists():
        logger.info('Computing Cell Tracking Challenge metrics...')
        
        # Create temporary evaluation structure
        eval_uuid = uuid.uuid4().hex[:8]
        eval_base = f'eval_temp_{eval_uuid}'
        gt_seg_dir, gt_tra_dir, res_dir = create_evaluation_structure(eval_base)
        
        # Save in CTC format
        save_predictions_ctc_format(anns, predictions, gt_seg_dir, gt_tra_dir, res_dir)
        
        # Run evaluation
        seg_value = run_evaluation_ctc('SEGMeasure', eval_base, binary_path, '01', '04')
        det_value = run_evaluation_ctc('DETMeasure', eval_base, binary_path, '01', '04')
        csb_value = (seg_value + det_value) / 2 if seg_value and det_value else None
        
        # Cleanup
        try:
            if os.path.exists(eval_base):
                shutil.rmtree(eval_base)
        except:
            pass
        
        results = {
            'seg': seg_value,
            'det': det_value,
            'csb': csb_value
        }
    else:
        logger.warning('CTC binaries not found, skipping metric computation')
        results = {
            'seg': None,
            'det': None,
            'csb': None
        }
    
    # Add basic statistics
    num_imgs = len(predictions)
    total_cells_pred = sum(len(np.unique(pred)) - 1 for pred in predictions)
    total_cells_gt = sum(len(np.unique(ann)) - 1 for ann in anns)
    
    results['num_images'] = num_imgs
    results['total_cells_predicted'] = total_cells_pred
    results['total_cells_ground_truth'] = total_cells_gt
    results['avg_cells_per_image_pred'] = total_cells_pred / num_imgs if num_imgs > 0 else 0
    results['avg_cells_per_image_gt'] = total_cells_gt / num_imgs if num_imgs > 0 else 0
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate DINOCell model')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint (.pt file)')
    parser.add_argument('--model-size', choices=['small', 'base', 'large', '7b'], default='small',
                       help='DINOv3 model size (must match training)')
    
    # Dataset arguments
    parser.add_argument('--dataset', nargs='+', required=True,
                       help='Path(s) to dataset folder(s) for evaluation')
    
    # Threshold arguments
    parser.add_argument('--cells-max', type=float, default=0.47,
                       help='Cell peak threshold (default: 0.47)')
    parser.add_argument('--cell-fill', type=float, default=0.09,
                       help='Cell fill threshold (default: 0.09)')
    parser.add_argument('--threshold-search', action='store_true',
                       help='Search for optimal thresholds')
    parser.add_argument('--cells-max-range', nargs=2, type=float, default=[0.3, 0.7],
                       help='Range for cells_max search (default: 0.3 0.7)')
    parser.add_argument('--cell-fill-range', nargs=2, type=float, default=[0.05, 0.15],
                       help='Range for cell_fill search (default: 0.05 0.15)')
    
    # Other arguments
    parser.add_argument('--binary-path', type=str, default=None,
                       help='Path to Cell Tracking Challenge evaluation binaries')
    parser.add_argument('--output', type=str, default='evaluation_results.csv',
                       help='Output CSV file for results')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f'Using device: {device}')
    
    # Load model
    logger.info(f'Loading model from {args.model}...')
    model = create_dinocell_model(
        model_size=args.model_size,
        freeze_backbone=True,  # Always freeze for inference
        pretrained=False  # Will load from checkpoint
    )
    model.load_weights(args.model, map_location=torch.device(device))
    model.to(device)
    model.eval()
    logger.info('✓ Model loaded successfully')
    
    # Evaluate on each dataset
    all_results = []
    
    for dataset_path in args.dataset:
        dataset_name = Path(dataset_path).name
        logger.info(f'\n{"="*70}')
        logger.info(f'Evaluating on {dataset_name}')
        logger.info(f'{"="*70}')
        
        if args.threshold_search:
            logger.info('Performing threshold grid search...')
            
            # Generate threshold ranges
            cells_max_values = np.arange(args.cells_max_range[0], args.cells_max_range[1], 0.05)
            cell_fill_values = np.arange(args.cell_fill_range[0], args.cell_fill_range[1], 0.01)
            
            best_csb = -1
            best_params = None
            
            for cells_max in cells_max_values:
                for cell_fill in cell_fill_values:
                    results = evaluate_on_dataset(
                        model, dataset_path, cells_max, cell_fill, args.binary_path, device
                    )
                    
                    if results and results['csb'] is not None:
                        if results['csb'] > best_csb:
                            best_csb = results['csb']
                            best_params = (cells_max, cell_fill)
                        
                        results['dataset'] = dataset_name
                        results['cells_max'] = cells_max
                        results['cell_fill'] = cell_fill
                        all_results.append(results)
            
            logger.info(f'\n✓ Best parameters: cells_max={best_params[0]:.2f}, cell_fill={best_params[1]:.2f}')
            logger.info(f'  CSB: {best_csb:.4f}')
        
        else:
            # Single evaluation with specified thresholds
            results = evaluate_on_dataset(
                model, dataset_path, args.cells_max, args.cell_fill, args.binary_path, device
            )
            
            if results:
                results['dataset'] = dataset_name
                results['cells_max'] = args.cells_max
                results['cell_fill'] = args.cell_fill
                all_results.append(results)
                
                logger.info(f'\nResults:')
                if results['seg'] is not None:
                    logger.info(f'  SEG: {results["seg"]:.4f}')
                    logger.info(f'  DET: {results["det"]:.4f}')
                    logger.info(f'  CSB: {results["csb"]:.4f}')
                logger.info(f'  Predicted cells: {results["total_cells_predicted"]}')
                logger.info(f'  Ground truth cells: {results["total_cells_ground_truth"]}')
    
    # Save results to CSV
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(args.output, index=False)
        logger.info(f'\n✓ Results saved to {args.output}')
        
        # Print summary
        logger.info(f'\n{"="*70}')
        logger.info('Summary')
        logger.info(f'{"="*70}')
        print(df.to_string(index=False))
    else:
        logger.warning('No results to save')


if __name__ == '__main__':
    main()



