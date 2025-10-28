"""
DINOCell Command Line Interface
================================

Simple CLI for running DINOCell segmentation.

Usage:
    # Segment an image
    dinocell segment cells.png --model dinocell-generalist.pt --output results/
    
    # With custom thresholds
    dinocell segment cells.png --model dinocell-generalist.pt \\
                               --cells-max 0.5 --cell-fill 0.1 --output results/
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import logging
import sys

from .model import create_dinocell_model
from .pipeline import DINOCellPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def segment_image(args):
    """Run segmentation on an image."""
    # Load image
    logger.info(f'Loading image from {args.input}')
    image = cv2.imread(args.input, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        logger.error(f'Could not load image: {args.input}')
        return
    
    logger.info(f'Image shape: {image.shape}')
    
    # Load model
    logger.info(f'Loading model from {args.model}')
    model = create_dinocell_model(model_size=args.model_size, freeze_backbone=True, pretrained=False)
    model.load_weights(args.model)
    
    # Create pipeline
    pipeline = DINOCellPipeline(
        model=model,
        device=args.device,
        crop_size=256,
        cells_max=args.cells_max,
        cell_fill=args.cell_fill
    )
    
    # Run segmentation
    logger.info('Running segmentation...')
    labels, dist_map = pipeline.run(image, return_dist_map=True)
    
    num_cells = len(np.unique(labels)) - 1
    logger.info(f'✓ Segmentation complete! Found {num_cells} cells')
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save labels as 16-bit TIFF
    labels_path = output_dir / f'{Path(args.input).stem}_labels.tif'
    cv2.imwrite(str(labels_path), labels.astype(np.uint16))
    logger.info(f'✓ Saved labels to {labels_path}')
    
    # Save distance map
    if args.save_dist_map:
        dist_map_path = output_dir / f'{Path(args.input).stem}_distmap.png'
        dist_map_vis = (dist_map * 255).astype(np.uint8)
        cv2.imwrite(str(dist_map_path), dist_map_vis)
        logger.info(f'✓ Saved distance map to {dist_map_path}')
    
    # Save visualization
    if args.save_viz:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(dist_map, cmap='jet')
        axes[1].set_title('Distance Map')
        axes[1].axis('off')
        
        axes[2].imshow(image, cmap='gray')
        # Overlay labels
        labels_colored = labels.astype(float)
        labels_colored[labels_colored == 0] = np.nan
        axes[2].imshow(labels_colored, cmap='tab20', alpha=0.5)
        axes[2].set_title(f'Segmentation ({num_cells} cells)')
        axes[2].axis('off')
        
        plt.tight_layout()
        viz_path = output_dir / f'{Path(args.input).stem}_visualization.png'
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f'✓ Saved visualization to {viz_path}')


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description='DINOCell Command Line Interface')
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Segment command
    segment_parser = subparsers.add_parser('segment', help='Segment cells in an image')
    segment_parser.add_argument('input', help='Input image path')
    segment_parser.add_argument('--model', required=True, help='Path to model checkpoint')
    segment_parser.add_argument('--model-size', choices=['small', 'base', 'large'], default='small',
                               help='Model size (must match training)')
    segment_parser.add_argument('--output', default='results/', help='Output directory')
    segment_parser.add_argument('--cells-max', type=float, default=0.47, help='Cell peak threshold')
    segment_parser.add_argument('--cell-fill', type=float, default=0.09, help='Cell fill threshold')
    segment_parser.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    segment_parser.add_argument('--save-dist-map', action='store_true', help='Save distance map')
    segment_parser.add_argument('--save-viz', action='store_true', help='Save visualization')
    
    args = parser.parse_args()
    
    if args.command == 'segment':
        segment_image(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()



