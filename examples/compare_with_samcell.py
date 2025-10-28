"""
Compare DINOCell with SAMCell
==============================

Side-by-side comparison of DINOCell and SAMCell on the same image.

This script runs both models and visualizes their predictions for comparison.
"""

import sys
from pathlib import Path

# Add both DINOCell and SAMCell to path
dinocell_path = Path(__file__).parent.parent / 'src'
samcell_path = Path(__file__).parent.parent.parent / 'SAMCell' / 'src'

sys.path.insert(0, str(dinocell_path))
sys.path.insert(0, str(samcell_path))

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from dinocell import create_dinocell_model, DINOCellPipeline
from samcell import FinetunedSAM, SlidingWindowPipeline as SAMCellPipeline


def main():
    # Configuration
    IMAGE_PATH = 'path/to/cell_image.png'
    DINOCELL_MODEL = '../checkpoints/dinocell/best_model.pt'
    SAMCELL_MODEL = '../checkpoints/samcell-generalist.pt'
    
    print("="*70)
    print("DINOCell vs SAMCell Comparison")
    print("="*70)
    
    # Load image
    print(f"\nLoading image from {IMAGE_PATH}...")
    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Error: Could not load image")
        return
    
    print(f"✓ Image shape: {image.shape}")
    
    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load DINOCell
    print("\n--- DINOCell ---")
    try:
        dinocell = create_dinocell_model(model_size='small', freeze_backbone=True, pretrained=False)
        dinocell.load_weights(DINOCELL_MODEL)
        dinocell_pipeline = DINOCellPipeline(dinocell, device=device)
        print("✓ DINOCell loaded")
        
        labels_dino, distmap_dino = dinocell_pipeline.run(image, return_dist_map=True)
        num_cells_dino = len(np.unique(labels_dino)) - 1
        print(f"✓ DINOCell found {num_cells_dino} cells")
    except Exception as e:
        print(f"✗ DINOCell error: {e}")
        labels_dino = None
    
    # Load SAMCell
    print("\n--- SAMCell ---")
    try:
        samcell = FinetunedSAM('facebook/sam-vit-base')
        samcell.load_weights(SAMCELL_MODEL)
        samcell_pipeline = SAMCellPipeline(samcell, device=device)
        print("✓ SAMCell loaded")
        
        labels_sam, distmap_sam = samcell_pipeline.run(image, return_dist_map=True)
        num_cells_sam = len(np.unique(labels_sam)) - 1
        print(f"✓ SAMCell found {num_cells_sam} cells")
    except Exception as e:
        print(f"✗ SAMCell error: {e}")
        labels_sam = None
    
    # Visualize comparison
    print("\nGenerating comparison visualization...")
    
    if labels_dino is not None and labels_sam is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Row 1: DINOCell
        axes[0, 0].imshow(image, cmap='gray')
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(distmap_dino, cmap='jet')
        axes[0, 1].set_title(f'DINOCell Distance Map')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(image, cmap='gray')
        labels_dino_colored = labels_dino.astype(float)
        labels_dino_colored[labels_dino_colored == 0] = np.nan
        axes[0, 2].imshow(labels_dino_colored, cmap='tab20', alpha=0.5)
        axes[0, 2].set_title(f'DINOCell: {num_cells_dino} cells')
        axes[0, 2].axis('off')
        
        # Row 2: SAMCell
        axes[1, 0].imshow(image, cmap='gray')
        axes[1, 0].set_title('Original Image')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(distmap_sam, cmap='jet')
        axes[1, 1].set_title(f'SAMCell Distance Map')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(image, cmap='gray')
        labels_sam_colored = labels_sam.astype(float)
        labels_sam_colored[labels_sam_colored == 0] = np.nan
        axes[1, 2].imshow(labels_sam_colored, cmap='tab20', alpha=0.5)
        axes[1, 2].set_title(f'SAMCell: {num_cells_sam} cells')
        axes[1, 2].axis('off')
        
        plt.suptitle('DINOCell vs SAMCell Comparison', fontsize=16, y=0.98)
        plt.tight_layout()
        
        output_path = 'comparison_result.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved comparison to {output_path}")
        
        plt.show()
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"DINOCell: {num_cells_dino} cells")
        print(f"SAMCell:  {num_cells_sam} cells")
        print(f"Difference: {abs(num_cells_dino - num_cells_sam)} cells")
        print("="*70)
    else:
        print("Cannot create comparison - one or both models failed to load")


if __name__ == '__main__':
    main()



