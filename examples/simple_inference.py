"""
DINOCell Simple Inference Example
==================================

Minimal example showing how to segment cells with DINOCell.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import cv2
import numpy as np
import matplotlib.pyplot as plt
from dinocell import create_dinocell_model, DINOCellPipeline


def main():
    # Configuration
    IMAGE_PATH = 'path/to/your/cell_image.png'  # Update this!
    MODEL_PATH = '../checkpoints/dinocell/best_model.pt'  # Update this!
    MODEL_SIZE = 'small'
    DEVICE = 'cuda'  # or 'cpu'
    
    print("="*70)
    print("DINOCell Simple Inference Example")
    print("="*70)
    
    # Load image
    print(f"\n1. Loading image from {IMAGE_PATH}...")
    image = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"✗ Error: Could not load image from {IMAGE_PATH}")
        print("Please update IMAGE_PATH in the script to point to a valid microscopy image")
        return
    
    print(f"✓ Image loaded successfully. Shape: {image.shape}")
    
    # Create model
    print(f"\n2. Loading DINOCell model ({MODEL_SIZE})...")
    model = create_dinocell_model(
        model_size=MODEL_SIZE,
        freeze_backbone=True,  # For inference, always freeze
        pretrained=False  # Will load from checkpoint
    )
    
    # Load weights
    try:
        model.load_weights(MODEL_PATH)
        print(f"✓ Model loaded from {MODEL_PATH}")
    except:
        print(f"✗ Could not load weights from {MODEL_PATH}")
        print("Continuing with randomly initialized decoder (for demo purposes)")
    
    # Create pipeline
    print("\n3. Creating inference pipeline...")
    pipeline = DINOCellPipeline(
        model=model,
        device=DEVICE,
        crop_size=256,
        cells_max=0.47,  # Default threshold
        cell_fill=0.09   # Default threshold
    )
    print("✓ Pipeline ready")
    
    # Run segmentation
    print("\n4. Running segmentation...")
    labels, dist_map = pipeline.run(image, return_dist_map=True)
    
    num_cells = len(np.unique(labels)) - 1  # -1 for background
    print(f"✓ Segmentation complete!")
    print(f"  Found {num_cells} cells")
    
    # Visualize results
    print("\n5. Visualizing results...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Distance map
    im1 = axes[1].imshow(dist_map, cmap='jet')
    axes[1].set_title('Predicted Distance Map')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)
    
    # Segmentation
    axes[2].imshow(image, cmap='gray')
    labels_colored = labels.astype(float)
    labels_colored[labels_colored == 0] = np.nan
    axes[2].imshow(labels_colored, cmap='tab20', alpha=0.5)
    axes[2].set_title(f'Segmentation ({num_cells} cells)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    output_path = 'dinocell_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to {output_path}")
    
    plt.show()
    
    # Save labels
    labels_path = 'dinocell_labels.tif'
    cv2.imwrite(labels_path, labels.astype(np.uint16))
    print(f"✓ Saved labels to {labels_path}")
    
    print("\n" + "="*70)
    print("✓ Inference complete!")
    print("="*70)


if __name__ == '__main__':
    main()



