#!/usr/bin/env python3
"""
Test attention visualization with pretrained DINOv3 on example cell images
"""

import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, '/home/shadeform/DINOCell/dinov3_modified/dinov3')

def load_cell_image(image_path, size=224):
    """Load and preprocess cell image for DINOv3"""
    # Load grayscale TIFF
    img = cv2.imread(str(image_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
    
    # Normalize to 0-255
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)
    
    # Convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Resize
    pil_img = Image.fromarray(img_rgb).resize((size, size), Image.BICUBIC)
    
    # To tensor
    img_array = np.array(pil_img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, pil_img

def extract_attention_with_hook(model, image_tensor):
    """Extract attention using forward hook (works with FSDP)"""
    attention_maps = []
    
    def hook_fn(module, input, output):
        """Capture attention weights"""
        # For vision transformer attention blocks
        # Output might be tuple (features, attention) or just features
        if isinstance(output, tuple) and len(output) > 1:
            attn = output[1]
            if isinstance(attn, torch.Tensor) and attn.ndim >= 3:
                attention_maps.append(attn.detach().cpu())
    
    # Find last attention block
    hooks = []
    if hasattr(model, 'blocks'):
        for block in model.blocks[-3:]:  # Try last 3 blocks
            if hasattr(block, 'attn'):
                hook = block.attn.register_forward_hook(hook_fn)
                hooks.append(hook)
    
    # Forward pass
    with torch.no_grad():
        try:
            _ = model.forward_features(image_tensor)
        except:
            _ = model(image_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return attention_maps

def visualize_attention_map(image_pil, attention, save_path):
    """Create attention visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image_pil)
    axes[0].set_title('Cell Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    if attention is not None and len(attention) > 0:
        # Get attention weights
        attn = attention[-1]  # Last captured attention
        
        # Handle different attention formats
        if attn.ndim == 4:  # [B, num_heads, seq_len, seq_len]
            attn = attn[0]  # First image
        if attn.ndim == 3:  # [num_heads, seq_len, seq_len]
            attn = attn.mean(0)  # Average across heads
        
        # Get CLS token attention to patches
        attn_map = attn[0, 1:].numpy()  # CLS attention to patches (skip CLS itself)
        
        # Reshape to spatial grid
        h = w = int(np.sqrt(len(attn_map)))
        attn_map = attn_map.reshape(h, w)
        
        # Attention heatmap
        axes[1].imshow(attn_map, cmap='jet', interpolation='bilinear')
        axes[1].set_title('Attention Heatmap', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Overlay on original
        attn_resized = cv2.resize(attn_map, (image_pil.size[0], image_pil.size[1]))
        attn_resized = (attn_resized - attn_resized.min()) / (attn_resized.max() - attn_resized.min() + 1e-8)
        
        # Create heatmap overlay
        heatmap = plt.cm.jet(attn_resized)[:, :, :3]
        img_array = np.array(image_pil) / 255.0
        overlay = 0.5 * img_array + 0.5 * heatmap
        overlay = np.clip(overlay, 0, 1)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Attention Overlay', fontsize=14, fontweight='bold')
        axes[2].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'No attention\ncaptured', ha='center', va='center', fontsize=16)
        axes[1].axis('off')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Saved: {save_path}")
    plt.close()

def main():
    print("="*70)
    print("🔬 DINOv3 Attention Visualization Test")
    print("="*70)
    
    # Load pretrained DINOv3
    print("\n1. Loading pretrained DINOv3 ViT-S/16...")
    from dinov3.hub.backbones import dinov3_vits16
    
    model = dinov3_vits16(pretrained=True)
    print("✅ Model loaded (auto-downloads from Meta)")
    
    model = model.cuda()
    model.eval()
    
    # Load example cell images
    print("\n2. Loading example cell images...")
    example_images = list(Path('/home/shadeform/example_images').rglob('*-ch1*.tiff'))[:4]
    
    if not example_images:
        print("❌ No example images found")
        return
    
    print(f"✅ Found {len(example_images)} cell images")
    
    # Process each image
    for idx, img_path in enumerate(example_images):
        print(f"\n3. Processing image {idx+1}: {img_path.parent.name}/{img_path.name}")
        
        # Load and preprocess
        image_tensor, image_pil = load_cell_image(img_path)
        image_tensor = image_tensor.cuda()
        
        # Extract attention
        print("   Extracting attention maps...")
        attention_maps = extract_attention_with_hook(model, image_tensor)
        
        if attention_maps:
            print(f"   ✅ Captured {len(attention_maps)} attention tensors")
            print(f"   Attention shape: {attention_maps[-1].shape}")
        else:
            print("   ⚠️  No attention captured (trying direct features)")
        
        # Create visualization
        save_path = f'attention_viz_{idx+1}_{img_path.parent.name}.png'
        visualize_attention_map(image_pil, attention_maps, save_path)
    
    print("\n" + "="*70)
    print("✅ Attention Visualization Test Complete!")
    print("="*70)
    print("Check the generated PNG files:")
    for idx, img_path in enumerate(example_images):
        print(f"  - attention_viz_{idx+1}_{img_path.parent.name}.png")
    print("\nIf these look good, wandb logging will work during training!")
    print("="*70)

if __name__ == '__main__':
    main()

