# DINOCell Architecture

Detailed technical documentation of DINOCell's architecture and design choices.

## Overview

DINOCell combines:
1. **DINOv3 Backbone**: Pretrained Vision Transformer for feature extraction
2. **U-Net Decoder**: Multi-scale feature fusion for distance map prediction
3. **Watershed Post-Processing**: Converting distance maps to cell masks

## Architecture Diagram

```
Input Image (H×W, grayscale)
         ↓
    CLAHE Enhancement
         ↓
Sliding Window (256×256 patches, 32px overlap)
         ↓
    ┌─────────────────────────┐
    │   DINOCell Model        │
    │                         │
    │   DINOv3 Backbone       │
    │   ├─ Block 2  →  F1     │  (Shallow features)
    │   ├─ Block 5  →  F2     │  (Low-level features)
    │   ├─ Block 8  →  F3     │  (Mid-level features)
    │   └─ Block 11 →  F4     │  (Deep features)
    │                         │
    │   U-Net Decoder         │
    │   ├─ Lateral Conv       │
    │   ├─ Up + Fusion        │
    │   ├─ Up + Fusion        │
    │   ├─ Up + Fusion        │
    │   └─ Final Conv         │
    │                         │
    │   Output: Distance Map  │
    └─────────────────────────┘
         ↓
Distance Map (256×256, [0,1])
         ↓
Blend Overlapping Patches (cosine falloff)
         ↓
Full Distance Map (H×W, [0,1])
         ↓
Watershed Post-Processing
├─ Threshold 1 (0.47) → Cell Centers
├─ Threshold 2 (0.09) → Cell Boundaries
└─ Watershed Algorithm → Cell Masks
         ↓
Cell Labels (H×W, int)
```

## 1. DINOv3 Backbone

### Why DINOv3?

**Advantages over SAM**:
- Larger pretraining dataset (1.7B images vs 11M)
- Self-supervised learning objectives (DINO, iBOT, KoLeo)
- Better dense features for segmentation tasks
- Proven state-of-the-art on depth/segmentation benchmarks
- More efficient architecture for dense prediction

### Feature Extraction

DINOCell extracts features from 4 intermediate ViT layers:

**ViT-Small/Base (12 blocks)**:
- Layer 2: Shallow features (edges, textures)
- Layer 5: Low-level features (cell patterns)
- Layer 8: Mid-level features (cell structures)
- Layer 11: Deep features (semantic understanding)

**ViT-Large (24 blocks)**:
- Layers: [4, 11, 17, 23]

**ViT-7B (40 blocks)**:
- Layers: [9, 19, 29, 39]

### Feature Dimensions

| Model | Embedding Dim | Feature Maps | Total Features |
|-------|---------------|--------------|----------------|
| ViT-S/16 | 384 | 16×16 per patch | 4×384 = 1536 |
| ViT-B/16 | 768 | 16×16 per patch | 4×768 = 3072 |
| ViT-L/16 | 1024 | 16×16 per patch | 4×1024 = 4096 |

## 2. U-Net Decoder

### Design Philosophy

The decoder progressively:
1. Reduces feature dimensions via 1×1 convolutions
2. Upsamples features via transposed convolutions
3. Fuses multi-scale information via skip connections
4. Produces final distance map via 1×1 conv

### Layer-by-Layer Breakdown

**Input**: 4 feature maps from DINOv3
- F1: (B, 384, 14, 14) - from layer 2
- F2: (B, 384, 14, 14) - from layer 5
- F3: (B, 384, 14, 14) - from layer 8
- F4: (B, 384, 14, 14) - from layer 11

*Note: All layers have same spatial dimensions (14×14 for 224×224 input)
but DINOCell processes 256×256, giving 16×16*

**Lateral Convolutions**:
- Reduce each feature map to decoder channels
- F1 → 256 channels
- F2 → 128 channels
- F3 → 64 channels
- F4 → 32 channels

**Upsampling Blocks**:
```
Level 1: (B, 256, 16, 16) → (B, 128, 32, 32) + F2
Level 2: (B, 128, 32, 32) → (B, 64, 64, 64) + F3
Level 3: (B, 64, 64, 64) → (B, 32, 128, 128) + F4
```

**Final Upsampling**:
```
(B, 32, 128, 128) → (B, 16, 256, 256) → (B, 8, 512, 512)
Then crop/resize to (B, 8, 256, 256)
```

**Distance Map Head**:
```
(B, 8, 256, 256) → (B, 1, 256, 256)  via 1×1 Conv
```

**Output**: (B, 1, 256, 256) raw distance map logits

### Skip Connections

U-Net style skip connections preserve spatial information:
- Connect corresponding encoder-decoder levels
- Concatenate before upsampling
- Help recover fine-grained boundaries

## 3. Sliding Window Inference

### Why Sliding Window?

1. **Memory Efficiency**: Process large images in small chunks
2. **Preserve Resolution**: Maintain pixel-level detail
3. **Match Training**: Same 256×256 patches as training
4. **DINOv3 Compatibility**: Works with DINOv3's native input sizes

### Parameters

- **Patch Size**: 256×256
- **Overlap**: 32 pixels on each side
- **Stride**: 192 pixels (256 - 2×32)
- **Upsampling**: Input patches upsampled to 224×224 for DINOv3

### Blending Strategy

Uses cosine falloff for smooth blending:
```python
weight = 0.5 * (1 - cos(π * distance_from_edge / overlap))
```

Advantages:
- No visible seams between patches
- Smooth transitions in overlapping regions
- Preserves prediction quality

## 4. Watershed Post-Processing

### Algorithm

1. **Threshold 1 (cells_max = 0.47)**:
   - Create binary mask of high-confidence cell regions
   - Find connected components
   - Extract centroids as markers

2. **Threshold 2 (cell_fill = 0.09)**:
   - Create binary mask of all cell pixels
   - Defines valid region for watershed

3. **Watershed Flooding**:
   - Treat distance map as topography
   - Flood from markers (cell centers)
   - Stop at boundaries (low distance values)
   - Separate cells where watersheds meet

### Threshold Selection

Optimized from ablation studies on SAMCell:
- **cells_max = 0.47**: Best average across datasets
- **cell_fill = 0.09**: Best average across datasets

Dataset-specific tuning can improve results:
- PBL-HEK: May benefit from higher cells_max (0.50-0.55)
- PBL-N2a: Works well with default values

## Design Decisions

### Why U-Net Decoder instead of SAM's Mask Decoder?

1. **Multi-Scale Fusion**: U-Net naturally combines multi-scale features
2. **Dense Prediction**: Designed specifically for pixel-wise tasks
3. **Simplicity**: Cleaner, more interpretable architecture
4. **Flexibility**: Easy to modify/extend for experiments

### Why Distance Maps?

1. **Continuous Values**: Better optimization than binary masks
2. **Separation**: Natural valleys between touching cells
3. **Compatibility**: Works perfectly with watershed algorithm
4. **Proven Approach**: Validated by SAMCell's success

### Why Freeze Backbone?

**Advantages**:
- Faster training (2-3× speedup)
- Lower memory requirements
- Leverages pretrained features effectively
- Good performance with limited data

**When to Fine-Tune**:
- Very large datasets (>10k images)
- Domain is very different from pretraining
- Maximum performance needed

## Computational Requirements

### Training (per epoch)

| Config | Model Size | Batch Size | VRAM | Time (A100) |
|--------|-----------|------------|------|-------------|
| Frozen | Small | 8 | ~6GB | ~2 min |
| Frozen | Base | 4 | ~10GB | ~4 min |
| Frozen | Large | 2 | ~20GB | ~10 min |
| Fine-tuned | Small | 4 | ~12GB | ~5 min |
| Fine-tuned | Base | 2 | ~24GB | ~10 min |

### Inference (per image)

| Model Size | Device | 512×512 Image | 2048×2048 Image |
|-----------|--------|---------------|------------------|
| Small | RTX 4090 | ~0.5s | ~3s |
| Small | CPU | ~5s | ~30s |
| Base | RTX 4090 | ~1s | ~6s |
| Large | A100 | ~2s | ~12s |

## Comparison with SAMCell

| Aspect | SAMCell | DINOCell |
|--------|---------|----------|
| **Backbone** | SAM ViT-B | DINOv3 ViT (S/B/L/7B) |
| **Pretraining** | 11M images (supervised) | 1.7B images (self-supervised) |
| **Decoder** | SAM mask decoder | U-Net decoder |
| **Parameters** | 89M | 21M-6.7B |
| **Flexibility** | Fixed architecture | Multiple sizes |
| **Training** | Full fine-tuning only | Freeze or fine-tune |
| **Features** | Single scale | Multi-scale (4 layers) |

## Future Improvements

Potential enhancements for DINOCell:

1. **Multi-Task Learning**: Joint prediction of distance maps + cell types
2. **Attention Mechanisms**: Spatial attention in decoder
3. **Instance-Aware Features**: Per-cell feature aggregation
4. **3D Extension**: Volumetric cell segmentation
5. **Temporal Tracking**: Video-based cell tracking
6. **Uncertainty Estimation**: Bayesian or ensemble approaches

## References

1. DINOv3: Siméoni et al., 2025
2. SAMCell: VandeLoo et al., PLOS ONE 2025
3. U-Net: Ronneberger et al., MICCAI 2015
4. Watershed: Beucher & Lantuéjoul, 1979



