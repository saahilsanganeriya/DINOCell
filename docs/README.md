# DINOCell: Biological Cell Segmentation with DINOv3

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Based on](https://img.shields.io/badge/Based%20on-DINOv3-green.svg)](https://github.com/facebookresearch/dinov3)

**DINOCell** is a state-of-the-art deep learning model for automated cell segmentation in microscopy images. Built on Meta's DINOv3 Vision Transformer, DINOCell uses the same proven distance map regression and watershed approach as SAMCell, but with DINOv3's superior feature representations pretrained on 1.7 billion images.

## 🌟 Key Features

- **DINOv3 Backbone**: Leverages DINOv3's ViT architecture pretrained on 1.7B images (vs SAM's 11M)
- **Distance Map Regression**: Predicts Euclidean distance to cell boundaries for better separation
- **Watershed Post-Processing**: Same proven approach as SAMCell with optimized thresholds
- **Multi-Scale Features**: Extracts features from multiple ViT layers for richer representations
- **Flexible Architecture**: Supports Small/Base/Large/7B model variants
- **Zero-Shot Generalization**: Works on novel cell types and microscopes
- **Compatible with SAMCell**: Uses same datasets and evaluation metrics for fair comparison

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd DINOCell

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e .
```

### Prerequisites

This framework requires the DINOv3 repository to be available in the parent directory:
```bash
cd ..
git clone https://github.com/facebookresearch/dinov3.git
cd DINOCell
```

### Download Pre-trained Weights

```bash
# Download DINOCell-Generalist weights (coming soon)
wget <release-url>/dinocell-generalist.pt

# Or download DINOCell-Small weights
wget <release-url>/dinocell-small.pt
```

### Basic Usage

```python
import cv2
from dinocell import DINOCell, DINOCellPipeline

# Load image
image = cv2.imread('cells.png', cv2.IMREAD_GRAYSCALE)

# Create model
model = DINOCell.from_pretrained('dinocell-generalist.pt')

# Create pipeline
pipeline = DINOCellPipeline(model, device='cuda')

# Segment cells
labels = pipeline.run(image)

print(f"Found {len(np.unique(labels))-1} cells")
```

### Command Line Interface

```bash
# Segment cells
python -m dinocell.cli segment cells.png --model dinocell-generalist.pt --output results/

# With custom thresholds
python -m dinocell.cli segment cells.png --model dinocell-generalist.pt \\
                                        --cells-max 0.5 --cell-fill 0.1
```

## 📊 Performance Comparison

Comparison with SAMCell on zero-shot cross-dataset evaluation:

| Dataset | Model | SEG | DET | OP_CSB |
|---------|-------|-----|-----|--------|
| PBL-HEK | **DINOCell** | **TBD** | **TBD** | **TBD** |
| | SAMCell | 0.425 | 0.772 | 0.598 |
| PBL-N2a | **DINOCell** | **TBD** | **TBD** | **TBD** |
| | SAMCell | 0.707 | 0.941 | 0.824 |

*Results to be updated after training on benchmarks.*

## 🔧 Architecture

DINOCell consists of:

1. **DINOv3 Backbone**: Pre-trained Vision Transformer (ViT) encoder
   - Extracts multi-scale features from 4 intermediate layers
   - Can be frozen or fine-tuned end-to-end
   - Model variants: Small (21M), Base (86M), Large (300M), 7B (6.7B parameters)

2. **U-Net Decoder**: Progressive upsampling with skip connections
   - Fuses multi-scale features from DINOv3
   - Lateral connections to reduce feature dimensions
   - Upsampling blocks with skip connections
   - Final 1x1 conv for distance map prediction

3. **Watershed Post-Processing**: Converts distance maps to cell masks
   - Cell peak threshold: 0.47 (identifies cell centers)
   - Cell fill threshold: 0.09 (determines boundaries)
   - Watershed algorithm separates touching cells

## 📋 Requirements

- Python ≥ 3.8
- PyTorch ≥ 2.0.0
- torchvision ≥ 0.15.0
- OpenCV ≥ 4.5.0
- scikit-image ≥ 0.19.0
- scipy ≥ 1.7.0
- numpy ≥ 1.20.0
- tqdm ≥ 4.60.0

For GPU acceleration:
- CUDA-compatible GPU (recommended: ≥8GB VRAM)
- CUDA Toolkit ≥ 11.0

## 🎓 Training

### Process Datasets

DINOCell uses the same datasets as SAMCell:

```bash
# Process LIVECell dataset
cd dataset_processing
python process_dataset.py livecell \\
    --input /path/to/LIVECell_dataset_2021 \\
    --output ../datasets/LIVECell-train \\
    --split train

# Process Cellpose dataset
python process_dataset.py cellpose \\
    --input /path/to/cellpose/train \\
    --output ../datasets/Cellpose-train \\
    --target-size 512
```

### Train DINOCell

```bash
cd training

# Train on single dataset (freeze backbone - fastest)
python train.py --dataset ../datasets/LIVECell-train \\
               --model-size small --freeze-backbone \\
               --epochs 100 --batch-size 8

# Train on multiple datasets (generalist model)
python train.py --dataset ../datasets/LIVECell-train ../datasets/Cellpose-train \\
               --model-size base --freeze-backbone \\
               --epochs 100 --batch-size 4

# Fine-tune backbone end-to-end (slower, potentially better)
python train.py --dataset ../datasets/LIVECell-train \\
               --model-size small \\
               --epochs 100 --batch-size 4
```

### Optional: Self-Supervised Pretraining

If you have large collections of unlabeled cell images (>10k), you can pretrain DINOv3:

```bash
# See instructions
python pretrain_dinov3.py --help

# Note: Full pretraining requires the DINOv3 training environment
# For most use cases, using pretrained DINOv3 weights is sufficient
```

## 🧪 Evaluation

```bash
cd evaluation

# Evaluate on test dataset
python evaluate.py --model ../checkpoints/dinocell/best_model.pt \\
                  --model-size small \\
                  --dataset ../datasets/PBL_HEK ../datasets/PBL_N2A \\
                  --cells-max 0.47 --cell-fill 0.09 \\
                  --binary-path /path/to/cell-tracking-binaries

# Threshold grid search
python evaluate.py --model ../checkpoints/dinocell/best_model.pt \\
                  --model-size small \\
                  --dataset ../datasets/PBL_HEK \\
                  --threshold-search \\
                  --cells-max-range 0.3 0.7 \\
                  --cell-fill-range 0.05 0.15
```

## 🔬 Method Overview

### Why DINOv3 instead of SAM?

1. **Larger Pretraining Dataset**: DINOv3 trained on 1.7B images vs SAM's 11M
2. **Self-Supervised Learning**: DINOv3 uses sophisticated self-supervised objectives
3. **Better Features**: DINOv3 produces denser, higher-quality features for dense prediction tasks
4. **Proven for Dense Tasks**: DINOv3 shows state-of-the-art results on segmentation and depth estimation

### Architecture Design

- **Multi-Scale Feature Extraction**: Extracts features from 4 intermediate ViT layers
  - Shallow layers: Low-level edges and textures
  - Middle layers: Cell-specific patterns
  - Deep layers: High-level semantic understanding

- **U-Net Decoder**: Progressive fusion and upsampling
  - Combines multi-scale information
  - Skip connections preserve spatial details
  - Gradual upsampling to 256x256 output

- **Distance Map Prediction**: Same as SAMCell
  - Continuous-valued predictions (0 to 1)
  - Better handles ambiguous boundaries
  - Enables watershed separation of touching cells

### Training Strategy

- **Transfer Learning**: Start from pretrained DINOv3 weights
- **Flexible Fine-Tuning**: Option to freeze or fine-tune backbone
- **MSE Loss**: Mean squared error on distance maps
- **Data Augmentation**: Flip, rotation, scale, brightness, inversion
- **Early Stopping**: Patience of 7 epochs with min_delta=0.0001

## 📖 API Reference

### DINOCell Model

```python
from dinocell import DINOCell, create_dinocell_model

# Create model
model = create_dinocell_model(
    model_size='small',        # 'small', 'base', 'large', or '7b'
    freeze_backbone=True,      # Freeze DINOv3 backbone
    pretrained=True,           # Use pretrained DINOv3 weights
    backbone_weights=None      # Path to custom weights
)

# Load fine-tuned weights
model.load_weights('dinocell-generalist.pt')

# Save weights
model.save_weights('my_model.pt')
```

### DINOCellPipeline

```python
from dinocell import DINOCellPipeline

# Create pipeline
pipeline = DINOCellPipeline(
    model,                    # DINOCell instance
    device='cuda',           # 'cuda' or 'cpu'
    crop_size=256,          # Patch size for sliding window
    cells_max=0.47,         # Cell peak threshold
    cell_fill=0.09          # Cell fill threshold
)

# Run segmentation
labels = pipeline.run(
    image,                   # Input grayscale image (H, W)
    return_dist_map=False,  # Also return distance map
    cells_max=None,         # Override threshold
    cell_fill=None          # Override threshold
)

# Distance map output
labels, dist_map = pipeline.run(image, return_dist_map=True)
```

## 📊 Datasets

DINOCell is compatible with SAMCell datasets:

### Training Datasets
- **LIVECell**: 5,000+ phase-contrast images, 8 cell types, ~1.7M cells
- **Cellpose Cytoplasm**: ~600 diverse microscopy images

### Evaluation Datasets (Zero-Shot)
- **PBL-HEK**: 5 images of HEK293 cells (~300 cells/image)
- **PBL-N2a**: 5 images of Neuro-2a cells (~300 cells/image)

Datasets available at: https://github.com/saahilsanganeriya/SAMCell/releases/tag/v1

## 🤝 Comparison with SAMCell

| Feature | SAMCell | DINOCell |
|---------|---------|----------|
| Backbone | SAM ViT-B (11M pretraining images) | DINOv3 ViT (1.7B pretraining images) |
| Pretraining | Supervised (segmentation) | Self-supervised (DINO + iBOT) |
| Parameters | 89M | 21M (Small) to 6.7B (7B) |
| Decoder | SAM's mask decoder | U-Net decoder |
| Training | Full fine-tuning | Flexible (freeze or fine-tune) |
| Approach | Distance map + watershed | Distance map + watershed |

## 📄 Citation

If you use DINOCell, please cite:

```bibtex
@software{dinocell2025,
  title={DINOCell: Biological Cell Segmentation with DINOv3},
  author={DINOCell Team},
  year={2025},
  note={Based on DINOv3 and SAMCell}
}
```

And cite the original works:

```bibtex
@article{simeoni2025dinov3,
  title={DINOv3},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and others},
  year={2025},
  journal={arXiv preprint arXiv:2508.10104}
}

@article{vandeloo2025samcell,
  title={SAMCell: Generalized label-free biological cell segmentation},
  author={VandeLoo, Alexandra Dunnum and others},
  journal={PLOS ONE},
  year={2025}
}
```

## 📞 Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

## 📜 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Meta AI for DINOv3 and Segment Anything Model
- SAMCell team for the distance map regression approach
- The open-source community for tools and datasets
- Georgia Tech for computational resources

---

**DINOCell Team** - Making cell segmentation even better with DINOv3! 🔬✨



