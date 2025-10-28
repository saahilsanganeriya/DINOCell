# DINOCell: Cell Segmentation with DINOv3

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DINOCell** is a state-of-the-art deep learning framework for automated cell segmentation in microscopy images. Built on Meta's DINOv3 Vision Transformer, DINOCell combines powerful self-supervised pretraining with a distance map regression approach for accurate cell instance segmentation.

## 🌟 Key Features

- **DINOv3 Backbone**: Leverages DINOv3's powerful vision transformer pretrained on 1.7B images
- **Distance Map Regression**: Predicts Euclidean distance maps for cell centers
- **Watershed Post-processing**: Robust cell instance segmentation
- **Multi-Channel SSL Pretraining**: Advanced multi-view consistency learning for microscopy
- **AWS S3 Streaming**: Train on massive datasets without local storage
- **Wandb Integration**: Comprehensive training monitoring and visualization

## 📁 Repository Structure

```
DINOCell/
├── dinocell/                      # Main DINOCell package
│   ├── model.py                  # DINOCell architecture (DINOv3 + U-Net)
│   ├── pipeline.py               # Inference pipeline with sliding window
│   ├── preprocessing.py          # Image preprocessing utilities
│   ├── dataset.py                # PyTorch dataset classes
│   ├── slidingWindow.py          # Sliding window helper
│   └── cli.py                    # Command-line interface
│
├── dinov3_modified/              # Modified DINOv3 with JUMP dataset support
│   └── dinov3/
│       ├── data/datasets/
│       │   ├── jump_cellpainting.py        # JUMP dataset loader
│       │   ├── jump_cellpainting_multiview.py  # Multi-view learning
│       │   └── jump_cellpainting_s3.py     # S3 streaming
│       └── logging/
│           └── wandb_logger.py              # Wandb integration
│
├── training/                      # Training scripts
│   ├── ssl_pretraining/          # Self-supervised pretraining
│   │   ├── configs/             # Pretraining configs
│   │   ├── launch_ssl_with_s3_wandb.sh  # Main launch script
│   │   └── validate_channel_consistency.py
│   └── finetune/                # DINOCell fine-tuning
│       ├── train.py             # Training script
│       └── train_with_config.py
│
├── evaluation/                    # Evaluation scripts
│   ├── evaluate.py               # Cell Tracking Challenge metrics
│   └── evaluation_utils.py
│
├── dataset_processing/            # Dataset processing utilities
│   ├── process_dataset.py        # Convert datasets to DINOCell format
│   └── dataset_utils.py
│
├── examples/                      # Example scripts
│   ├── simple_inference.py
│   └── compare_with_samcell.py
│
└── docs/                          # Documentation
    ├── GETTING_STARTED.md
    ├── SSL_PRETRAINING.md        # SSL pretraining guide
    ├── S3_STREAMING.md           # AWS S3 streaming guide
    └── ARCHITECTURE.md

```

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <your-repo-url>
cd DINOCell

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Basic Usage

```python
import cv2
from dinocell import create_dinocell_model, DINOCellPipeline

# Load image
image = cv2.imread('cells.png', cv2.IMREAD_GRAYSCALE)

# Create model and pipeline
model = create_dinocell_model('small', freeze_backbone=True, pretrained=True)
pipeline = DINOCellPipeline(model, device='cuda')

# Segment cells
labels = pipeline.run(image)
print(f"Found {len(np.unique(labels))-1} cells!")
```

## 📚 Documentation

- **[Getting Started](docs/GETTING_STARTED.md)** - Complete installation and setup guide
- **[SSL Pretraining](docs/SSL_PRETRAINING.md)** - Self-supervised pretraining on JUMP dataset
- **[S3 Streaming](docs/S3_STREAMING.md)** - Train without downloading massive datasets
- **[Architecture](docs/ARCHITECTURE.md)** - Technical details of DINOCell
- **[Training Guide](docs/TRAINING_GUIDE.md)** - Fine-tuning DINOCell on cell datasets

## 🔬 Self-Supervised Pretraining

DINOCell supports advanced SSL pretraining with multi-view consistency learning:

```bash
cd training/ssl_pretraining
./launch_ssl_with_s3_wandb.sh
```

This will:
- ✅ Stream 3M JUMP Cell Painting images from AWS S3 (no local storage needed!)
- ✅ Learn channel-invariant features across 5 fluorescent channels
- ✅ Log training metrics and attention maps to Wandb
- ✅ Train ViT-Small with patch size 8 for high resolution

See [`docs/SSL_PRETRAINING.md`](docs/SSL_PRETRAINING.md) for details.

## 🎯 Training DINOCell

```bash
cd training/finetune
python train.py \
    --dataset ../../datasets/LIVECell-train \
    --model-size small \
    --epochs 100 \
    --freeze-backbone
```

## 📊 Evaluation

```bash
cd evaluation
python evaluate.py \
    --model ../checkpoints/dinocell_best.pt \
    --dataset ../datasets/PBL_HEK \
    --cells-max 0.47 \
    --cell-fill 0.09
```

## 🔑 Key Innovations

### 1. Multi-View SSL Pretraining
Instead of simply averaging fluorescent channels, we use multi-view consistency learning:
- Global view 1: Average of all 5 channels
- Global view 2: Random single channel
- DINO loss enforces: same features regardless of channel
- **Result**: Channel-invariant cell representations!

### 2. AWS S3 Streaming
Train on 3 million images without downloading:
- LRU caching (1000 images in memory)
- Public bucket (no AWS credentials needed)
- Saves ~500GB local storage

### 3. Comprehensive Wandb Logging
- Training metrics (loss, learning rate, etc.)
- Attention map visualizations
- Feature PCA plots
- Gradient statistics
- Channel consistency validation

## 🗂️ Modified DINOv3

We've extended the official DINOv3 repository with:
- **JUMP Cell Painting Dataset Loaders** (3 variants: local, multiview, S3 streaming)
- **Multi-Channel Augmentation** for microscopy images
- **Wandb Logger** for comprehensive training monitoring

All modifications are in `dinov3_modified/dinov3/`.

## 📦 Dependencies

Core:
- PyTorch ≥ 2.0
- DINOv3 (included as `dinov3_modified/`)
- OpenCV, scikit-image, scipy

Optional:
- boto3, smart-open (for S3 streaming)
- wandb (for training monitoring)
- pycocotools (for LIVECell dataset)

## 🏗️ Architecture

```
Input Image → CLAHE → Sliding Window (256×256) →
    DINOv3 Backbone (frozen/trainable) →
    U-Net Decoder (4 upsampling stages) →
    Distance Map Head →
    Watershed Post-processing → Cell Masks
```

## 📖 Citation

If you use DINOCell, please cite:

```bibtex
@software{dinocell2025,
  title={DINOCell: Cell Segmentation with DINOv3},
  author={},
  year={2025},
  url={}
}
```

And cite the original DINOv3:
```bibtex
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Siméoni, Oriane and Vo, Huy V. and others},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

DINOv3 components under DINOv3 License - see [dinov3_modified/LICENSE.md](dinov3_modified/LICENSE.md).

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.
