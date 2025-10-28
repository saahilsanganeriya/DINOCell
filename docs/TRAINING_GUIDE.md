# DINOCell Training Guide

This guide walks you through training DINOCell on cell segmentation datasets.

## Prerequisites

1. **DINOv3 Repository**: Ensure DINOv3 is cloned in the parent directory
```bash
cd ..
git clone https://github.com/facebookresearch/dinov3.git
cd DINOCell
```

2. **Environment Setup**:
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

3. **Download Datasets**: See [Datasets](#datasets) section below

## Quick Start

### 1. Process Your Dataset

```bash
cd dataset_processing

# Process LIVECell (recommended)
python process_dataset.py livecell \
    --input /path/to/LIVECell_dataset_2021 \
    --output ../datasets/LIVECell-train \
    --split train

# Process Cellpose
python process_dataset.py cellpose \
    --input /path/to/cellpose/train \
    --output ../datasets/Cellpose-train \
    --target-size 512
```

### 2. Train DINOCell

```bash
cd ../training

# Option A: Freeze backbone (fastest, recommended for getting started)
python train.py \
    --dataset ../datasets/LIVECell-train \
    --model-size small \
    --freeze-backbone \
    --batch-size 8 \
    --epochs 100 \
    --output ../checkpoints/dinocell-frozen

# Option B: Fine-tune backbone (slower, potentially better performance)
python train.py \
    --dataset ../datasets/LIVECell-train \
    --model-size small \
    --batch-size 4 \
    --epochs 100 \
    --output ../checkpoints/dinocell-finetuned

# Option C: Multi-dataset generalist model
python train.py \
    --dataset ../datasets/LIVECell-train ../datasets/Cellpose-train \
    --model-size base \
    --freeze-backbone \
    --batch-size 4 \
    --epochs 100 \
    --output ../checkpoints/dinocell-generalist
```

### 3. Evaluate Your Model

```bash
cd ../evaluation

# Evaluate on test dataset
python evaluate.py \
    --model ../checkpoints/dinocell-frozen/best_model.pt \
    --model-size small \
    --dataset ../datasets/PBL_HEK ../datasets/PBL_N2A \
    --cells-max 0.47 \
    --cell-fill 0.09

# Threshold grid search to find optimal parameters
python evaluate.py \
    --model ../checkpoints/dinocell-frozen/best_model.pt \
    --model-size small \
    --dataset ../datasets/PBL_HEK \
    --threshold-search \
    --cells-max-range 0.3 0.7 \
    --cell-fill-range 0.05 0.15
```

## Datasets

### Required Datasets

**LIVECell** (Primary training dataset)
- Download: https://sartorius-research.github.io/LIVECell/
- Size: ~5GB
- Contains: 5000+ phase-contrast images, 8 cell types

**Cellpose Cytoplasm** (Secondary training dataset for diversity)
- Download from: https://www.cellpose.org/dataset
- Size: ~500MB
- Contains: ~600 diverse microscopy images

### Evaluation Datasets

**PBL-HEK and PBL-N2a** (For zero-shot testing)
- Download: https://github.com/saahilsanganeriya/SAMCell/releases/tag/v1
- Size: ~50MB each
- Contains: 5 images each of novel cell types

## Training Strategies

### Strategy 1: Frozen Backbone (Recommended for Getting Started)

**When to use**: 
- Limited GPU memory
- Quick experiments
- Transfer learning from pretrained DINOv3

**Advantages**:
- Faster training (2-3x speedup)
- Lower memory requirements
- Good performance leveraging pretrained features

**Command**:
```bash
python train.py --dataset ../datasets/LIVECell-train \
               --model-size small --freeze-backbone \
               --batch-size 8 --epochs 100
```

**Expected Training Time**: ~2-3 hours on A100 GPU

### Strategy 2: End-to-End Fine-Tuning

**When to use**:
- Maximum performance needed
- Sufficient GPU memory (≥24GB VRAM)
- Domain-specific fine-tuning

**Advantages**:
- Best performance
- Adapts features specifically for cells

**Command**:
```bash
python train.py --dataset ../datasets/LIVECell-train \
               --model-size small \
               --batch-size 4 --epochs 100
```

**Expected Training Time**: ~5-6 hours on A100 GPU

### Strategy 3: Multi-Dataset Generalist

**When to use**:
- Building a generalist model
- Maximizing zero-shot performance
- Multiple dataset sources available

**Advantages**:
- Best cross-dataset generalization
- Robust to dataset variations

**Command**:
```bash
python train.py --dataset ../datasets/LIVECell-train ../datasets/Cellpose-train \
               --model-size base --freeze-backbone \
               --batch-size 4 --epochs 100
```

**Expected Training Time**: ~4-5 hours on A100 GPU

## Model Sizes

| Size | Backbone | Parameters | VRAM (frozen) | VRAM (fine-tune) | Speed |
|------|----------|------------|---------------|------------------|-------|
| Small | ViT-S/16 | 21M | ~6GB | ~12GB | Fast |
| Base | ViT-B/16 | 86M | ~10GB | ~20GB | Medium |
| Large | ViT-L/16 | 300M | ~20GB | ~40GB | Slow |
| 7B | ViT-7B/16 | 6.7B | ~80GB | N/A | Very Slow |

**Recommendations**:
- **For most users**: `small` with frozen backbone
- **For best performance**: `base` with frozen backbone
- **For research**: `large` or fine-tuned models

## Hyperparameters

### Default Settings (Optimized from SAMCell)

```python
# Optimizer
lr = 1e-4
weight_decay = 0.1
betas = (0.9, 0.999)

# Training
batch_size = 8 (frozen) or 4 (fine-tuned)
epochs = 100 (with early stopping)
patience = 7
min_delta = 0.0001

# Data
val_split = 0.1
crop_size = 256
augmentation = True

# Post-processing
cells_max_threshold = 0.47
cell_fill_threshold = 0.09
```

### Tuning Hyperparameters

**Learning Rate**:
- Frozen backbone: 1e-4 to 1e-3
- Fine-tuned backbone: 1e-5 to 1e-4

**Batch Size**:
- Adjust based on GPU memory
- Larger batches can improve training stability

**Thresholds**:
- Run grid search on validation set
- Dataset-specific tuning may improve performance

## Advanced: Self-Supervised Pretraining

For users with large collections (>10k images) of unlabeled cell images:

### Step 1: Prepare Unlabeled Images

```bash
# Collect unlabeled cell images into a folder
/path/to/unlabeled_cells/
    image_001.png
    image_002.png
    ...
    image_10000.png
```

### Step 2: Pretrain Using DINOv3

Follow instructions in `training/pretrain_dinov3.py` for full details.

This involves:
1. Setting up DINOv3 training environment
2. Creating a custom config for cell images
3. Running multi-GPU pretraining
4. Using pretrained weights for DINOCell fine-tuning

**Note**: This is computationally expensive and requires multiple GPUs.
For most users, using the pretrained DINOv3 weights works excellently.

## Monitoring Training

### Using Weights & Biases (Optional)

```bash
# Install wandb
pip install wandb

# Login
wandb login

# Train with logging
python train.py --dataset ../datasets/LIVECell-train \
               --model-size small --freeze-backbone \
               --use-wandb --wandb-project dinocell
```

### Checkpointing

Training automatically saves:
- `best_model.pt`: Best model by validation loss
- `final_model.pt`: Model after last epoch
- `checkpoint_epoch_N.pt`: Periodic checkpoints
- `config.txt`: Training configuration

## Troubleshooting

### Out of Memory (OOM)

**Solutions**:
1. Reduce batch size: `--batch-size 2`
2. Use smaller model: `--model-size small`
3. Freeze backbone: `--freeze-backbone`
4. Enable gradient checkpointing (advanced)

### Poor Performance

**Solutions**:
1. Check data preprocessing
2. Try different model sizes
3. Fine-tune hyperparameters
4. Add more training data
5. Tune post-processing thresholds

### Slow Training

**Solutions**:
1. Use frozen backbone: `--freeze-backbone`
2. Increase batch size if memory allows
3. Reduce number of workers if I/O bound
4. Use smaller model size

## Next Steps

After training:

1. **Evaluate** on test sets and cross-dataset benchmarks
2. **Tune** post-processing thresholds for your specific use case
3. **Compare** with SAMCell and other baselines
4. **Deploy** in your microscopy workflow

See `README.md` for usage examples and API documentation.



