# DINOCell Training Guide

This guide describes how DINOCell is meant to be fine-tuned on cell segmentation datasets.

> **Status:** at this commit the DINOCell decoder fails its first forward pass (see the
> [repository README](../README.md#status)), so `train.py` and `evaluate.py` stop with an error.
> The commands below are the intended workflow.

## Prerequisites

1. **DINOv3 code**: bundled in `dinov3_modified/dinov3/`; the model loads it from there, so no separate clone is needed.
   Pretrained DINOv3 weights need access from Meta (see the README).

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
cd ../training/finetune

# Option A: Freeze backbone (fastest, recommended for getting started)
python train.py \
    --dataset ../../datasets/LIVECell-train \
    --model-size small \
    --freeze-backbone \
    --batch-size 8 \
    --epochs 100 \
    --output ../../checkpoints/dinocell-frozen

# Option B: Fine-tune backbone (slower, potentially better performance)
python train.py \
    --dataset ../../datasets/LIVECell-train \
    --model-size small \
    --batch-size 4 \
    --epochs 100 \
    --output ../../checkpoints/dinocell-finetuned

# Option C: Multi-dataset generalist model
python train.py \
    --dataset ../../datasets/LIVECell-train ../../datasets/Cellpose-train \
    --model-size base \
    --freeze-backbone \
    --batch-size 4 \
    --epochs 100 \
    --output ../../checkpoints/dinocell-generalist
```

### 3. Evaluate Your Model

```bash
cd ../../evaluation

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
- Faster training (only the decoder is updated)
- Lower memory requirements
- Uses the pretrained features as they are

**Command**:
```bash
python train.py --dataset ../../datasets/LIVECell-train \
               --model-size small --freeze-backbone \
               --batch-size 8 --epochs 100
```


### Strategy 2: End-to-End Fine-Tuning

**When to use**:
- The frozen features are not enough
- More GPU memory is available
- Domain-specific fine-tuning

**Advantages**:
- Adapts the backbone features to cells

**Command**:
```bash
python train.py --dataset ../../datasets/LIVECell-train \
               --model-size small \
               --batch-size 4 --epochs 100
```


### Strategy 3: Multi-Dataset Generalist

**When to use**:
- Building a generalist model
- Aiming for zero-shot use on new datasets
- Multiple dataset sources available

**Advantages**:
- Sees more imaging conditions during training

**Command**:
```bash
python train.py --dataset ../../datasets/LIVECell-train ../../datasets/Cellpose-train \
               --model-size base --freeze-backbone \
               --batch-size 4 --epochs 100
```


## Model Sizes

| Size | Backbone | Parameters |
|------|----------|------------|
| Small | ViT-S/16 | 21M |
| Base | ViT-B/16 | 86M |
| Large | ViT-L/16 | 300M |
| 7B | ViT-7B/16 | 6.7B |

Memory use and speed per size were not measured, and the sizes were not compared on accuracy.

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
python train.py --dataset ../../datasets/LIVECell-train \
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

See the [repository README](../README.md) for usage examples.



