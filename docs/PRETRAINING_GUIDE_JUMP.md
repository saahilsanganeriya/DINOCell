# DINOv3 Pretraining on JUMP Cell Painting Dataset

Complete guide for pretraining DINOv3 on 3 million JUMP Cell Painting images for DINOCell.

## 🎯 Your Setup

- **Dataset**: JUMP Cell Painting (~3M images)
- **Channels**: 5 fluorescent + 3 brightfield per field
- **Hardware**: Single A100 GPU (24-48 hours available)
- **Goal**: Patch size 8 (higher resolution than standard 16)
- **Strategy**: Continue from pretrained checkpoint (faster convergence)

## 📊 Recommendations Based on Constraints

### Model Choice: **ViT-Small with Patch Size 8**

**Why ViT-Small?**
- ✅ Fits in 24-48 hours on single A100
- ✅ 3M images is sufficient for ViT-S (21M params)
- ✅ Patch size 8 = 4x more patches than p16 (manageable for ViT-S)
- ✅ Proven to work well with continued training
- ❌ ViT-Base would be too slow for your time limit
- ❌ ViT-Large would require multi-GPU or much longer

**Patch Size 8 Implications**:
- Standard 224×224 image: **28×28 = 784 patches** (vs 196 for p16)
- More computational cost (~4x) but higher resolution
- Better for small cell features
- Requires modifying DINOv3 configs

### Training Strategy: **Continue from Pretrained Checkpoint**

**Why continue training?**
- ✅ Faster convergence (5-10x faster than scratch)
- ✅ Better final performance (transfer learning)
- ✅ Fits in 24-48 hour window
- ✅ Recommended by DINOv3 team (1/10 LR)

**From scratch would need**: 1-2 weeks on your timeline

## 🎨 Multi-Channel Strategy

### Option 1: Channel Averaging (Recommended for Time Limit)

**Approach**: Convert multi-channel to grayscale by averaging

```python
# For each field of view:
# - Average 5 fluorescent channels → single image
# - Use middle brightfield plane → single image
# Total: ~3M single-channel images
```

**Advantages**:
- ✅ Simple data loading
- ✅ Standard DINOv3 pipeline
- ✅ Works with existing configs
- ✅ Fastest to implement

**Code**:
```python
# In your dataset loader
img_5channel = load_channels(path)  # (5, H, W)
img_gray = img_5channel.mean(axis=0)  # (H, W)
img_rgb = np.stack([img_gray]*3, axis=-1)  # (H, W, 3) for DINOv3
```

### Option 2: Multi-View Consistency (Better but Slower)

**Approach**: Treat different channels as different views of same content

Modify DINOv3's multi-crop strategy:
- Global crop 1: Average of all channels
- Global crop 2: Random single channel
- Local crops: Random channels

This enforces consistency across channels via DINO loss.

**Advantages**:
- ✅ Leverages multi-channel nature
- ✅ Learns channel-invariant features
- ❌ Requires custom data augmentation
- ❌ More complex implementation

### Recommendation: **Use Option 1 for 24-48hr timeline**

You can later fine-tune channel-specific heads after pretraining.

## 📝 Step-by-Step Guide

### Step 1: Setup Environment (30 minutes)

```bash
# Navigate to dinov3
cd ../dinov3

# Install dependencies
micromamba env create -f conda.yaml
micromamba activate dinov3

# or with pip
pip install -r requirements.txt
```

### Step 2: Prepare Dataset (2-4 hours)

Create a custom dataset loader for JUMP images:

```bash
cd dinov3/dinov3/data/datasets
```

Create `jump_cellpainting.py`:

```python
# See config file below - I'll create this
```

### Step 3: Create Training Config (see below)

### Step 4: Launch Training (24-48 hours)

```bash
cd dinov3

# Single GPU training
PYTHONPATH=. python dinov3/train/train.py \
    --config-file ../DINOCell/training/configs/dinov3_vits8_jump_pretrain.yaml \
    --output-dir ../DINOCell/checkpoints/dinov3_vits8_jump_pretrained
```

### Step 5: Use Pretrained Weights in DINOCell

```bash
cd ../DINOCell/training

# Fine-tune DINOCell using your pretrained backbone
python train.py \
    --dataset ../datasets/LIVECell-train \
    --model-size small \
    --backbone-weights ../checkpoints/dinov3_vits8_jump_pretrained/eval/final/teacher_checkpoint.pth \
    --epochs 100
```

## ⚙️ Training Configuration

Based on GitHub discussions and your constraints:

**Key Parameters**:
- Start from ViT-S pretrained checkpoint
- LR: 1/10 of original (5e-5 instead of 5e-4)
- Patch size: 8 (non-standard, requires config change)
- Resolution: 224 (standard) or 256 (slightly higher)
- Batch size: 32-64 per GPU (adjust for p8)
- Epochs: 50-100 (with early stopping)
- Disable Gram loss (not needed for continued training)
- Use bf16 (not fp16 to avoid NaN)

**Expected Training Time**:
- ~24-36 hours on single A100
- ~1500-2000 iterations per hour
- Total: ~50k-80k iterations

## 🚨 Important Notes from GitHub Issues

Based on community experience:

1. **Avoid NaN Loss**:
   - ✅ Use `bf16` (not fp16)
   - ✅ Set `qkv_bias: false` for ViT-B (or use default true carefully)
   - ✅ Monitor gradients
   - ✅ Lower learning rate if unstable

2. **Continue Training Best Practices**:
   - ✅ Use `student.pretrained_weights` to load backbone only
   - ✅ Don't use `student.resume_from_teacher_chkpt` (that's for high-res adaptation)
   - ✅ Start with 1/10 LR of original training
   - ✅ Shorter warmup (5-10 epochs vs 30)

3. **For Medical/Domain-Specific Images**:
   - ✅ Lower LR (5e-5 to 1e-4)
   - ✅ Longer warmup
   - ✅ Monitor loss carefully
   - ✅ May need different augmentation params

4. **Gram Loss**:
   - ❌ Not needed for distilled models (ViT-S/B/L)
   - ❌ Only use for training ViT-7B from scratch
   - ✅ Keep `gram.use_loss: false`

## ✅ Implementation Complete

I've created:
1. ✅ Custom dataset loader: `dinov3/dinov3/data/datasets/jump_cellpainting.py`
2. ✅ Training config: `training/configs/dinov3_vits8_jump_pretrain.yaml`
3. ✅ Launch script: `training/launch_pretraining.sh`
4. ✅ Updated dinov3 to support JUMP dataset

---

## 🚀 Complete Pretraining Workflow

### Prerequisites Check

```bash
# 1. Verify DINOv3 is available
ls ../dinov3
# Should show: dinov3/ folder, README.md, etc.

# 2. Verify JUMP dataset
ls ../../2024_Chandrasekaran_NatureMethods_CPJUMP1/
# Should show: 2020_11_04_CPJUMP1/ and other batch folders

# 3. Check GPU
nvidia-smi
# Should show: A100 with ~80GB memory
```

### Step-by-Step Execution

#### Step 1: Download Pretrained Checkpoint (5 minutes)

```bash
cd ../dinov3
mkdir -p checkpoints

# Download ViT-S/16 pretrained weights
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
    -O checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

#### Step 2: Setup Environment (10 minutes)

```bash
# Option A: Using micromamba (recommended)
micromamba env create -f conda.yaml
micromamba activate dinov3

# Option B: Using pip
pip install -r requirements.txt
```

#### Step 3: Verify Dataset Loader (5 minutes)

```bash
cd dinov3

# Test the JUMP dataset loader
PYTHONPATH=. python -c "
from dinov3.data.datasets import JUMPCellPainting
dataset = JUMPCellPainting(root='../../2024_Chandrasekaran_NatureMethods_CPJUMP1')
print(f'✅ Dataset loaded: {len(dataset)} samples')
sample = dataset[0]
print(f'✅ Sample loaded successfully')
"
```

**Expected output**:
```
Discovering JUMP Cell Painting images...
Found X fields in batch 2020_11_04_CPJUMP1
...
✅ Dataset loaded: ~3000000 samples
✅ Sample loaded successfully
```

#### Step 4: Launch Pretraining (24-48 hours)

```bash
cd ../DINOCell/training

# Make script executable (if not already)
chmod +x launch_pretraining.sh

# Launch training
./launch_pretraining.sh

# Or with custom parameters:
./launch_pretraining.sh train.batch_size_per_gpu=32 optim.epochs=60
```

**What happens**:
1. Loads pretrained ViT-S/16 weights
2. Modifies to patch size 8
3. Trains on JUMP images with SSL objectives
4. Saves checkpoints every 2500 iterations (~2.5 hours)
5. Evaluates every 5000 iterations
6. Auto-resumes if interrupted

#### Step 5: Monitor Training

```bash
# Watch training logs
tail -f ../../DINOCell/checkpoints/dinov3_vits8_jump_pretrained/logs/log.txt

# Check checkpoints
ls ../../DINOCell/checkpoints/dinov3_vits8_jump_pretrained/ckpt/
```

**Monitor for**:
- ✅ Loss decreasing steadily
- ✅ No NaN values
- ✅ Checkpoints saving regularly
- ⚠️ If loss plateaus, may need to adjust LR

**Expected loss values**:
- Initial: ~8-10 (continuing from pretrained)
- After 10k iters: ~6-8
- After 50k iters: ~4-6
- Final: ~3-5 (domain-specific should converge lower)

#### Step 6: Extract Final Checkpoint (5 minutes)

```bash
# After training completes, teacher checkpoint is at:
TEACHER_CKPT="../../DINOCell/checkpoints/dinov3_vits8_jump_pretrained/eval/final/teacher_checkpoint.pth"

# Copy to convenient location
cp $TEACHER_CKPT ../../DINOCell/checkpoints/dinov3_vits8_jump_pretrained.pth

echo "✅ Pretrained weights ready at:"
echo "   ../../DINOCell/checkpoints/dinov3_vits8_jump_pretrained.pth"
```

### Step 7: Fine-Tune DINOCell (3-4 hours)

```bash
cd ../../DINOCell/training

# Fine-tune DINOCell using JUMP-pretrained weights
python train.py \
    --dataset ../datasets/LIVECell-train \
    --model-size small \
    --backbone-weights ../checkpoints/dinov3_vits8_jump_pretrained.pth \
    --epochs 100 \
    --batch-size 8 \
    --output ../checkpoints/dinocell_jump_pretrained
```

---

## 🔧 Configuration Details

### Config File Breakdown

**File**: `training/configs/dinov3_vits8_jump_pretrain.yaml`

**Key Settings**:

```yaml
# Model: ViT-Small with Patch 8
student:
  arch: vit_small
  patch_size: 8  # ← Changed from 16
  pretrained_weights: 'path/to/vits16.pth'  # ← Load pretrained

# Optimization: 1/10 LR for continued training
optim:
  lr: 5.0e-5  # ← 1/10 of original 5e-4
  epochs: 80  # ← Shorter than from-scratch
  warmup_epochs: 10  # ← Shorter warmup

# Data: Adapted for cells
crops:
  global_crops_size: 224  # ← 28×28 patches with p8
  local_crops_size: 96   # ← 12×12 patches with p8
  global_crops_scale: [0.4, 1.0]  # ← Larger min for cells

# Hardware: Single GPU
train:
  batch_size_per_gpu: 48  # ← Adjust based on memory
  OFFICIAL_EPOCH_LENGTH: 1000  # ← ~3M / 48 / 1 GPU
```

### Dataset Loader

**File**: `dinov3/dinov3/data/datasets/jump_cellpainting.py`

**What it does**:
1. Discovers all fields across 6 batches
2. Groups 5 fluorescent + 3 brightfield channels per field
3. Converts to RGB by averaging channels (or using specific channel)
4. Applies CLAHE preprocessing
5. Returns PIL Image compatible with DINOv3

**Channel modes**:
- `'average'`: Average all 5 fluorescent channels (recommended)
- `'brightfield'`: Use middle brightfield plane
- `'channel1'`, `'channel2'`, etc.: Use specific channel

---

## ⏱️ Training Timeline (Single A100)

### Estimated Breakdown

| Phase | Duration | Iterations | What's Happening |
|-------|----------|------------|------------------|
| **Setup** | 1 hour | 0 | Load checkpoint, initialize |
| **Warmup** | 2-3 hours | 0-10k | Learning rate warmup |
| **Training** | 20-30 hours | 10k-60k | Main SSL training |
| **Final** | 1-2 hours | 60k-65k | Convergence |
| **Total** | **24-36 hours** | **~65k** | Complete pretraining |

### Checkpointing Strategy

- **Automatic saves**: Every 2500 iterations (~2.5 hours)
- **Evaluation**: Every 5000 iterations (~5 hours)
- **Kept checkpoints**: Last 3 only (to save disk space)

**Resume capability**: If training is interrupted, just run the script again - it auto-resumes from latest checkpoint!

---

## 📊 Multi-Channel Handling Strategy

### Your Insight: "Same cells, different channels"

This is brilliant! Here are three approaches:

### Approach 1: Average Channels (Implemented - Fastest)

**What**: Average 5 fluorescent channels into single grayscale

```python
# In JUMPImageDecoder
channels = [load(ch1), load(ch2), ..., load(ch5)]
img_avg = np.mean(channels, axis=0)
```

**Pros**:
- ✅ Simple, fast
- ✅ Works with standard DINOv3
- ✅ Captures all channel information

**Cons**:
- ❌ Loses channel-specific details

### Approach 2: Multi-View Consistency (Advanced - For Later)

**What**: Treat channels as multiple views in DINO framework

Modify crops to use different channels:
- Global crop 1: Average of all channels
- Global crop 2: Random single channel
- Forces consistency via DINO loss

**Implementation** (if you want to try):
1. Modify `dinov3/dinov3/data/augmentations.py`
2. Load different channels for different crops
3. DINO loss naturally enforces consistency

**Benefit**: Model learns "same cell" regardless of channel!

### Approach 3: Sequential Fine-Tuning (Your Idea)

**What**: 
1. Pretrain on averaged channels (this guide)
2. Later fine-tune separate distance map heads per channel

**Workflow**:
```
SSL Pretrain on avg channels (24-48h)
         ↓
Fine-tune head on Ch1 cells (2h)
Fine-tune head on Ch2 cells (2h)
Fine-tune head on Ch5 cells (2h)
Fine-tune head on Brightfield (2h)
```

**Total time**: ~30-36 hours for all modalities!

---

## 🐛 Troubleshooting

### Issue 1: NaN Loss During Training

**Symptoms**:
```
ibot_loss: nan
backbone_grad_norm: nan
```

**Solutions**:
```yaml
# Option A: Disable problematic loss
ibot:
  loss_weight: 0.0  # Temporarily disable iBOT

# Option B: Use fp32 for attention
compute_precision:
  param_dtype: fp32  # Was bf16

# Option C: Disable qkv_bias
student:
  qkv_bias: false  # Helps with ViT-B, try for ViT-S if needed

# Option D: Lower learning rate further
optim:
  lr: 1.0e-5  # Was 5e-5
```

### Issue 2: Out of Memory

**Solutions**:
```yaml
# Reduce batch size
train:
  batch_size_per_gpu: 32  # Was 48

# Enable gradient checkpointing
train:
  checkpointing: true

# Smaller crops
crops:
  global_crops_size: 192  # Was 224
```

### Issue 3: Too Slow / Won't Finish in 48h

**Solutions**:

```yaml
# Option A: Fewer epochs
optim:
  epochs: 50  # Was 80

# Option B: Longer epoch length (fewer epochs, same iterations)
train:
  OFFICIAL_EPOCH_LENGTH: 2000  # Was 1000

# Option C: Early stopping
# Just stop training after 40 hours and use that checkpoint
```

### Issue 4: Checkpoint Loading Fails

**If you get**: `KeyError: 'teacher'` when loading checkpoint

**Solution**:
The public checkpoints don't have 'teacher' key. The config is set up to handle this correctly via `student.pretrained_weights`, not `student.resume_from_teacher_chkpt`.

**Verify**:
```python
import torch
ckpt = torch.load('dinov3_vits16_pretrain_lvd1689m-08c60483.pth')
print(ckpt.keys())  # Should show model weights, not 'teacher' key
```

---

## 📈 Expected Results

### Training Metrics

**Healthy training looks like**:
```
Epoch 10:
  dino_local_crops_loss: 7.2
  dino_global_crops_loss: 6.8
  koleo_loss: -0.3
  ibot_loss: 8.5
  
Epoch 50:
  dino_local_crops_loss: 4.5
  dino_global_crops_loss: 4.2
  koleo_loss: -0.4
  ibot_loss: 5.8
```

**Signs of good convergence**:
- ✅ Losses decreasing steadily
- ✅ No NaN values
- ✅ Koleo loss negative and stable
- ✅ Gradient norms reasonable (1-100)

### Performance Expectations

**After pretraining, expect**:
- Better features for cell images than vanilla DINOv3
- Improved generalization on cell segmentation
- Especially good for multi-channel microscopy
- Should outperform SAM-based approaches

**Quantitative** (to be validated):
- Linear probe accuracy on JUMP test set
- K-NN retrieval of similar cells
- Distance map prediction quality

---

## 🎯 After Pretraining: Fine-Tuning Guide

### Using Pretrained Weights in DINOCell

**Method 1: Direct Loading** (Recommended)

```python
# In DINOCell model.py, when loading backbone:
model = create_dinocell_model(
    model_size='small',
    backbone_weights='../checkpoints/dinov3_vits8_jump_pretrained.pth',
    freeze_backbone=False,  # Fine-tune or freeze as needed
    pretrained=False  # Don't load default weights
)
```

**Method 2: Training Script**

```bash
python train.py \
    --dataset ../datasets/LIVECell-train \
    --model-size small \
    --backbone-weights ../checkpoints/dinov3_vits8_jump_pretrained.pth \
    --epochs 100 \
    --freeze-backbone  # Or omit to fine-tune
```

### Channel-Specific Fine-Tuning

**For each channel modality**:

```bash
# Fluorescence Ch1 (Nuclei)
python train.py \
    --dataset ../datasets/JUMP_Ch1_cells \
    --backbone-weights ../checkpoints/dinov3_vits8_jump_pretrained.pth \
    --freeze-backbone --output ../checkpoints/dinocell_ch1

# Fluorescence Ch2 (ER)
python train.py \
    --dataset ../datasets/JUMP_Ch2_cells \
    --backbone-weights ../checkpoints/dinov3_vits8_jump_pretrained.pth \
    --freeze-backbone --output ../checkpoints/dinocell_ch2

# Brightfield
python train.py \
    --dataset ../datasets/JUMP_brightfield_cells \
    --backbone-weights ../checkpoints/dinov3_vits8_jump_pretrained.pth \
    --freeze-backbone --output ../checkpoints/dinocell_brightfield
```

---

## 📊 Monitoring Training

### Real-Time Monitoring

**Terminal 1** (Training):
```bash
./launch_pretraining.sh
```

**Terminal 2** (Logs):
```bash
# Watch training progress
tail -f ../../DINOCell/checkpoints/dinov3_vits8_jump_pretrained/logs/log.txt

# Or with grep for key metrics
tail -f ../../DINOCell/checkpoints/dinov3_vits8_jump_pretrained/logs/log.txt | \
    grep -E "EPOCH|loss|grad_norm"
```

**Terminal 3** (GPU Usage):
```bash
watch -n 1 nvidia-smi
```

### Parse Training Metrics

```python
import json
import pandas as pd
import matplotlib.pyplot as plt

# Load metrics
metrics_file = "../../DINOCell/checkpoints/dinov3_vits8_jump_pretrained/training_metrics.json"
with open(metrics_file) as f:
    metrics = pd.DataFrame([json.loads(line) for line in f])

# Plot losses
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].plot(metrics.iteration, metrics.dino_local_crops_loss)
axes[0, 0].set_title('DINO Local Loss')

axes[0, 1].plot(metrics.iteration, metrics.dino_global_crops_loss)
axes[0, 1].set_title('DINO Global Loss')

axes[1, 0].plot(metrics.iteration, metrics.ibot_loss)
axes[1, 0].set_title('iBOT Loss')

axes[1, 1].plot(metrics.iteration, metrics.koleo_loss)
axes[1, 1].set_title('KoLeo Loss')

plt.tight_layout()
plt.savefig('training_progress.png')
print("✅ Saved training progress to training_progress.png")
```

---

## ⚙️ Configuration Recommendations by Scenario

### Scenario A: Maximum Quality (48 hours available)

```yaml
optim:
  epochs: 100
  lr: 5.0e-5
  
train:
  batch_size_per_gpu: 48
  OFFICIAL_EPOCH_LENGTH: 1000
```

**Expected**: ~65k-70k iterations

### Scenario B: Faster Convergence (24 hours)

```yaml
optim:
  epochs: 60
  lr: 1.0e-4  # Slightly higher LR
  warmup_epochs: 5  # Shorter warmup
  
train:
  batch_size_per_gpu: 64  # If memory allows
  OFFICIAL_EPOCH_LENGTH: 800
```

**Expected**: ~48k iterations

### Scenario C: Conservative (Avoid NaN)

```yaml
optim:
  lr: 2.0e-5  # Very conservative
  epochs: 80
  
student:
  qkv_bias: false  # More stable
  drop_path_rate: 0.05  # Lower dropout

compute_precision:
  param_dtype: fp32  # Use fp32 if bf16 causes issues
```

---

## 🎁 What You'll Get

### Outputs

After 24-48 hours of training:

```
checkpoints/dinov3_vits8_jump_pretrained/
├── ckpt/
│   ├── 2500/  # Checkpoint at iteration 2500
│   ├── 5000/
│   └── ...
├── eval/
│   ├── final/
│   │   └── teacher_checkpoint.pth  ← Use this for DINOCell!
│   ├── 5000/
│   └── ...
├── logs/
│   └── log.txt
├── training_metrics.json
└── config.yaml
```

### Expected Benefits

**Compared to vanilla DINOv3 on cells**:
- ✅ Better cell-specific features
- ✅ Better separation of cell boundaries
- ✅ Better handling of multi-channel data
- ✅ Improved zero-shot performance

**Compared to SAM**:
- ✅ 155x more pretraining images (1.7B vs 11M)
- ✅ Self-supervised learning (better generalization)
- ✅ Cell-specific adaptation (3M cell images)
- ✅ Higher resolution features (patch 8 vs 16)

---

## 🔬 Advanced: Multi-View Consistency (Future Work)

If you want to leverage all channels simultaneously:

### Modified Augmentation Strategy

```python
# In dinov3/dinov3/data/augmentations.py
# Modify DataAugmentationDINO class

class JUMPDataAugmentation:
    def __call__(self, image_channels):
        # image_channels: list of 5 channels
        
        # Global crop 1: Average all channels
        crop1 = self.geometric_augmentation(average_channels(image_channels))
        
        # Global crop 2: Random single channel
        channel_idx = random.randint(0, 4)
        crop2 = self.geometric_augmentation(image_channels[channel_idx])
        
        # DINO loss will enforce consistency between these views!
        # i.e., model learns "same cell" regardless of channel
        
        return {'global_crops': [crop1, crop2], ...}
```

**Benefit**: Model learns channel-invariant cell features automatically!

This is more complex but could be very powerful. For your 24-48hr timeline, stick with the averaging approach.

---

## 📞 Quick Reference Commands

```bash
# Download checkpoint
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

# Test dataset
PYTHONPATH=. python -c "from dinov3.data.datasets import JUMPCellPainting; d=JUMPCellPainting(root='...'); print(len(d))"

# Launch training
cd DINOCell/training && ./launch_pretraining.sh

# Monitor
tail -f ../../DINOCell/checkpoints/dinov3_vits8_jump_pretrained/logs/log.txt

# Stop gracefully
# Ctrl+C (will save checkpoint)

# Resume
./launch_pretraining.sh  # Auto-resumes from latest

# Use weights
python train.py --backbone-weights ../checkpoints/dinov3_vits8_jump_pretrained.pth
```

---

## 🎊 Summary

### Your Pretraining Plan:

1. ✅ **Model**: ViT-Small with Patch-8 (21M parameters, higher resolution)
2. ✅ **Strategy**: Continue from pretrained checkpoint (10x faster)
3. ✅ **Dataset**: 3M JUMP images averaged across channels
4. ✅ **Hardware**: Single A100 for 24-48 hours
5. ✅ **Config**: Custom config with patch-8 and lower LR
6. ✅ **Output**: Cell-adapted DINOv3 backbone for DINOCell

### Timeline:
- Setup: 30 minutes
- Training: 24-36 hours (automated)
- Fine-tuning DINOCell: 3-4 hours
- **Total**: ~28-41 hours

### Expected Outcome:
- DINOv3 backbone adapted for cell microscopy
- Higher resolution features (patch-8 vs patch-16)
- Ready for DINOCell distance map fine-tuning
- Should outperform vanilla DINOv3 and SAM on cells

---

## 🚀 Ready to Start?

```bash
cd DINOCell/training
./launch_pretraining.sh
```

Let it run for 24-48 hours, then use the pretrained weights for DINOCell training!

**Good luck with your pretraining! 🔬✨**

