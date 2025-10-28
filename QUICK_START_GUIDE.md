# 🚀 DINOCell Quick Start Guide

**For future users who clone this repository**

---

## Installation (5 minutes)

### Step 1: Create Environment
```bash
conda env create -f environment.yml
conda activate dinocell
```

### Step 2: Install Packages
```bash
# Install dinov3
cd dinov3_modified/dinov3
pip install -e .
cd ../..

# Install DINOCell  
pip install -e .
```

### Step 3: Verify
```bash
python -c "import torch; from dinov3.data.datasets import JUMPS3Dataset; print('✅ Ready!')"
```

---

## Quick Test (5 minutes)

Test the pipeline before starting 30-40hr training:

```bash
cd training/ssl_pretraining
bash test_local.sh
```

This will:
- Load 10 example fields
- Run 50 training iterations
- Verify everything works
- Complete in ~1 minute

---

## Full Training (30-40 hours)

### Configure Wandb & Storage
```bash
# Set wandb cache location
export WANDB_DIR=/home/shadeform/wandb_cache
export WANDB_CACHE_DIR=/home/shadeform/wandb_cache
export WANDB_DATA_DIR=/home/shadeform/wandb_cache

# Login to wandb
wandb login
```

### Launch Training
```bash
cd training/ssl_pretraining
bash launch_ssl_with_s3_wandb.sh
```

### Monitor Training
```bash
# Quick status
bash monitor_training.sh

# Watch logs
tail -f full_training.log

# GPU monitoring
watch -n 1 nvidia-smi
```

---

## What Gets Trained

- **Model:** DINOv3 ViT-Small with patch size 8
- **Dataset:** 3M JUMP Cell Painting images (streamed from S3)
- **Method:** Multi-view consistency learning (channel-invariant)
- **Duration:** 30-40 hours on A100-80GB
- **Output:** Pretrained backbone for DINOCell

---

## Wandb Dashboard

Training metrics and attention maps will appear at:
```
https://wandb.ai/[your-username]/dinocell-ssl-pretraining
```

Logged every 100 iterations:
- Loss values
- Learning rate
- Gradient norms

Logged every 1000 iterations:
- Attention maps overlayed on cell images
- Feature visualizations

---

## After Training

### Extract Checkpoint
```bash
cp output/eval/final/teacher_checkpoint.pth \
   ../../checkpoints/dinov3_vits8_jump_s3_multiview.pth
```

### Fine-tune DINOCell
```bash
cd ../finetune
python train.py \
    --dataset ../../datasets/LIVECell-train \
    --backbone-weights ../../checkpoints/dinov3_vits8_jump_s3_multiview.pth \
    --model-size small \
    --freeze-backbone
```

---

## Troubleshooting

### Training not starting?
```bash
# Check process
ps aux | grep train.py

# Check logs
tail -100 full_training.log

# Restart if needed
pkill -f train.py
bash launch_ssl_with_s3_wandb.sh
```

### Wandb not logging?
```bash
# Check login
wandb status

# Re-login
wandb login --relogin
```

### Out of disk space?
```bash
# Check space
df -h /

# Clean old logs if needed
rm -rf test_output/ pretraining_*.log
```

---

## Key Files

| File | Purpose |
|------|---------|
| `environment.yml` | Conda environment specification |
| `requirements.txt` | All dependencies |
| `SETUP.md` | Detailed installation guide |
| `test_local.sh` | Quick test script |
| `launch_ssl_with_s3_wandb.sh` | Full training launcher |
| `monitor_training.sh` | Training monitor |

---

**Everything is automated and documented. Happy training! 🔬✨**

