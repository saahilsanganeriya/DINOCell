# SSL Pretraining with JUMP Cell Painting Dataset

Complete guide for self-supervised pretraining of DINOv3 on 3 million JUMP Cell Painting images.

## 🎯 Overview

This guide covers advanced SSL pretraining with:
- **Multi-view consistency learning** across 5 fluorescent channels
- **AWS S3 streaming** (no local storage needed)
- **Wandb monitoring** with attention maps and metrics
- **Patch size 8** for higher resolution

## 🚀 Quick Start

```bash
cd training/ssl_pretraining

# Install S3 and wandb dependencies
pip install boto3 smart-open wandb

# Login to wandb (optional but recommended)
wandb login

# Launch pretraining!
./launch_ssl_with_s3_wandb.sh
```

**That's it!** Training will run for 30-40 hours on A100.

## 📊 What Happens During Training

### Multi-View Consistency Learning

Your JUMP images have 5 fluorescent channels showing the SAME cells:
- **Channel 1**: Golgi, Plasma Membrane
- **Channel 2**: ER, AGP, Mitochondria  
- **Channel 3**: RNA
- **Channel 4**: Actin, ER, Mitochondria
- **Channel 5**: Nuclei (Hoechst)

**Standard approach**: Average all 5 → train  
**Our approach**: Teach model "these are the same cells!"

**How it works**:
1. Global crop 1: Average of all 5 channels
2. Global crop 2: Random single channel (ch1, ch2, ch3, ch4, or ch5)
3. DINO loss enforces: `features(averaged) ≈ features(single_channel)`
4. Result: Model learns channel-invariant representations!

### AWS S3 Streaming

Instead of downloading 500GB of images:
- Images streamed directly from `s3://cellpainting-gallery`
- LRU cache keeps 1000 recent images in RAM
- No AWS credentials needed (public bucket)
- Start training immediately!

### Wandb Logging

Every 100 iterations:
- Loss curves (DINO, iBOT, KoLeo)
- Learning rate schedule
- Gradient statistics

Every 1000 iterations:
- Attention map visualizations
- Feature PCA plots
- Channel consistency metrics

## 🎛️ Configuration

The config file: `configs/dinov3_vits8_jump_s3_multiview.yaml`

Key settings:
```yaml
student:
  arch: vit_small          # 21M parameters
  patch_size: 8            # Higher resolution than standard 16
  pretrained_weights: '<DINOV3_CHECKPOINT_URL>'  # Continue from pretrained

train:
  batch_size_per_gpu: 64   # Fits in A100 80GB
  dataset_path: 'JUMPS3MultiView:bucket=cellpainting-gallery:prefix=...'
  
optim:
  epochs: 100
  lr: 5e-5                  # 1/10 of original (continued training)
  
wandb:
  enabled: true
  project: 'dinocell-ssl-pretraining'
  log_attention_maps: true
```

## 📈 Expected Results

**Timeline** (Single A100 80GB):
- Iteration 0: ~0.8s/iter
- After warmup (1k iters): ~0.7s/iter  
- **Total**: ~30-40 hours for 100 epochs

**Metrics**:
- DINO loss: Starts ~4.5 → Converges ~2.8
- iBOT loss: Starts ~6.0 → Converges ~4.2
- KoLeo loss: Should stay ~0.1-0.3 (prevents collapse)

**Channel Consistency** (validate with `validate_channel_consistency.py`):
- Before: ~0.4-0.6 cosine similarity across channels
- After: ~0.85-0.95 cosine similarity (channel-invariant!)

## 🔍 Monitor Training

Open Wandb dashboard:
```bash
# Your run URL will be printed at start, e.g.:
# https://wandb.ai/your-entity/dinocell-ssl-pretraining/runs/xxx
```

Watch for:
1. **Losses decreasing steadily**
2. **Attention maps sharpening** (focusing on cells)
3. **Channel consistency improving**

## 🛑 Troubleshooting

### Out of Memory

Reduce batch size in config:
```yaml
train:
  batch_size_per_gpu: 48  # Down from 64
```

### S3 Connection Issues

Test S3 access:
```bash
python -c "
import boto3
from botocore import UNSIGNED
from botocore.config import Config
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
response = s3.list_objects_v2(
    Bucket='cellpainting-gallery',
    Prefix='cpg0000-jump-pilot/source_4/images/',
    MaxKeys=10
)
print(f'✅ S3 accessible! Found {len(response.get(\"Contents\", []))} objects')
"
```

If this fails, fall back to local dataset:
- Edit config: `dataset_path: 'JUMPCellPaintingMultiView:root=/path/to/local/jump'`

### Wandb Login Issues

Disable wandb:
```bash
./launch_ssl_with_s3_wandb.sh --no-wandb
```

## 📁 Output Structure

```
training/ssl_pretraining/
└── output/
    ├── config.yaml              # Saved config
    ├── ckpt/                    # Checkpoints
    │   ├── 10000/              # Every 1k iterations
    │   ├── 20000/
    │   └── final/              # Final checkpoint
    ├── eval/                    # Evaluation checkpoints
    │   └── 12500/              # Every 12.5k iters
    └── logs/
        └── log.txt             # Training logs
```

## 🔄 Resume Training

Training automatically resumes from the latest checkpoint:
```bash
# Just rerun the same command
./launch_ssl_with_s3_wandb.sh
```

It will find the latest checkpoint in `output/ckpt/` and continue.

## ✅ After Pretraining

Once complete, you'll have:
- **Pretrained backbone**: `output/eval/final/teacher_checkpoint.pth`
- **Wandb run**: Full metrics, plots, and visualizations

Use this checkpoint for DINOCell fine-tuning:
```bash
cd ../finetune
python train.py \
    --backbone-weights ../ssl_pretraining/output/eval/final/teacher_checkpoint.pth \
    --dataset ../../datasets/LIVECell-train \
    --freeze-backbone
```

## 🧪 Validate Results

Test channel consistency:
```bash
python validate_channel_consistency.py \
    --model output/eval/final/teacher_checkpoint.pth \
    --test-images /path/to/jump/test/fields
```

Should show >0.85 cosine similarity across all channel pairs!

## 📚 Technical Details

See full technical documentation:
- [Multi-view implementation](../docs/MULTIVIEW_IMPLEMENTATION.md)
- [S3 streaming details](../docs/S3_STREAMING.md)
- [Wandb integration](../docs/WANDB_LOGGING.md)

## 🤝 Comparison with Original DINOv3

| Aspect | Original DINOv3 | Our Adaptation |
|--------|----------------|----------------|
| Dataset | ImageNet-21k (14M) | JUMP Cell Painting (3M) |
| Images | Natural images (RGB) | Microscopy (5-channel) |
| Patch size | 14 or 16 | **8** (higher res) |
| Augmentation | Standard RGB | **Multi-channel consistency** |
| Storage | Local download required | **S3 streaming** |
| Logging | Minimal | **Wandb with visualizations** |
| Time (1 GPU) | N/A (needs 32+ GPUs) | **30-40 hrs on single A100** |

## 🎓 Citation

If you use SSL pretraining, please cite both DINOCell and DINOv3:

```bibtex
@software{dinocell2025,
  title={DINOCell: Cell Segmentation with DINOv3},
  year={2025},
}

@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Siméoni, Oriane and Vo, Huy V. and others},
  year={2025},
  eprint={2508.10104},
}
```


