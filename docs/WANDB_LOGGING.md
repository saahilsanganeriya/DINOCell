# Weights & Biases Integration

Comprehensive training monitoring with Wandb for DINOCell SSL pretraining.

## 🎯 What Gets Logged

### Every 100 Iterations
- **Loss curves**: DINO, iBOT, KoLeo losses
- **Learning rate**: Current LR from cosine schedule
- **Gradient stats**: Mean, std, max gradients
- **System metrics**: GPU memory, utilization

### Every 1000 Iterations
- **Attention maps**: Visualize what the model attends to
- **Feature PCA**: 2D projection of learned features
- **Channel consistency**: Cosine similarity across channels

### Every Checkpoint (10k iterations)
- **Model artifacts**: Optionally upload checkpoints to wandb

## 🚀 Quick Setup

### 1. Install Wandb

```bash
pip install wandb
```

### 2. Login

```bash
wandb login
# Paste your API key from https://wandb.ai/authorize
```

### 3. Configure

Edit config: `training/ssl_pretraining/configs/dinov3_vits8_jump_s3_multiview.yaml`

```yaml
wandb:
  enabled: true
  project: 'dinocell-ssl-pretraining'  # Your project name
  entity: 'your-username'              # Your wandb username/team
  name: 'vits8-jump-multiview'         # Run name (auto-generated if null)
  log_attention_maps: true             # Enable attention visualizations
  log_interval: 100                    # Log metrics every N iters
  attention_log_interval: 1000         # Log attention every N iters
```

### 4. Launch Training

```bash
cd training/ssl_pretraining
./launch_ssl_with_s3_wandb.sh
```

Your wandb URL will be printed:
```
Wandb logging enabled: https://wandb.ai/your-username/dinocell-ssl-pretraining/runs/xyz
```

## 📊 Dashboard Overview

### Main Metrics Panel

**Losses** (lower is better):
- `loss/dino`: DINO self-distillation loss (should decrease 4.5 → 2.8)
- `loss/ibot`: iBOT masked prediction loss (should decrease 6.0 → 4.2)  
- `loss/koleo`: KoLeo regularization (should stay ~0.1-0.3)
- `loss/total`: Combined weighted loss

**Training**:
- `training/lr`: Learning rate (warmup then cosine decay)
- `training/epoch`: Current epoch
- `training/iteration`: Current iteration
- `system/gpu_memory_gb`: GPU memory usage

**Gradients**:
- `gradients/backbone_mean`: Average gradient magnitude
- `gradients/decoder_mean`: Decoder gradients
- `gradients/global_norm`: Global gradient norm

### Attention Maps

Every 1000 iterations, visualizations show:
- **Top row**: Original images (averaged channels)
- **Bottom row**: Attention from [CLS] token to patches

**What to look for**:
- Early training: Attention scattered randomly
- Mid training: Attention focusing on high-contrast regions
- Late training: Attention sharply focused on cell centers/boundaries

### Feature PCA

2D projection of learned features colored by:
- Channel type (multi-channel consistency)
- Image batch (data diversity)

**Good behavior**:
- Clusters forming based on cell types
- Channels overlapping (channel-invariant features)

### Channel Consistency

Cosine similarity between features from different channels:
- `consistency/ch1_vs_ch2`: Channel 1 vs 2 similarity
- `consistency/ch1_vs_ch3`: Channel 1 vs 3 similarity
- ... (all 10 pairs)
- `consistency/average`: Average across all pairs

**Target**: >0.85 average similarity = channel-invariant!

## 🎨 Custom Logging

### Add Custom Metrics

Edit `dinov3_modified/dinov3/logging/wandb_logger.py`:

```python
# In your training loop
from dinov3.logging.wandb_logger import create_wandb_logger

wandb_logger = create_wandb_logger(cfg)

# Log custom metric
wandb_logger.log_metrics({
    'custom/my_metric': my_value
}, step=iteration)
```

### Log Images

```python
import wandb

# Log single image
wandb.log({
    "visualization/my_image": wandb.Image(image_array)
}, step=iteration)

# Log multiple images
wandb.log({
    "gallery": [wandb.Image(img) for img in images]
}, step=iteration)
```

### Log Histograms

```python
wandb_logger.log_histogram(
    name='backbone/weights', 
    values=model.backbone.weight.flatten(),
    step=iteration
)
```

## 📈 Monitoring Best Practices

### 1. Watch for Overfitting

Plot training vs validation loss:
- Divergence indicates overfitting
- Solution: Reduce epochs or add regularization

### 2. Check Gradient Health

- **Too large** (>10): Potential instability
- **Too small** (<0.001): Potential vanishing gradients
- **Just right** (0.01-1.0): Healthy training

### 3. Monitor GPU Utilization

- **<50%**: Increase batch size or reduce preprocessing
- **90-95%**: Optimal usage
- **100% with OOM**: Reduce batch size

### 4. Attention Quality

Good attention maps show:
- Focused patterns (not random noise)
- Semantic meaning (cells, boundaries)
- Sharpening over time

## 🛑 Troubleshooting

### Wandb not logging

Check configuration:
```python
import wandb
print(f"Wandb version: {wandb.__version__}")
print(f"Logged in: {wandb.api.api_key is not None}")
```

### Attention maps not showing

Ensure in config:
```yaml
wandb:
  log_attention_maps: true
```

And model has attention export:
```python
# In model code
def get_last_selfattention(self, x):
    return self.blocks[-1].attn.get_attention_map(x)
```

### Too much logging (slow training)

Reduce logging frequency:
```yaml
wandb:
  log_interval: 200          # Default: 100
  attention_log_interval: 5000  # Default: 1000
```

## 📦 Offline Mode

No internet during training? Use offline mode:

```bash
export WANDB_MODE=offline
./launch_ssl_with_s3_wandb.sh
```

Then sync later:
```bash
wandb sync training/ssl_pretraining/output/wandb/latest-run
```

## 🎯 Wandb Features We Use

### 1. Run Comparison

Compare multiple pretraining runs:
- Different patch sizes (8 vs 16)
- Channel strategies (averaging vs multi-view)
- Learning rates

### 2. Hyperparameter Sweeps

```bash
wandb sweep sweep_config.yaml
wandb agent your-username/project/sweep-id
```

### 3. Artifacts

Upload checkpoints to wandb:
```python
wandb_logger.log_model_checkpoint(
    checkpoint_path='output/ckpt/final/checkpoint.pth',
    step=iteration
)
```

### 4. Reports

Create shareable reports with:
- Training curves
- Best attention maps
- Final metrics

## 📝 Example Wandb Config

Complete configuration example:

```yaml
wandb:
  # Basic settings
  enabled: true
  project: 'dinocell-ssl'
  entity: 'my-team'
  name: null  # Auto-generated: vits8-jump-multiview-<timestamp>
  group: 'jump-experiments'  # Group related runs
  tags: ['vit-small', 'patch-8', 'multiview']
  
  # Logging frequency
  log_interval: 100              # Metrics every 100 iters
  attention_log_interval: 1000   # Attention every 1k iters
  pca_log_interval: 5000         # PCA every 5k iters
  
  # What to log
  log_attention_maps: true
  log_feature_pca: true
  log_gradients: true
  log_model_checkpoints: false   # Large files! Enable carefully
  
  # Advanced
  offline: false                 # Set true for offline mode
  resume: 'allow'               # Allow resuming runs
```

## 🌟 Pro Tips

1. **Use groups** - Organize related experiments
2. **Add tags** - Easy filtering (model-size, dataset, etc.)
3. **Write notes** - Document insights directly in wandb
4. **Share reports** - Collaborate with team
5. **Download artifacts** - Get best checkpoints from any run

## 📚 Resources

- [Wandb Documentation](https://docs.wandb.ai/)
- [PyTorch Integration](https://docs.wandb.ai/guides/integrations/pytorch)
- [Experiment Tracking](https://docs.wandb.ai/guides/track)
- [Attention Visualization](https://wandb.ai/wandb/attention-visualization)

## 🎓 Example Wandb Run

Here's what a good training run looks like:

**Losses** (smoothed):
```
DINO:  4.5 → 4.2 → 3.8 → 3.4 → 3.0 → 2.8 ✓
iBOT:  6.0 → 5.6 → 5.2 → 4.8 → 4.4 → 4.2 ✓
KoLeo: 0.2 → 0.15 → 0.12 → 0.10 → 0.08 ✓
```

**Channel Consistency**:
```
Iteration 0:    0.45 (random)
Iteration 10k:  0.62 (improving)
Iteration 30k:  0.78 (good)
Iteration 50k:  0.87 (excellent) ✓
```

**Attention Quality**:
```
Early: Scattered, noisy patterns
Mid:   Focusing on high-contrast regions
Late:  Sharp focus on cell structures ✓
```

## 🏁 Summary

Wandb integration provides:
- ✅ Real-time training monitoring
- ✅ Attention and feature visualizations  
- ✅ Easy experiment comparison
- ✅ Shareable results
- ✅ Checkpoint management

Essential for successful SSL pretraining on large microscopy datasets!


