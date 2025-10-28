# Wandb Logging Configuration - Optimized

## ✅ Current Settings (Perfect for Your Needs)

```yaml
wandb:
  enabled: true
  log_interval: 1                    # ← Logs losses EVERY iteration
  log_attention_maps: true
  attention_log_interval: 100        # ← Attention images every 100 iterations only
```

---

## 📊 What Gets Logged When

### Every Iteration (Low Storage)
**Logged:** Scalars only (small data size)

```python
Metrics logged at iteration 0, 1, 2, 3, ..., 90000:
{
    'total_loss': 18.44,
    'dino_local_crops_loss': 11.11,
    'dino_global_crops_loss': 11.11,
    'ibot_loss': 5.56,
    'koleo_loss': 8.77,
    'lr': 1.5e-7,  # Actual value (displays as charts)
    'wd': 0.04,
    'teacher_temp': 0.04,
    'backbone_grad_norm': 890.0,
    'dino_head_grad_norm': 0.015,
    'ibot_head_grad_norm': 0.51
}
```

**Storage:** ~100 bytes per iteration × 90,000 = ~9MB total
**Benefit:** Smooth loss curves in wandb charts

### Every 100 Iterations (High Storage)
**Logged:** Attention map images

```python
At iterations 100, 200, 300, ..., 90000:
{
    'attention_maps': wandb.Image(...)  # 2×4 grid visualization
}
```

**Storage:** ~500KB per image × 900 logs = ~450MB total
**Benefit:** Visual monitoring without overwhelming storage

---

## 🎯 What You'll See in Wandb Dashboard

### Charts Tab (Real-time)
Once training starts (after S3 discovery), you'll see:

**Loss Charts:**
- `total_loss` - Main training loss (should decrease)
- `dino_local_crops_loss` - Local crop consistency
- `dino_global_crops_loss` - Global crop consistency  
- `ibot_loss` - Masked image modeling
- `koleo_loss` - Feature diversity

**Training Charts:**
- `lr` - Learning rate (will show warmup ramp from 0 → 5e-5)
- `wd` - Weight decay
- `teacher_temp` - Teacher temperature
- `backbone_grad_norm` - Gradient magnitude

All updating **every iteration** with smooth curves!

### Media/Images Tab
Starting at iteration 100, then every 100 iterations:
- Attention map visualizations
- Cell images with attention overlays
- 900 total images over full training

---

## 📈 Learning Rate Display Issue - Explained

### Why LR Shows as 0.0000

**Actual values during early warmup:**
```
Iteration 10:   LR = 5e-8  → displays as 0.0000
Iteration 30:   LR = 1.5e-7 → displays as 0.0000
Iteration 100:  LR = 5e-7  → displays as 0.0000
Iteration 1000: LR = 5e-6  → displays as 0.0000
Iteration 5000: LR = 2.5e-5 → displays as 0.0000
Iteration 10000: LR = 5e-5  → displays as 0.0001 ← First non-zero!
```

**Wandb charts will show the ACTUAL values:**
- Not truncated to 4 decimals like console
- You'll see the warmup ramp clearly
- Even tiny LR values will appear as a curve

### The Model IS Learning

**Evidence at iteration 30 (with LR = 1.5e-7):**
- Loss: 18.44 → 18.40 ✅
- KoLeo: 8.77 → 8.69 ✅
- Gradients: 890 (healthy) ✅
- Parameters updating ✅

Very slow updates during warmup is **intentional** for stability!

---

## 🔍 Current Status

### Training Process
```
PID: 45055 (running)
Phase: S3 discovery (scanning S3 batches)
Config: Batch size 80, log_interval 1 ✅
Expected: ~10 min for discovery
```

### Wandb Configuration
```
✅ Enabled: true
✅ Online mode: active
✅ Log interval: 1 (every iteration)
✅ Attention interval: 100 (every 100 iterations)
✅ Project: dinocell-ssl-pretraining
```

### What Will Happen Next
1. **~10 minutes:** S3 discovery completes (finding ~19,584 fields)
2. **Iteration 0:** First metrics logged to wandb
3. **Iteration 1-99:** Losses logged every iteration (charts appear!)
4. **Iteration 100:** First attention maps appear
5. **Continuous:** Loss charts update in real-time

---

## 📞 Wandb URLs

### Current Run
Will appear at iteration 0 (after S3 discovery):
```
https://wandb.ai/saahilsanganeria666-georgia-institute-of-technology/dinocell-ssl-pretraining
```

Look for run: **vits8-jump-multiview-s3**

### Previous Test Run (Completed Successfully)
https://wandb.ai/saahilsanganeria666-georgia-institute-of-technology/dinocell-test/runs/z4jlmg05

This one has metrics logged every 5 iterations - you can see what the charts will look like!

---

## 🎯 Summary

### Configuration: ✅ PERFECT

```yaml
log_interval: 1              # ✅ Losses every iteration → smooth charts
attention_log_interval: 100  # ✅ Images every 100 iter → low storage
```

### Expected Wandb Storage

**Metrics (every iteration):**
- 90,000 iterations × ~100 bytes = ~9MB
- **Cost:** Negligible

**Attention maps (every 100 iterations):**
- 900 images × ~500KB = ~450MB
- **Cost:** Reasonable

**Total:** ~460MB for entire 90k iteration run

### Next Steps
1. Wait ~10 minutes for S3 discovery
2. Training starts at iteration 0
3. Check wandb dashboard - you'll see:
   - Loss charts updating every iteration
   - Attention maps appearing at 100, 200, 300...
4. Model learns with visible progress!

---

**Everything is configured exactly as you want! Loss charts will be smooth and continuous, attention maps will appear periodically without bloating storage.** 🎯

