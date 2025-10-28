# Multi-View Consistency: Quick Start Guide

## 🎯 What You're Doing

Training DINOv3 to learn that **different channels show the same cells**.

**Simple Version** (averaging): Merge channels → train  
**Advanced Version** (multi-view): Keep channels separate → teach model they're the same

**Benefit**: Model learns "channel-invariant" features → works on ANY channel!

---

## ⚡ 30-Second Start

```bash
cd DINOCell/training
./launch_pretraining_multiview.sh
```

**Wait 30-40 hours. Done!**

---

## 📋 Pre-Flight Checklist

```bash
# 1. DINOv3 installed?
ls ../../dinov3  # Should show dinov3 repo

# 2. JUMP dataset accessible?
ls ../../2024_Chandrasekaran_NatureMethods_CPJUMP1  # Should show batch folders

# 3. GPU available?
nvidia-smi  # Should show A100 with 80GB

# 4. Checkpoint downloaded?
ls ../../dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

# If missing:
cd ../../dinov3/checkpoints
wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

All ✅? **Launch training!**

---

## 🚀 Launch Command

```bash
cd DINOCell/training
./launch_pretraining_multiview.sh
```

**What happens**:
1. Loads pretrained ViT-S/16
2. Adapts to patch-8
3. Trains with multi-view consistency
4. Saves every 2.5 hours
5. Completes in 30-40 hours

---

## 📊 Monitor Training

```bash
# Watch logs in real-time
tail -f ../../DINOCell/checkpoints/dinov3_vits8_jump_multiview/logs/log.txt

# Check GPU
watch -n 1 nvidia-smi

# List checkpoints
ls ../../DINOCell/checkpoints/dinov3_vits8_jump_multiview/ckpt/
```

**Look for**:
- ✅ Losses decreasing
- ✅ No NaN values
- ✅ Regular checkpoints (every ~2.5 hrs)

---

## 🎯 After Training (5 min)

```bash
# Extract final checkpoint
cp ../../DINOCell/checkpoints/dinov3_vits8_jump_multiview/eval/final/teacher_checkpoint.pth \
   ../checkpoints/dinov3_vits8_jump_multiview.pth

# Validate channel consistency
python validate_channel_consistency.py \
    --model-multiview ../checkpoints/dinov3_vits8_jump_multiview.pth \
    --test-images ../../2024_Chandrasekaran_NatureMethods_CPJUMP1/2020_11_04_CPJUMP1/images/BR00117010*

# Fine-tune DINOCell
python train.py \
    --dataset ../datasets/LIVECell-train \
    --backbone-weights ../checkpoints/dinov3_vits8_jump_multiview.pth \
    --model-size small --freeze-backbone --epochs 100
```

---

## 💡 Key Differences from Averaging

| What | Averaging | Multi-View |
|------|-----------|------------|
| **Input** | Single averaged image | List of 5 channel images |
| **Global Crop 1** | crop(averaged) | crop(averaged) ✅ Same |
| **Global Crop 2** | crop(averaged) | crop(random_channel) ⭐ Different! |
| **DINO Loss** | Spatial consistency | **Channel consistency** ⭐ |
| **Result** | General features | Channel-invariant features ⭐ |

---

## 🎁 What You Get

**After 30-40 hours**:
- ✅ DINOv3 backbone trained on 3M cell images
- ✅ Patch-8 resolution (4x higher than standard)
- ✅ **Channel-invariant features** (key innovation!)
- ✅ Ready for any JUMP modality

**Use for**:
- Segment cells in any channel
- Cross-channel retrieval
- Missing channel robustness
- Unified cell representation

---

## 🎊 Summary

**One command to rule them all**:
```bash
./launch_pretraining_multiview.sh
```

**Why multi-view**:
- Explicitly learns channel invariance
- Better than averaging
- Novel research contribution
- Production-grade results

**Timeline**:
- 30-40 hours training
- Auto-saves, auto-resumes
- Then fine-tune DINOCell (3-4 hrs)

**Ready to start? Run the command! 🚀**

