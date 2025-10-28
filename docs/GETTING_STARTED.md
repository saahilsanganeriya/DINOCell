# Getting Started with DINOCell

Complete guide to get DINOCell up and running.

## 📋 Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended: A100, V100, or RTX 3090+)
- 16GB+ RAM
- Linux or macOS (Windows may work but untested)

## 🚀 Installation

### Step 1: Clone Repository

```bash
git clone <your-repo-url>
cd DINOCell
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install DINOCell
pip install -r requirements.txt
pip install -e .
```

### Step 4: Verify Installation

```bash
python -c "
from dinocell import create_dinocell_model, DINOCellPipeline
print('✓ DINOCell installed successfully!')
"
```

## 🎯 Choose Your Path

### Path 1: Just Inference (Quickest) ⚡

Use pretrained models for cell segmentation:

```python
import cv2
import numpy as np
from dinocell import create_dinocell_model, DINOCellPipeline

# Load image
image = cv2.imread('path/to/cells.png', cv2.IMREAD_GRAYSCALE)

# Create model (downloads pretrained weights)
model = create_dinocell_model('small', pretrained=True)

# Create pipeline
pipeline = DINOCellPipeline(model, device='cuda')

# Segment!
labels = pipeline.run(image)

print(f"Found {len(np.unique(labels))-1} cells!")
```

### Path 2: Fine-tune on Your Data (Recommended) 🎓

Train DINOCell on your own cell dataset:

1. **Prepare dataset**: [See Dataset Guide](DATASET_PREPARATION.md)
2. **Train model**: 
   ```bash
   cd training/finetune
   python train.py --dataset ../../datasets/MyDataset-train --model-size small
   ```
3. **Evaluate**: 
   ```bash
   cd ../../evaluation
   python evaluate.py --model ../training/finetune/checkpoints/best.pt
   ```

### Path 3: SSL Pretraining from Scratch (Advanced) 🔬

Pretrain DINOv3 on millions of unlabeled cell images:

1. **Setup AWS** (optional): For S3 streaming
2. **Configure wandb**: For monitoring
3. **Launch SSL**:
   ```bash
   cd training/ssl_pretraining
   ./launch_ssl_with_s3_wandb.sh
   ```
4. **Wait**: 30-40 hours on A100
5. **Use checkpoint**: Fine-tune with your SSL-pretrained backbone

## 📂 Download Pretrained Models

### DINOCell Models

```bash
# Download DINOCell-Small (generalist model)
wget <url-to-dinocell-small> -O checkpoints/dinocell_small_generalist.pt

# Download DINOCell-Base (better performance)
wget <url-to-dinocell-base> -O checkpoints/dinocell_base_generalist.pt
```

### Pretrained DINOv3 Backbones

DINOCell automatically downloads DINOv3 backbones from torch.hub on first use.

To manually download:
```bash
python -c "
import torch
# This downloads to ~/.cache/torch/hub/checkpoints/
model = torch.hub.load('facebookresearch/dinov3', 'dinov3_vits16', pretrained=True)
print('✓ DINOv3 ViT-Small downloaded')
"
```

## 📊 Download Datasets

### LIVECell (Required for training)

```bash
# Create datasets directory
mkdir -p datasets
cd datasets

# Download LIVECell (~5GB)
wget http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021.zip
unzip LIVECell_dataset_2021.zip

# Process for DINOCell
cd ../dataset_processing
python process_dataset.py livecell \
    --input ../datasets/LIVECell_dataset_2021 \
    --output ../datasets/LIVECell-train \
    --split train

python process_dataset.py livecell \
    --input ../datasets/LIVECell_dataset_2021 \
    --output ../datasets/LIVECell-val \
    --split val
```

### Cellpose (Optional, for generalist model)

```bash
# Download Cellpose cytoplasm dataset
# Visit: https://www.cellpose.org/dataset
# Download and extract to datasets/Cellpose/

# Process for DINOCell
cd dataset_processing
python process_dataset.py cellpose \
    --input ../datasets/Cellpose/train \
    --output ../datasets/Cellpose-train \
    --target-size 512
```

### JUMP Cell Painting (Optional, for SSL pretraining)

**Option 1: S3 Streaming** (Recommended)
```bash
# No download needed! Just install boto3
pip install boto3 smart-open
```

**Option 2: Local Download** (If you have 500GB+ storage)
```bash
# Install AWS CLI
pip install awscli

# Download JUMP dataset
aws s3 sync s3://cellpainting-gallery/cpg0000-jump-pilot/source_4/images/ \
    datasets/JUMP/ \
    --no-sign-request
```

## 🧪 Test Your Setup

Run comprehensive tests:

```bash
# Test 1: Basic imports
python -c "
from dinocell import create_dinocell_model, DINOCellPipeline
from dinocell.preprocessing import apply_clahe
print('✓ Imports work')
"

# Test 2: Model creation
python -c "
from dinocell import create_dinocell_model
model = create_dinocell_model('small', pretrained=True)
print(f'✓ Model created: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters')
"

# Test 3: GPU availability
python -c "
import torch
print(f'✓ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
"
```

If all tests pass, you're ready to go!

## 📖 Next Steps

### For Cell Segmentation Users

1. **Read**: [Quick Start Tutorial](examples/simple_inference.py)
2. **Try**: Segment your first image
3. **Optimize**: Tune thresholds for your cell type
4. **Compare**: Run comparison with SAMCell ([guide](examples/compare_with_samcell.py))

### For Researchers Fine-tuning

1. **Read**: [Training Guide](TRAINING_GUIDE.md)
2. **Prepare**: Process your dataset
3. **Train**: Fine-tune on your data (2-4 hours)
4. **Evaluate**: Cell Tracking Challenge metrics

### For Advanced SSL Pretraining

1. **Read**: [SSL Pretraining Guide](SSL_PRETRAINING.md)
2. **Setup**: Configure S3 and wandb
3. **Launch**: Start 30-40 hour pretraining run
4. **Monitor**: Track progress with wandb
5. **Validate**: Test channel consistency

## 🐛 Common Issues

### Issue: "ModuleNotFoundError: No module named 'dinocell'"

**Solution**: Install in editable mode:
```bash
cd DINOCell
pip install -e .
```

### Issue: "DINOv3 repository not found"

**Solution**: Verify structure:
```bash
ls dinov3_modified/dinov3/
# Should show: dinov3/, README.md, etc.
```

If missing, reinitialize:
```bash
cd dinov3_modified
git clone https://github.com/facebookresearch/dinov3.git dinov3
```

### Issue: "CUDA out of memory"

**Solutions**:
- Reduce batch size: `--batch-size 4` (default: 8)
- Use smaller model: `--model-size small` instead of `base`
- Reduce image size during preprocessing
- Use gradient checkpointing (automatic in SSL pretraining)

### Issue: "Can't download DINOv3 weights"

**Solution**: Download manually:
```bash
wget <dinov3-checkpoint-url> -O ~/.cache/torch/hub/checkpoints/dinov3_vits16.pth
```

### Issue: "S3 connection failed"

**Solutions**:
1. Check internet: `ping s3.amazonaws.com`
2. Install dependencies: `pip install boto3 smart-open`
3. Test access: See [S3 Streaming Guide](S3_STREAMING.md)
4. Fallback to local: Use `--no-s3` flag

## 💡 Tips

### 1. Start Small

Begin with a small test dataset:
```bash
# Use just 100 images for quick testing
python dataset_processing/process_dataset.py livecell \
    --input datasets/LIVECell_dataset_2021 \
    --output datasets/LIVECell-test \
    --split train \
    --max-images 100
```

### 2. Use Pretrained Weights

Always start from pretrained DINOv3:
```python
model = create_dinocell_model('small', pretrained=True)  # ✓ Much faster convergence
# vs.
model = create_dinocell_model('small', pretrained=False)  # ✗ Slow, poor results
```

### 3. Monitor Training

Use wandb or check loss files:
```bash
tail -f training/finetune/checkpoints/training.log
```

### 4. Validate Frequently

Run evaluation every few epochs:
```bash
python evaluation/evaluate.py --model checkpoints/epoch_10.pt --dataset datasets/MyDataset-val
```

## 🎓 Learning Resources

### Tutorials
- [Simple Inference Example](../examples/simple_inference.py) - Segment your first image (5 min)
- [Compare with SAMCell](../examples/compare_with_samcell.py) - Side-by-side comparison (10 min)

### Documentation
- [Architecture Guide](ARCHITECTURE.md) - Understanding DINOCell (15 min)
- [Training Guide](TRAINING_GUIDE.md) - Comprehensive training guide (30 min)
- [SSL Pretraining](SSL_PRETRAINING.md) - Advanced self-supervised learning (1 hour)

### Code
- `dinocell/model.py` - Model architecture
- `dinocell/pipeline.py` - Inference pipeline
- `training/finetune/train.py` - Training loop

## 🤝 Getting Help

1. **Check docs**: Most questions answered in `docs/`
2. **Run examples**: `examples/` directory has working code
3. **Read code**: Well-commented, easy to follow
4. **Open issue**: Create GitHub issue if stuck

## ✅ Quick Validation Checklist

Before starting serious work, verify:

- [ ] Python 3.8+ installed
- [ ] CUDA available and working
- [ ] DINOCell imports work
- [ ] Can load pretrained model
- [ ] Can run inference on test image
- [ ] (Optional) Wandb configured
- [ ] (Optional) S3 access working

## 🎉 You're Ready!

You should now be able to:
- ✅ Run cell segmentation inference
- ✅ Fine-tune on your datasets
- ✅ (Advanced) Perform SSL pretraining

Choose your path above and start experimenting!

## 📚 Additional Resources

- **SAMCell (comparison)**: `../SAMCell/`
- **JUMP Dataset**: https://jump-cellpainting.broadinstitute.org/
- **DINOv3 Paper**: https://arxiv.org/abs/2508.10104
- **Cell Tracking Challenge**: http://celltrackingchallenge.net/

Happy segmenting! 🔬🧬
