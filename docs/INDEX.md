# DINOCell Complete File Index

Quick reference guide to all files in the DINOCell project.

## 📖 Documentation Files

| File | Purpose | Audience |
|------|---------|----------|
| `README.md` | Main project documentation | Everyone |
| `QUICKSTART.md` | 5-minute getting started guide | New users |
| `GETTING_STARTED.md` | Detailed setup instructions | New users |
| `TRAINING_GUIDE.md` | Comprehensive training guide | Researchers |
| `DATASETS.md` | Dataset information and download | Data scientists |
| `ARCHITECTURE.md` | Technical architecture details | Engineers |
| `PROJECT_SUMMARY.md` | Complete project overview | Stakeholders |
| `CHANGELOG.md` | Version history and updates | Contributors |
| `INDEX.md` | This file - complete file listing | Everyone |

## 🔬 Core Source Code (`src/dinocell/`)

| File | Purpose | Key Classes/Functions |
|------|---------|----------------------|
| `__init__.py` | Package initialization | Exports main API |
| `model.py` | Model architecture | `DINOCell`, `create_dinocell_model` |
| `pipeline.py` | Inference pipeline | `DINOCellPipeline` |
| `preprocessing.py` | Image preprocessing | `apply_clahe`, `preprocess_for_dinov3` |
| `dataset.py` | Dataset classes | `DINOCellDataset` |
| `slidingWindow.py` | Sliding window helper | `SlidingWindowHelper` |
| `cli.py` | Command-line interface | `main()` |

## 🎓 Training Scripts (`training/`)

| File | Purpose | Usage |
|------|---------|-------|
| `train.py` | Main training script | `python train.py --dataset ... --model-size small` |
| `train_with_config.py` | Config-based training | `python train_with_config.py --config config.yaml` |
| `pretrain_dinov3.py` | Pretraining guide | `python pretrain_dinov3.py --help` |

## 📊 Evaluation Scripts (`evaluation/`)

| File | Purpose | Usage |
|------|---------|-------|
| `evaluate.py` | Main evaluation script | `python evaluate.py --model ... --dataset ...` |
| `evaluation_utils.py` | CTC metric utilities | Imported by evaluate.py |

## 🗂️ Dataset Processing (`dataset_processing/`)

| File | Purpose | Usage |
|------|---------|-------|
| `process_dataset.py` | Dataset converter | `python process_dataset.py livecell --input ... --output ...` |
| `dataset_utils.py` | Processing utilities | Distance map creation, resizing |

## 📚 Examples (`examples/`)

| File | Purpose | Complexity |
|------|---------|-----------|
| `simple_inference.py` | Basic usage example | Beginner |
| `compare_with_samcell.py` | DINOCell vs SAMCell | Intermediate |

## ⚙️ Configuration Files

| File | Purpose |
|------|---------|
| `configs/training_config.yaml` | Training configuration template |
| `setup.py` | Package installation configuration |
| `requirements.txt` | Python dependencies |
| `.gitignore` | Git ignore rules |
| `LICENSE` | MIT license |

## 📓 Interactive Tutorials

| File | Purpose |
|------|---------|
| `Tutorial.ipynb` | Interactive Jupyter tutorial |

## 🗂️ Directories

| Directory | Contents | Created By |
|-----------|----------|------------|
| `checkpoints/` | Trained model weights | Training scripts |
| `datasets/` | Processed datasets | Dataset processing |
| `results/` | Inference results | Inference scripts |
| `eval_*/` | Temporary evaluation files | Evaluation scripts |

## 📝 File Relationships

### Training Workflow
```
process_dataset.py → imgs.npy + dist_maps.npy
                          ↓
                    train.py → best_model.pt
                          ↓
                   evaluate.py → results.csv
```

### Inference Workflow
```
your_image.png → cli.py / pipeline.py → labels.tif
                                      → visualization.png
```

### Dependencies
```
model.py
  ├─ requires: DINOv3 (torch.hub)
  └─ imports: preprocessing.py

pipeline.py
  ├─ imports: model.py
  ├─ imports: slidingWindow.py
  └─ imports: preprocessing.py

train.py
  ├─ imports: model.py
  └─ imports: dataset.py

evaluate.py
  ├─ imports: model.py
  ├─ imports: pipeline.py
  └─ imports: evaluation_utils.py
```

## 🔍 Finding What You Need

**"I want to..."**

- **...understand the project**: Start with `README.md`
- **...get started quickly**: Read `QUICKSTART.md`
- **...set up from scratch**: Follow `GETTING_STARTED.md`
- **...train a model**: Check `TRAINING_GUIDE.md`
- **...understand the architecture**: Read `ARCHITECTURE.md`
- **...use the API**: See `README.md` API Reference section
- **...process my dataset**: Use `dataset_processing/process_dataset.py`
- **...run inference**: Use `src/dinocell/cli.py` or `examples/simple_inference.py`
- **...evaluate performance**: Use `evaluation/evaluate.py`
- **...compare with SAMCell**: Run `examples/compare_with_samcell.py`
- **...modify the model**: Edit `src/dinocell/model.py`
- **...change preprocessing**: Edit `src/dinocell/preprocessing.py`
- **...adjust post-processing**: Modify `src/dinocell/pipeline.py` (watershed section)

## 📦 Package Structure

```
dinocell/                  # Python package
├── model.py              # Core model
├── pipeline.py           # Inference
├── preprocessing.py      # Image processing
├── dataset.py            # Data loading
├── slidingWindow.py      # Sliding window
└── cli.py                # Command line
```

**Import examples**:
```python
from dinocell import DINOCell, DINOCellPipeline
from dinocell.model import create_dinocell_model
from dinocell.preprocessing import apply_clahe
from dinocell.dataset import DINOCellDataset
```

## 🔧 Configuration Files

**Training Config** (`configs/training_config.yaml`):
```yaml
model:
  size: small
  freeze_backbone: true
  
training:
  epochs: 100
  batch_size: 8
  learning_rate: 1.0e-4
```

**Usage**:
```bash
python training/train_with_config.py --config configs/training_config.yaml
```

## 📊 Output Files

**After Training**:
```
checkpoints/dinocell/
├── best_model.pt         # Best model by validation loss
├── final_model.pt        # Model after last epoch
├── checkpoint_epoch_N.pt # Periodic checkpoints
└── config.txt            # Training configuration
```

**After Evaluation**:
```
evaluation_results.csv    # SEG/DET/CSB metrics
```

**After Inference**:
```
results/
├── image_labels.tif      # Segmentation masks
├── image_distmap.png     # Distance map
└── image_visualization.png  # Combined visualization
```

## 🆘 Quick Help

```bash
# Training help
python training/train.py --help

# Evaluation help
python evaluation/evaluate.py --help

# Dataset processing help
python dataset_processing/process_dataset.py --help

# CLI help
python -m dinocell.cli --help
```

## 📧 Support

For questions or issues:
1. Check documentation (README.md, GETTING_STARTED.md)
2. Review examples (examples/ folder)
3. Read guides (TRAINING_GUIDE.md, ARCHITECTURE.md)
4. Open GitHub issue

---

**Quick Links**:
- [Main README](README.md)
- [Getting Started](GETTING_STARTED.md)
- [Training Guide](TRAINING_GUIDE.md)
- [Architecture](ARCHITECTURE.md)



