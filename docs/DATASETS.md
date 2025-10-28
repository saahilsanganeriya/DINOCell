# DINOCell Datasets Guide

This document provides detailed information about datasets for training and evaluating DINOCell.

## Overview

DINOCell uses the same datasets as SAMCell for fair comparison:
- **Training**: LIVECell, Cellpose Cytoplasm
- **Zero-shot Evaluation**: PBL-HEK, PBL-N2a

## Dataset Download

### 1. LIVECell Dataset

**Description**: Large-scale phase-contrast microscopy dataset

**Statistics**:
- 5,239 total images
- 8 cell types (A172, BT474, BV2, Huh7, MCF7, SHSY5Y, SkBr3, SKOV3)
- ~1.7 million annotated cells
- Image size: 704×520 pixels
- Format: COCO JSON annotations

**Download**:
```bash
# Option 1: Manual download
# Visit: https://sartorius-research.github.io/LIVECell/
# Download: LIVECell_dataset_2021.zip (~5GB)

# Option 2: Using wget
wget http://livecell-dataset.s3.eu-central-1.amazonaws.com/LIVECell_dataset_2021.zip
unzip LIVECell_dataset_2021.zip
```

**Directory Structure**:
```
LIVECell_dataset_2021/
├── annotations/
│   └── LIVECell/
│       ├── livecell_coco_train.json
│       ├── livecell_coco_val.json
│       └── livecell_coco_test.json
└── images/
    ├── livecell_train_val_images/
    └── livecell_test_images/
```

### 2. Cellpose Cytoplasm Dataset

**Description**: Diverse microscopy images from internet

**Statistics**:
- ~600 images
- Various cell types and imaging modalities
- Mixed bright-field and fluorescent microscopy
- Variable image sizes
- Format: Numbered image and mask pairs

**Download**:
```bash
# Download from Cellpose website
# Visit: https://www.cellpose.org/dataset
# Or use their dataset download script
```

**Expected Structure**:
```
cellpose_train/
├── 000_img.png
├── 000_masks.png
├── 001_img.png
├── 001_masks.png
...
```

### 3. PBL-HEK Dataset (Evaluation Only)

**Description**: Phase-contrast images of HEK293 cells

**Statistics**:
- 5 images
- ~300 cells per image
- Novel microscope (different from training)
- Densely packed, irregular morphology
- Image size: Variable

**Download**:
```bash
wget https://github.com/saahilsanganeriya/SAMCell/releases/download/v1/PBL_HEK.zip
unzip PBL_HEK.zip
```

### 4. PBL-N2a Dataset (Evaluation Only)

**Description**: Phase-contrast images of Neuro-2a cells

**Statistics**:
- 5 images
- ~300 cells per image
- Novel cell line and microscope
- More circular morphology
- Image size: Variable

**Download**:
```bash
wget https://github.com/saahilsanganeriya/SAMCell/releases/download/v1/PBL_N2A.zip
unzip PBL_N2A.zip
```

## Dataset Processing

After downloading, process datasets for DINOCell training:

### Process LIVECell

```bash
cd dataset_processing

# Full training set
python process_dataset.py livecell \
    --input /path/to/LIVECell_dataset_2021 \
    --output ../datasets/LIVECell-train \
    --split train

# Validation set
python process_dataset.py livecell \
    --input /path/to/LIVECell_dataset_2021 \
    --output ../datasets/LIVECell-val \
    --split val

# Test set
python process_dataset.py livecell \
    --input /path/to/LIVECell_dataset_2021 \
    --output ../datasets/LIVECell-test \
    --split test
```

### Process Cellpose

```bash
python process_dataset.py cellpose \
    --input /path/to/cellpose_train \
    --output ../datasets/Cellpose-train \
    --target-size 512
```

### Process Evaluation Datasets

```bash
# PBL-HEK
python process_dataset.py custom \
    --images /path/to/PBL_HEK/images \
    --masks /path/to/PBL_HEK/ground_truth_masks \
    --output ../datasets/PBL_HEK \
    --target-size 512

# PBL-N2a
python process_dataset.py custom \
    --images /path/to/PBL_N2A/images \
    --masks /path/to/PBL_N2A/ground_truth_masks \
    --output ../datasets/PBL_N2A \
    --target-size 512
```

## Output Format

Processed datasets contain:
```
dataset_name/
├── imgs.npy         # (N, H, W, 3) uint8 - preprocessed images
├── dist_maps.npy    # (N, H, W) float32 - distance maps [0,1]
└── anns.npy         # (N, H, W) int16 - original annotations
```

## Custom Datasets

To use your own datasets:

### Format 1: Separate Image and Mask Folders

```
my_dataset/
├── images/
│   ├── img001.png
│   ├── img002.png
│   └── ...
└── masks/
    ├── img001_mask.png
    ├── img002_mask.png
    └── ...
```

Process:
```bash
python process_dataset.py custom \
    --images /path/to/my_dataset/images \
    --masks /path/to/my_dataset/masks \
    --output ../datasets/MyDataset \
    --image-ext .png \
    --mask-suffix _mask \
    --target-size 512
```

### Format 2: COCO Format

If your dataset uses COCO format, modify `process_dataset.py` or convert to Format 1.

## Unlabeled Data for Pretraining (Optional)

For self-supervised pretraining, collect unlabeled cell images:

### Recommended Sources:

1. **Broad Bioimage Benchmark Collection (BBBC)**
   - URL: https://bbbc.broadinstitute.org/
   - Thousands of annotated and unannotated cell images
   - Various cell types and imaging modalities

2. **Cell Image Library**
   - URL: http://www.cellimagelibrary.org/
   - Large collection of cellular images
   - Multiple microscopy techniques

3. **Your Own Data**
   - Collect unlabeled microscopy images from your lab
   - No annotations needed
   - More images = better pretraining

### Processing for Pretraining:

```bash
# Simply organize images in a folder
unlabeled_cells/
├── cell_img_0001.png
├── cell_img_0002.png
└── ...

# No preprocessing needed - handled by pretraining script
```

## Dataset Statistics

### LIVECell

| Split | Images | Cells | Avg Cells/Image |
|-------|--------|-------|-----------------|
| Train | 3,447 | ~1.2M | ~348 |
| Val | 576 | ~200K | ~347 |
| Test | 1,216 | ~420K | ~345 |

### Cellpose Cytoplasm

| Split | Images | Cells | Avg Cells/Image |
|-------|--------|-------|-----------------|
| Train | ~500 | ~50K | ~100 |

### PBL Datasets

| Dataset | Images | Cells | Avg Cells/Image | Characteristics |
|---------|--------|-------|-----------------|-----------------|
| PBL-HEK | 5 | ~1,500 | ~300 | Dense, irregular |
| PBL-N2a | 5 | ~1,500 | ~300 | Circular, distinct |

## Data Augmentation

During training, DINOCell applies:
- Random horizontal flip (p=0.5)
- Random rotation (-180° to 180°)
- Random scale (0.8-1.2×)
- Random brightness (0.95-1.05×)
- Random inversion (p=0.5)

These augmentations help the model generalize to:
- Different imaging orientations
- Various cell sizes
- Different contrast levels
- Different microscopy techniques (bright-field vs dark-field)

## Evaluation Metrics

DINOCell uses Cell Tracking Challenge metrics:

- **SEG**: Segmentation accuracy (Jaccard index)
- **DET**: Detection accuracy (AOGM-D based)
- **OP_CSB**: Overall performance (average of SEG and DET)

All metrics range from 0 to 1, higher is better.

## References

1. LIVECell: Edlund et al., Nature Methods 2021
2. Cellpose: Stringer et al., Nature Methods 2021  
3. Cell Tracking Challenge: Maška et al., Bioinformatics 2014
4. SAMCell: VandeLoo et al., PLOS ONE 2025



