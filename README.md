# DINOCell

DINOCell is an unfinished experiment in label-free cell segmentation: outlining every cell in a
microscopy image without anyone clicking on the cells. It keeps the method from
[SAMCell](https://doi.org/10.1371/journal.pone.0319532) (PLOS ONE, 2025): predict a distance map
that is high at each cell's centre and falls to zero at its edge, then split touching cells with a
watershed. It swaps SAMCell's Segment Anything backbone for Meta's [DINOv3](https://github.com/facebookresearch/dinov3),
and sets up DINOv3's self-supervised training to continue on JUMP Cell Painting microscopy images.

Saahil Sanganeriya built it alone in October 2025, after co-authoring SAMCell. DINOv3's code is
Meta's and is included under its own licence (see [Credits and licences](#credits-and-licences)).

## Status

| Part | State | Evidence |
|---|---|---|
| Self-supervised pretraining on JUMP (loaders, channel views, S3 reading, W&B logging) | Written; one run was started, and no finished run is recorded | The last status note in the history records iteration ~220 of 90,000 on 2025-10-28: `git show 8e09a3f:ALL_SYSTEMS_GO.md` |
| Segmentation model in `dinocell/` | Does not run: the decoder fails its first forward pass | `UpConvBlock` in `dinocell/model.py` concatenates the skip connection, doubling the channels, but its first convolution expects the un-doubled count (`RuntimeError: ... expected input[1, 256, 28, 28] to have 128 channels, but got 256`) |
| Fine-tuning, evaluation, CLI | Do not run, since they build the model above | same |
| Results | None | No checkpoint and no segmentation score is published, so this repository makes no accuracy claim |

## How it works

```
microscope image (grayscale)
   │  CLAHE contrast enhancement
   ▼
256×256 tiles, 32 px overlap
   │
   ▼
DINOv3 ViT backbone ──► features from 4 intermediate layers
   │
   ▼
U-Net style decoder ──► distance map per tile
   │  tiles blended back together
   ▼
watershed: seeds where the map is above 0.47, cell pixels where it is above 0.09
   │
   ▼
one integer label per cell
```

The backbone can be Meta's released weights or weights continued on microscopy images. For that
second step, each JUMP field of view is imaged in five fluorescent channels that show the same
cells. Pretraining builds DINO's views from different channels (one view averages the channels,
another uses a single random channel), so the DINO loss pushes the backbone toward the same features
whichever channel it sees. [docs/SSL_PRETRAINING.md](docs/SSL_PRETRAINING.md) has the details.

## Repository layout

```
dinocell/              model (DINOv3 backbone + U-Net decoder), inference pipeline, CLI
training/finetune/     fine-tuning script for the segmentation model
training/ssl_pretraining/
                       launch scripts and configs for pretraining on JUMP
evaluation/            Cell Tracking Challenge metrics (SEG, DET, OP_CSB) and threshold search
dataset_processing/    converts LIVECell, Cellpose and custom datasets to the training format
examples/              inference and SAMCell comparison scripts
dinov3_modified/       Meta's DINOv3 with the JUMP additions (DINOv3 License)
docs/                  architecture, datasets, pretraining, S3 reading, fine-tuning
```

## Setup

Python 3.11.

```bash
git clone https://github.com/saahilsanganeriya/DINOCell.git
cd DINOCell
conda env create -f environment.yml && conda activate dinocell
# or, in any Python 3.11 environment:
pip install -r requirements.txt && pip install -e .
```

DINOv3 weights are not in this repository. Request access through the
[DINOv3 repository](https://github.com/facebookresearch/dinov3); Meta e-mails the download URLs.

## Pretraining on JUMP Cell Painting

Put `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` in a `dinov3/checkpoints/` folder next to this
clone (the path in `student.pretrained_weights` of the config), then:

```bash
cd training/ssl_pretraining
./launch_ssl_with_s3_wandb.sh              # reads images from the public cellpainting-gallery S3 bucket
./launch_ssl_with_s3_wandb.sh --no-wandb   # without Weights & Biases
```

No AWS account is needed. The script tries to activate a conda environment named `dinov3`; if
there is none, it carries on in the active environment. The configuration and the other options are
in [docs/SSL_PRETRAINING.md](docs/SSL_PRETRAINING.md).

## Segmentation, fine-tuning and evaluation

These are the intended entry points. They stop at the decoder error described under [Status](#status).

```python
import cv2
import numpy as np
from dinocell import create_dinocell_model, DINOCellPipeline

image = cv2.imread("cells.png", cv2.IMREAD_GRAYSCALE)
model = create_dinocell_model("small", freeze_backbone=True, pretrained=False)
model.load_weights("checkpoints/dinocell/best_model.pt")  # a fine-tuned DINOCell checkpoint
labels = DINOCellPipeline(model, device="cuda").run(image)
print(len(np.unique(labels)) - 1, "cells")
```

```bash
dinocell segment cells.png --model checkpoints/dinocell/best_model.pt --output results/

cd training/finetune
python train.py --dataset ../../datasets/LIVECell-train --model-size small --freeze-backbone \
    --output ../../checkpoints/dinocell

cd ../../evaluation
python evaluate.py --model ../checkpoints/dinocell/best_model.pt --dataset ../datasets/PBL_HEK
```

[docs/DATASETS.md](docs/DATASETS.md) covers getting and converting the datasets;
[docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md) covers the training options.

## Credits and licences

- **This project's code** (everything outside `dinov3_modified/`): MIT, see [LICENSE](LICENSE).
- **`dinov3_modified/`**: a copy of Meta's DINOv3 with additions for this project. The whole folder,
  including the additions, is under the DINOv3 License in
  [`dinov3_modified/dinov3/LICENSE.md`](dinov3_modified/dinov3/LICENSE.md), not MIT.
  [`dinov3_modified/MODIFICATIONS.md`](dinov3_modified/MODIFICATIONS.md) lists what was added and changed.
- **SAMCell**: the distance-map method and the default watershed thresholds come from SAMCell, and
  `dinocell/slidingWindow.py` is adapted from SAMCell's code (MIT). Its licence notice is in [NOTICE](NOTICE).
- **Data**: the JUMP Cell Painting images come from the public `cellpainting-gallery` bucket on AWS.

## Citation

SAMCell: Generalized label-free biological cell segmentation with segment anything. PLOS ONE 20(9): e0319532 (2025).
https://doi.org/10.1371/journal.pone.0319532

```bibtex
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Siméoni, Oriane and Vo, Huy V. and others},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
}
```
