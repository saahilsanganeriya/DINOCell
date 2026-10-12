# Self-supervised pretraining on JUMP Cell Painting

This page explains how DINOCell continues DINOv3's self-supervised training on microscopy images
from the JUMP Cell Painting pilot (`cpg0000-jump-pilot`), and how to start a run.

No run in this repository reached the end of its schedule, so it has no measured loss curves,
checkpoints or channel-consistency numbers.

## What the run does

It starts from Meta's DINOv3 ViT-S/16 weights, switches the student to patch size 8 for finer
detail on small cells, and trains with the DINO, iBOT and KoLeo losses that DINOv3 already uses.
All of the training code is DINOv3's; the files this project added or changed are listed in
[`dinov3_modified/MODIFICATIONS.md`](../dinov3_modified/MODIFICATIONS.md).

### Multi-view consistency across channels

Each JUMP field of view is imaged in several fluorescent channels that show the same cells.
Instead of only averaging the channels into one image, the multi-channel augmentation
(`dinov3/data/augmentations_multichannel.py`) builds the DINO views from different channels:

```
one field of view (5 fluorescent channels)
        │
        ├── global crop 1: average of the channels
        ├── global crop 2: one channel picked at random
        └── local crops:   channels picked at random
                │
                ▼
DINO loss pulls the features of these views together,
so the backbone is pushed toward the same features whichever channel it sees
```

`use_multichannel_augmentation: true` in the config turns this on
(`dinov3/train/ssl_meta_arch.py` picks the augmentation).

### Reading images from S3

With the S3 config, images are read on demand from the public `cellpainting-gallery` bucket with
unsigned requests, so no AWS account or download is needed. An LRU cache keeps the 1000 most recent
images in RAM. Details: [S3_STREAMING.md](S3_STREAMING.md).

### Weights & Biases logging

`dinov3/train/train.py` creates a `WandbLogger` when `wandb.enabled` is true. It logs the training
losses, learning rate, weight decay and teacher temperature every `log_interval` iterations (1 in the
S3 config) and attention maps every `attention_log_interval` iterations (100).

## Before you start

1. Install the dependencies from the repository root: `pip install -r requirements.txt`.
2. Get DINOv3 weights from Meta. Access is by request through the
   [DINOv3 repository](https://github.com/facebookresearch/dinov3); Meta e-mails the download URLs.
3. Put `dinov3_vits16_pretrain_lvd1689m-08c60483.pth` where the config looks for it.
   `student.pretrained_weights` is `../../../dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth`,
   resolved from `dinov3_modified/dinov3/` (the training working directory), so that is a
   `dinov3/checkpoints/` folder next to your DINOCell clone. Edit the config to use another path.
4. Optional: `wandb login`.

## Start a run

Run the launch script from its own folder; its paths are relative to it.

```bash
cd training/ssl_pretraining
./launch_ssl_with_s3_wandb.sh                    # S3 + multi-view + W&B
./launch_ssl_with_s3_wandb.sh --no-wandb         # without W&B
./launch_ssl_with_s3_wandb.sh --no-s3            # local files (configs/dinov3_vits8_jump_multiview.yaml)
./launch_ssl_with_s3_wandb.sh --wandb-project my-project --wandb-name my-run
```

The script checks the S3 and W&B dependencies, tests that the bucket can be listed, activates a
conda or micromamba environment named `dinov3` if one of them is installed, and runs
`dinov3/train/train.py`. Output goes to `training/ssl_pretraining/output/`. If
`dinov3_modified/dinov3/checkpoints/` has no weights, the script first tries Meta's public weights URL;
without access that request is refused, and training still reads the file named in the config. The script's own estimate
for the full schedule is 30-40 hours on one A100; that is an estimate, not a measured time.

For local files, `dataset_path` in `configs/dinov3_vits8_jump_multiview.yaml` must point at a local
copy of the CPJUMP1 images.

## Configuration

Main values in `configs/dinov3_vits8_jump_s3_multiview.yaml`:

```yaml
student:
  arch: vit_small
  patch_size: 8
  pretrained_weights: '../../../dinov3/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth'

train:
  batch_size_per_gpu: 80
  dataset_path: JUMPS3MultiView:bucket=cellpainting-gallery:prefix=cpg0000-jump-pilot/source_4/images

optim:
  epochs: 90
  lr: 5.0e-5          # one tenth of DINOv3's rate, for continued training
  warmup_epochs: 10

wandb:
  enabled: true
  log_interval: 1
  attention_log_interval: 100
```

If the GPU runs out of memory, lower `train.batch_size_per_gpu`.

## After pretraining

Check how similar the features are across channels:

```bash
python validate_channel_consistency.py \
    --model-multiview output/eval/training_<iteration>/teacher_checkpoint.pth \
    --test-images /path/to/jump/test/fields
```

Add `--model-averaging <checkpoint>` to compare against a channel-averaging run.

The teacher checkpoint is meant to become the DINOCell backbone (`training/finetune/train.py
--backbone-weights <checkpoint>`). That fine-tuning step does not run yet; see the
[repository README](../README.md#status).

## Compared with DINOv3's own pretraining

| | DINOv3 release | This project |
|---|---|---|
| Data | LVD-1689M, 1.7B web images | JUMP Cell Painting pilot (microscopy) |
| Start | from scratch | continues from the ViT-S/16 release weights |
| Patch size | 16 | 8 |
| Views | colour image augmentations | channel views (above) |
| Data source | local files | public S3 bucket or local files |

## Citation

Cite DINOv3 for the backbone and training code:

```bibtex
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Siméoni, Oriane and Vo, Huy V. and others},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
}
```
