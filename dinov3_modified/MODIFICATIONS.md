# DINOv3 modifications for DINOCell

`dinov3_modified/dinov3/` is a copy of Meta's [DINOv3 repository](https://github.com/facebookresearch/dinov3),
added to this repository on 2025-10-28 with the changes below for training on JUMP Cell Painting
microscopy images. The upstream commit was not recorded; the copied `LICENSE.md` is dated August 19, 2025.

## Licence

Everything in this folder, including the files added for DINOCell, is distributed under the
DINOv3 License in [`dinov3/LICENSE.md`](dinov3/LICENSE.md), not under the MIT licence at the
repository root. The added files carry the same licence header as Meta's files.

## Added files

| File (under `dinov3/dinov3/`) | Purpose |
|---|---|
| `data/datasets/jump_cellpainting.py` | JUMP loader that averages the fluorescent channels into one image |
| `data/datasets/jump_cellpainting_multiview.py` | JUMP loader that returns each channel separately, for multi-view training |
| `data/datasets/jump_cellpainting_s3.py` | JUMP loader that reads images from the public `cellpainting-gallery` S3 bucket, with an LRU cache |
| `data/datasets/jump_simple_local.py` | Small local loader used for test runs |
| `data/augmentations_multichannel.py` | DINO augmentation that builds global and local views from different channels |
| `logging/wandb_logger.py` | Weights & Biases logger (losses, learning rate, attention maps) |

## Changed files

| File (under `dinov3/dinov3/`) | Change |
|---|---|
| `data/datasets/__init__.py` | Imports the JUMP datasets (the S3 one only if `boto3` is installed) |
| `data/loaders.py` | Accepts `JUMPCellPainting`, `JUMPCellPaintingMultiView`, `JUMPSimpleLocal`, `JUMPS3` and `JUMPS3MultiView` in `dataset_path` |
| `train/ssl_meta_arch.py` | Uses the multi-channel augmentation when `train.use_multichannel_augmentation` is true |
| `train/train.py` | Creates the W&B logger when `wandb.enabled` is true and logs metrics and attention maps |

Everything else is Meta's code as copied.

## Updating to a newer DINOv3

Copy the six added files into a fresh DINOv3 checkout, then reapply the four changes above by hand;
upstream may have moved the code they touch.
