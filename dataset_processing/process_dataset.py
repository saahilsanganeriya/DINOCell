"""
DINOCell Dataset Processing Script
===================================

Convert various dataset formats to DINOCell training format.
Compatible with SAMCell datasets (LIVECell, Cellpose, custom formats).

Output format:
- imgs.npy: Preprocessed images (N, H, W, 3) uint8, BGR format
- dist_maps.npy: Distance maps (N, H, W) float32, normalized [0, 1]
- anns.npy: Original annotations (N, H, W) int16 (for evaluation)

Usage:
    # LIVECell dataset
    python process_dataset.py livecell --input /path/to/LIVECell_dataset_2021 \\
                              --output ../datasets/LIVECell-train --split train
    
    # Cellpose dataset
    python process_dataset.py cellpose --input /path/to/cellpose/train \\
                              --output ../datasets/Cellpose-train --target-size 512
    
    # Custom dataset
    python process_dataset.py custom --images /path/to/images --masks /path/to/masks \\
                              --output ../datasets/Custom --target-size 512
"""

import sys
from pathlib import Path

# Add parent src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import numpy as np
import cv2
from tqdm import tqdm
import logging

from dinocell.preprocessing import preprocess_training_image
from dataset_utils import create_distance_map, resize_with_padding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_livecell(base_folder, split, output_dir):
    """Process LIVECell dataset (COCO format)."""
    try:
        from pycocotools.coco import COCO
    except ImportError:
        raise ImportError("pycocotools required. Install with: pip install pycocotools")
    
    base_folder = Path(base_folder)
    output_dir = Path(output_dir)
    
    # Map splits to annotation files
    split_map = {
        'train': 'annotations/LIVECell/livecell_coco_train.json',
        'val': 'annotations/LIVECell/livecell_coco_val.json',
        'test': 'annotations/LIVECell/livecell_coco_test.json',
    }
    
    if split not in split_map:
        raise ValueError(f"Unknown split: {split}. Options: {list(split_map.keys())}")
    
    ann_file = base_folder / split_map[split]
    img_folder = base_folder / 'images' / ('livecell_test_images' if split == 'test' else 'livecell_train_val_images')
    
    logger.info(f'Processing LIVECell {split} split')
    logger.info(f'Annotation file: {ann_file}')
    logger.info(f'Image folder: {img_folder}')
    
    # Load COCO
    coco = COCO(str(ann_file))
    img_ids = coco.getImgIds()
    imgs_meta = coco.loadImgs(img_ids)
    
    logger.info(f'Found {len(imgs_meta)} images')
    
    imgs = []
    dist_maps = []
    anns = []
    
    for img_meta in tqdm(imgs_meta, desc='Processing'):
        # Load image
        img_path = img_folder / img_meta['file_name']
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning(f'Could not load {img_path}')
            continue
        
        # Get annotations
        ann_ids = coco.getAnnIds(imgIds=img_meta['id'])
        anns_data = coco.loadAnns(ann_ids)
        
        # Create mask
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.int16)
        
        cell_id = 1
        for ann in anns_data:
            if ann.get('iscrowd', 0) != 0:
                continue
            cell_mask = coco.annToMask(ann)
            mask[cell_mask == 1] = cell_id
            cell_id += 1
        
        # Preprocess image
        img_processed = preprocess_training_image(img)
        
        # Create distance map
        dist_map = create_distance_map(mask)
        
        imgs.append(img_processed)
        dist_maps.append(dist_map)
        anns.append(mask)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'imgs.npy', np.array(imgs, dtype=np.uint8))
    np.save(output_dir / 'dist_maps.npy', np.array(dist_maps, dtype=np.float32))
    np.save(output_dir / 'anns.npy', np.array(anns, dtype=np.int16))
    
    logger.info(f'✓ Saved {len(imgs)} samples to {output_dir}')
    return imgs, dist_maps, anns


def process_cellpose(input_folder, output_dir, target_size=512):
    """Process Cellpose dataset (numbered image/mask pairs)."""
    input_folder = Path(input_folder)
    output_dir = Path(output_dir)
    
    logger.info(f'Processing Cellpose dataset from {input_folder}')
    logger.info(f'Target size: {target_size}x{target_size}')
    
    imgs = []
    dist_maps = []
    anns = []
    
    # Find image/mask pairs
    i = 0
    while True:
        img_file = input_folder / f'{i:03d}_img.png'
        mask_file = input_folder / f'{i:03d}_masks.png'
        
        if not img_file.exists() or not mask_file.exists():
            break
        
        # Load
        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        
        if img is None or mask is None:
            logger.warning(f'Could not load pair {i}')
            i += 1
            continue
        
        # Resize
        img_resized = resize_with_padding(img, target_size, is_mask=False)
        mask_resized = resize_with_padding(mask, target_size, is_mask=True)
        
        # Preprocess
        img_processed = preprocess_training_image(img_resized)
        dist_map = create_distance_map(mask_resized)
        
        imgs.append(img_processed)
        dist_maps.append(dist_map)
        anns.append(mask_resized)
        
        i += 1
    
    logger.info(f'Found {len(imgs)} image-mask pairs')
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'imgs.npy', np.array(imgs, dtype=np.uint8))
    np.save(output_dir / 'dist_maps.npy', np.array(dist_maps, dtype=np.float32))
    np.save(output_dir / 'anns.npy', np.array(anns, dtype=np.int16))
    
    logger.info(f'✓ Saved {len(imgs)} samples to {output_dir}')
    return imgs, dist_maps, anns


def process_custom(image_folder, mask_folder, output_dir, image_ext='.png', 
                   mask_suffix='_mask', target_size=512):
    """Process custom dataset with separate image and mask folders."""
    image_folder = Path(image_folder)
    mask_folder = Path(mask_folder)
    output_dir = Path(output_dir)
    
    logger.info(f'Processing custom dataset')
    logger.info(f'Images: {image_folder}')
    logger.info(f'Masks: {mask_folder}')
    
    # Find images
    img_files = sorted(list(image_folder.glob(f'*{image_ext}')))
    logger.info(f'Found {len(img_files)} images')
    
    imgs = []
    dist_maps = []
    anns = []
    
    for img_file in tqdm(img_files, desc='Processing'):
        # Find mask
        mask_file = mask_folder / f'{img_file.stem}{mask_suffix}{image_ext}'
        
        if not mask_file.exists():
            logger.warning(f'No mask for {img_file.name}')
            continue
        
        # Load
        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
        
        if img is None or mask is None:
            logger.warning(f'Could not load {img_file.name}')
            continue
        
        # Resize
        img_resized = resize_with_padding(img, target_size, is_mask=False)
        mask_resized = resize_with_padding(mask, target_size, is_mask=True)
        
        # Preprocess
        img_processed = preprocess_training_image(img_resized)
        dist_map = create_distance_map(mask_resized)
        
        imgs.append(img_processed)
        dist_maps.append(dist_map)
        anns.append(mask_resized)
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / 'imgs.npy', np.array(imgs, dtype=np.uint8))
    np.save(output_dir / 'dist_maps.npy', np.array(dist_maps, dtype=np.float32))
    np.save(output_dir / 'anns.npy', np.array(anns, dtype=np.int16))
    
    logger.info(f'✓ Saved {len(imgs)} samples to {output_dir}')
    return imgs, dist_maps, anns


def main():
    parser = argparse.ArgumentParser(description='Process datasets for DINOCell')
    subparsers = parser.add_subparsers(dest='dataset_type', required=True)
    
    # LIVECell
    livecell_parser = subparsers.add_parser('livecell', help='Process LIVECell dataset')
    livecell_parser.add_argument('--input', required=True, help='Path to LIVECell_dataset_2021')
    livecell_parser.add_argument('--output', required=True, help='Output directory')
    livecell_parser.add_argument('--split', choices=['train', 'val', 'test'], default='train')
    
    # Cellpose
    cellpose_parser = subparsers.add_parser('cellpose', help='Process Cellpose dataset')
    cellpose_parser.add_argument('--input', required=True, help='Input folder')
    cellpose_parser.add_argument('--output', required=True, help='Output directory')
    cellpose_parser.add_argument('--target-size', type=int, default=512)
    
    # Custom
    custom_parser = subparsers.add_parser('custom', help='Process custom dataset')
    custom_parser.add_argument('--images', required=True, help='Images folder')
    custom_parser.add_argument('--masks', required=True, help='Masks folder')
    custom_parser.add_argument('--output', required=True, help='Output directory')
    custom_parser.add_argument('--image-ext', default='.png', help='Image extension')
    custom_parser.add_argument('--mask-suffix', default='_mask', help='Mask filename suffix')
    custom_parser.add_argument('--target-size', type=int, default=512)
    
    args = parser.parse_args()
    
    logger.info('='*70)
    logger.info('DINOCell Dataset Processor')
    logger.info('='*70)
    
    if args.dataset_type == 'livecell':
        process_livecell(args.input, args.split, args.output)
    elif args.dataset_type == 'cellpose':
        process_cellpose(args.input, args.output, args.target_size)
    elif args.dataset_type == 'custom':
        process_custom(args.images, args.masks, args.output, args.image_ext, 
                      args.mask_suffix, args.target_size)
    
    logger.info('='*70)
    logger.info('✓ Processing complete!')
    logger.info('='*70)


if __name__ == '__main__':
    main()



