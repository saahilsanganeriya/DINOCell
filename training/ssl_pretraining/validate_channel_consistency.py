"""
Validate Channel Consistency of Pretrained Models
==================================================

This script validates that multi-view pretrained models have learned
channel-invariant representations by testing feature similarity across channels.

Usage:
    python validate_channel_consistency.py \\
        --model-multiview ../checkpoints/dinov3_vits8_jump_multiview.pth \\
        --model-averaging ../checkpoints/dinov3_vits8_jump_avg.pth \\
        --test-images /path/to/test/fields
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'dinov3'))

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import argparse
import logging
from tqdm import tqdm
from dinov3.hub.backbones import dinov3_vits16

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dinov3_model(checkpoint_path, patch_size=8):
    """Load DINOv3 model from checkpoint."""
    # Create model with correct patch size
    model = dinov3_vits16(pretrained=False, patch_size=patch_size)
    
    # Load weights
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'teacher' in state_dict:
        state_dict = state_dict['teacher']
    elif 'model' in state_dict:
        state_dict = state_dict['model']
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model


def extract_features(model, image, device='cuda'):
    """Extract features from image."""
    model = model.to(device)
    image = image.to(device)
    
    with torch.no_grad():
        output = model.forward_features(image)
        features = output['x_norm_clstoken']  # CLS token features
    
    return features.cpu()


def load_jump_field_channels(field_path):
    """
    Load all channels for a JUMP field.
    
    Returns dict: {'ch1': img, 'ch2': img, ..., 'avg': img}
    """
    field_path = Path(field_path)
    
    channels = {}
    channel_arrays = []
    
    # Load each channel
    for i in range(1, 6):  # 5 fluorescent channels
        ch_file = field_path / f'r01c01f01p01-ch{i}sk1fk1fl1.tiff'
        
        if ch_file.exists():
            img = cv2.imread(str(ch_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # CLAHE preprocessing
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                img = clahe.apply(img)
                channels[f'ch{i}'] = img
                channel_arrays.append(img)
    
    # Create averaged image
    if len(channel_arrays) > 0:
        channels['avg'] = np.mean(channel_arrays, axis=0).astype(np.uint8)
    
    return channels


def preprocess_for_dinov3(img):
    """Convert grayscale to RGB and normalize for DINOv3."""
    # To RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # To tensor and normalize
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor.unsqueeze(0)


def compute_channel_consistency(model, channels, device='cuda'):
    """
    Compute feature similarity across channels.
    
    Returns dict of similarities between all channel pairs.
    """
    # Extract features for each channel
    features = {}
    for ch_name, img in channels.items():
        img_tensor = preprocess_for_dinov3(img)
        feat = extract_features(model, img_tensor, device)
        features[ch_name] = feat
    
    # Compute all pairwise similarities
    similarities = {}
    channel_names = list(features.keys())
    
    for i, ch1 in enumerate(channel_names):
        for ch2 in channel_names[i+1:]:
            feat1 = features[ch1]
            feat2 = features[ch2]
            
            sim = F.cosine_similarity(feat1, feat2, dim=-1).item()
            similarities[f'{ch1}_vs_{ch2}'] = sim
    
    return similarities, features


def main():
    parser = argparse.ArgumentParser(description='Validate channel consistency')
    parser.add_argument('--model-multiview', required=True, help='Multi-view pretrained model')
    parser.add_argument('--model-averaging', default=None, help='Averaging pretrained model (for comparison)')
    parser.add_argument('--test-images', required=True, help='Path to test JUMP fields')
    parser.add_argument('--num-fields', type=int, default=100, help='Number of fields to test')
    parser.add_argument('--device', default='cuda', help='Device')
    
    args = parser.parse_args()
    
    logger.info("="*70)
    logger.info("Channel Consistency Validation")
    logger.info("="*70)
    
    # Load models
    logger.info(f"\nLoading multi-view model from {args.model_multiview}...")
    model_mv = load_dinov3_model(args.model_multiview, patch_size=8)
    logger.info("✅ Multi-view model loaded")
    
    model_avg = None
    if args.model_averaging:
        logger.info(f"\nLoading averaging model from {args.model_averaging}...")
        model_avg = load_dinov3_model(args.model_averaging, patch_size=8)
        logger.info("✅ Averaging model loaded")
    
    # Find test fields
    test_path = Path(args.test_images)
    field_dirs = list(test_path.glob('*/Images'))[:args.num_fields]
    
    logger.info(f"\nFound {len(field_dirs)} fields to test")
    
    # Test each field
    results_mv = []
    results_avg = []
    
    for field_dir in tqdm(field_dirs, desc="Testing fields"):
        try:
            # Load channels
            channels = load_jump_field_channels(field_dir)
            
            if len(channels) < 3:  # Need at least a few channels
                continue
            
            # Test multi-view model
            sims_mv, _ = compute_channel_consistency(model_mv, channels, args.device)
            results_mv.append(sims_mv)
            
            # Test averaging model if provided
            if model_avg:
                sims_avg, _ = compute_channel_consistency(model_avg, channels, args.device)
                results_avg.append(sims_avg)
                
        except Exception as e:
            logger.warning(f"Error processing field: {e}")
            continue
    
    # Aggregate results
    logger.info("\n" + "="*70)
    logger.info("Results: Channel Consistency")
    logger.info("="*70)
    
    if len(results_mv) > 0:
        # Average across all test fields
        all_keys = results_mv[0].keys()
        
        logger.info("\n Multi-View Model:")
        for key in sorted(all_keys):
            values = [r[key] for r in results_mv if key in r]
            mean_sim = np.mean(values)
            std_sim = np.std(values)
            logger.info(f"  {key}: {mean_sim:.3f} ± {std_sim:.3f}")
        
        # Overall average
        all_sims_mv = [v for r in results_mv for v in r.values()]
        logger.info(f"\n  Overall Average: {np.mean(all_sims_mv):.3f} ± {np.std(all_sims_mv):.3f}")
    
    if len(results_avg) > 0:
        logger.info("\n Averaging Model (Comparison):")
        for key in sorted(all_keys):
            values = [r[key] for r in results_avg if key in r]
            mean_sim = np.mean(values)
            std_sim = np.std(values)
            logger.info(f"  {key}: {mean_sim:.3f} ± {std_sim:.3f}")
        
        all_sims_avg = [v for r in results_avg for v in r.values()]
        logger.info(f"\n  Overall Average: {np.mean(all_sims_avg):.3f} ± {np.std(all_sims_avg):.3f}")
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("Interpretation:")
    logger.info("="*70)
    logger.info("High similarity (>0.85) = Good channel consistency")
    logger.info("Multi-view should have HIGHER similarity than averaging")
    logger.info("Especially for: ch1_vs_ch2, ch1_vs_ch5, ch2_vs_ch5")
    logger.info("="*70)
    
    # Save results
    import pandas as pd
    df_mv = pd.DataFrame(results_mv)
    df_mv.to_csv('channel_consistency_multiview.csv', index=False)
    logger.info(f"\n✅ Saved multi-view results to channel_consistency_multiview.csv")
    
    if len(results_avg) > 0:
        df_avg = pd.DataFrame(results_avg)
        df_avg.to_csv('channel_consistency_averaging.csv', index=False)
        logger.info(f"✅ Saved averaging results to channel_consistency_averaging.csv")


if __name__ == '__main__':
    main()

