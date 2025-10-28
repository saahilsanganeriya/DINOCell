"""
DINOCell Training Script
========================

Main training script for fine-tuning DINOCell on cell segmentation datasets.

Based on SAMCell's training approach:
- MSE loss on distance maps
- AdamW optimizer with warmup
- Early stopping
- Supports multiple datasets (LIVECell, Cellpose, custom)

Usage:
    python train.py --dataset ../datasets/LIVECell-train --model-size small --epochs 100
    python train.py --dataset ../datasets/Cellpose-train ../datasets/LIVECell-train --model-size base --freeze-backbone
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
import os

from dinocell.model import create_dinocell_model
from dinocell.dataset import DINOCellDataset


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def lr_warmup(step, warmup_steps=250):
    """Learning rate warmup schedule."""
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    return max(0.0, 1.0 - (step - warmup_steps) / (step - warmup_steps + 1))


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=7, min_delta=0.0001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
    
    def __call__(self, val_loss, epoch):
        score = -val_loss if self.mode == 'min' else val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        
        return self.early_stop


def train_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """Train for one epoch."""
    model.train()
    epoch_losses = []
    sigmoid = nn.Sigmoid()
    mse_loss = nn.MSELoss()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        # Get data
        pixel_values = batch['pixel_values'].to(device)
        ground_truth = batch['ground_truth_dist_map'].to(device)
        
        # Forward pass
        predicted = model(pixel_values)  # (B, 1, 256, 256)
        predicted = predicted.squeeze(1)  # (B, 256, 256)
        
        # Apply sigmoid
        predicted = sigmoid(predicted)
        
        # Compute loss
        loss = mse_loss(predicted, ground_truth)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Track loss
        epoch_losses.append(loss.item())
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return np.mean(epoch_losses)


def validate(model, dataloader, device):
    """Validate model on validation set."""
    model.eval()
    val_losses = []
    sigmoid = nn.Sigmoid()
    mse_loss = nn.MSELoss()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating'):
            pixel_values = batch['pixel_values'].to(device)
            ground_truth = batch['ground_truth_dist_map'].to(device)
            
            # Forward pass
            predicted = model(pixel_values)
            predicted = predicted.squeeze(1)
            predicted = sigmoid(predicted)
            
            # Compute loss
            loss = mse_loss(predicted, ground_truth)
            val_losses.append(loss.item())
    
    return np.mean(val_losses)


def main():
    parser = argparse.ArgumentParser(description='Train DINOCell model')
    
    # Data arguments
    parser.add_argument('--dataset', nargs='+', required=True,
                       help='Path(s) to dataset folder(s) containing imgs.npy and dist_maps.npy')
    parser.add_argument('--val-split', type=float, default=0.1,
                       help='Validation split ratio (default: 0.1)')
    
    # Model arguments
    parser.add_argument('--model-size', choices=['small', 'base', 'large', '7b'], default='small',
                       help='DINOv3 model size (default: small)')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze DINOv3 backbone (only train decoder)')
    parser.add_argument('--backbone-weights', type=str, default=None,
                       help='Path to custom DINOv3 backbone weights')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained DINOv3 weights (default: True)')
    parser.add_argument('--no-pretrained', dest='pretrained', action='store_false',
                       help='Train from random initialization')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum number of epochs (default: 100)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight-decay', type=float, default=0.1,
                       help='Weight decay (default: 0.1)')
    parser.add_argument('--warmup-steps', type=int, default=250,
                       help='Learning rate warmup steps (default: 250)')
    parser.add_argument('--patience', type=int, default=7,
                       help='Early stopping patience (default: 7)')
    parser.add_argument('--min-delta', type=float, default=0.0001,
                       help='Minimum improvement for early stopping (default: 0.0001)')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='../checkpoints/dinocell',
                       help='Output directory for checkpoints')
    parser.add_argument('--save-every', type=int, default=1,
                       help='Save checkpoint every N epochs (default: 1)')
    
    # Other arguments
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers (default: 4)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f'Using device: {device}')
    
    # Load datasets
    logger.info('Loading datasets...')
    datasets = []
    for dataset_path in args.dataset:
        dataset_path = Path(dataset_path)
        imgs_path = dataset_path / 'imgs.npy'
        dist_maps_path = dataset_path / 'dist_maps.npy'
        
        if not imgs_path.exists() or not dist_maps_path.exists():
            logger.error(f'Dataset not found at {dataset_path}')
            continue
        
        dataset = DINOCellDataset(str(imgs_path), str(dist_maps_path), augment=True)
        datasets.append(dataset)
        logger.info(f'Loaded dataset from {dataset_path}: {len(dataset)} samples')
    
    # Combine datasets if multiple
    if len(datasets) == 0:
        logger.error('No valid datasets found!')
        return
    elif len(datasets) == 1:
        full_dataset = datasets[0]
    else:
        logger.info(f'Combining {len(datasets)} datasets...')
        full_dataset = ConcatDataset(datasets)
    
    # Split into train/val
    dataset_size = len(full_dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    logger.info(f'Train set: {len(train_dataset)} samples')
    logger.info(f'Validation set: {len(val_dataset)} samples')
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    logger.info(f'Creating DINOCell model (size: {args.model_size})...')
    model = create_dinocell_model(
        model_size=args.model_size,
        freeze_backbone=args.freeze_backbone,
        pretrained=args.pretrained,
        backbone_weights=args.backbone_weights
    )
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')
    
    # Setup optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, 
                     betas=(0.9, 0.999))
    
    # Warmup + linear decay
    def lr_lambda(step):
        return lr_warmup(step, args.warmup_steps)
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config_file = output_dir / 'config.txt'
    with open(config_file, 'w') as f:
        f.write(f'Training Configuration\n')
        f.write(f'=====================\n')
        f.write(f'Timestamp: {datetime.now()}\n\n')
        for arg, value in vars(args).items():
            f.write(f'{arg}: {value}\n')
    
    logger.info(f'Configuration saved to {config_file}')
    
    # Training loop
    logger.info('Starting training...')
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        logger.info(f'\n=== Epoch {epoch+1}/{args.epochs} ===')
        
        # Train
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler, device, epoch+1)
        logger.info(f'Train loss: {train_loss:.6f}')
        
        # Validate
        val_loss = validate(model, val_dataloader, device)
        logger.info(f'Validation loss: {val_loss:.6f}')
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = output_dir / 'best_model.pt'
            model.save_weights(str(best_checkpoint_path))
            logger.info(f'✓ Saved best model to {best_checkpoint_path}')
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pt'
            model.save_weights(str(checkpoint_path))
            logger.info(f'✓ Saved checkpoint to {checkpoint_path}')
        
        # Early stopping check
        if early_stopping(val_loss, epoch):
            logger.info(f'\nEarly stopping triggered at epoch {epoch+1}')
            logger.info(f'Best epoch was {early_stopping.best_epoch+1} with val loss {-early_stopping.best_score:.6f}')
            break
    
    # Save final model
    final_checkpoint_path = output_dir / 'final_model.pt'
    model.save_weights(str(final_checkpoint_path))
    logger.info(f'\n✓ Training complete! Final model saved to {final_checkpoint_path}')
    logger.info(f'✓ Best model saved to {output_dir / "best_model.pt"}')


if __name__ == '__main__':
    main()



