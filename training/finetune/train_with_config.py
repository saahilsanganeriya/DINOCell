"""
DINOCell Training with YAML Config
===================================

Train DINOCell using a YAML configuration file.

Usage:
    python train_with_config.py --config ../configs/training_config.yaml
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import argparse
import yaml
import logging
from train import main as train_main

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def config_to_args(config):
    """Convert config dict to argument namespace."""
    class Args:
        pass
    
    args = Args()
    
    # Model config
    args.model_size = config['model']['size']
    args.freeze_backbone = config['model']['freeze_backbone']
    args.pretrained = config['model']['pretrained']
    args.backbone_weights = config['model'].get('backbone_weights')
    
    # Data config
    args.dataset = config['data']['datasets']
    args.val_split = config['data']['val_split']
    args.num_workers = config['data']['num_workers']
    
    # Training config
    args.epochs = config['training']['epochs']
    args.batch_size = config['training']['batch_size']
    args.lr = config['training']['learning_rate']
    args.weight_decay = config['training']['weight_decay']
    args.warmup_steps = config['training']['warmup_steps']
    args.patience = config['training']['patience']
    args.min_delta = config['training']['min_delta']
    args.save_every = config['training']['save_every']
    args.output = config['training']['output_dir']
    
    # Other config
    args.device = config.get('device', 'cuda')
    args.seed = config.get('seed', 42)
    
    return args


def main():
    parser = argparse.ArgumentParser(description='Train DINOCell with config file')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    
    args = parser.parse_args()
    
    # Load config
    logger.info(f'Loading config from {args.config}')
    config = load_config(args.config)
    
    # Convert to args
    train_args = config_to_args(config)
    
    # Update sys.argv for train_main
    sys.argv = ['train.py']
    
    logger.info('Starting training with configuration:')
    logger.info(yaml.dump(config, default_flow_style=False))
    
    # Import and modify train.py's main to accept args object
    # For now, just call it (you may need to modify train.py to accept args object)
    import train
    
    # Override argparse in train module
    original_parse_args = argparse.ArgumentParser.parse_args
    
    def mock_parse_args(self, args=None, namespace=None):
        return train_args
    
    argparse.ArgumentParser.parse_args = mock_parse_args
    
    try:
        train.main()
    finally:
        argparse.ArgumentParser.parse_args = original_parse_args


if __name__ == '__main__':
    main()



