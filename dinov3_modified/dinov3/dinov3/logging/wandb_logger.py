# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

"""
Weights & Biases Logging for DINOv3
====================================

Comprehensive logging of training metrics, attention maps, and visualizations.

Features:
- Training metrics (losses, learning rate, etc.)
- Attention map visualizations
- Feature PCA visualizations
- Gradient statistics
- Checkpoint uploading
"""

import logging
from typing import Dict, Any, Optional
import torch
import numpy as np

logger = logging.getLogger("dinov3")

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not available. Install with: pip install wandb")


class WandbLogger:
    """Weights & Biases logger for DINOv3 training."""
    
    def __init__(self, cfg, enabled=True):
        """
        Initialize wandb logger.
        
        Parameters
        ----------
        cfg : OmegaConf
            Configuration object
        enabled : bool
            Whether wandb logging is enabled
        """
        self.enabled = enabled and WANDB_AVAILABLE
        
        if not self.enabled:
            logger.info("Wandb logging disabled")
            return
        
        # Initialize wandb
        wandb_config = cfg.get('wandb', {})
        
        self.project = wandb_config.get('project', 'dinov3-training')
        self.name = wandb_config.get('name', None)
        self.entity = wandb_config.get('entity', None)
        self.log_interval = wandb_config.get('log_interval', 100)
        self.log_attention = wandb_config.get('log_attention_maps', False)
        self.attention_interval = wandb_config.get('attention_log_interval', 1000)
        
        # Initialize run
        wandb.init(
            project=self.project,
            name=self.name,
            entity=self.entity,
            config=dict(cfg),
            resume='allow'  # Allow resuming
        )
        
        logger.info(f"Wandb logging enabled: {wandb.run.url}")
    
    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """Log training metrics."""
        if not self.enabled:
            return
        
        if step % self.log_interval == 0:
            wandb.log(metrics, step=step)
    
    def log_attention_maps(self, model, images, step: int):
        """
        Log attention map visualizations.
        
        Parameters
        ----------
        model : nn.Module
            Model with attention layers
        images : torch.Tensor
            Input images (B, 3, H, W)
        step : int
            Training step
        """
        if not self.enabled or not self.log_attention:
            return
        
        if step % self.attention_interval != 0:
            return
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            
            # Unwrap FSDP if needed
            actual_model = model
            if hasattr(model, '_fsdp_wrapped_module'):
                actual_model = model._fsdp_wrapped_module
            elif hasattr(model, 'module'):
                actual_model = model.module
            
            # Store attention weights using hook
            attention_weights = []
            def hook_fn(module, input, output):
                # Capture attention weights from the last block
                if hasattr(module, 'attn_drop') or 'attn' in str(type(module)).lower():
                    if isinstance(output, tuple) and len(output) > 1:
                        attention_weights.append(output[1])  # Attention weights
            
            # Register hook on last attention block
            hooks = []
            if hasattr(actual_model, 'blocks') and len(actual_model.blocks) > 0:
                last_block = actual_model.blocks[-1]
                if hasattr(last_block, 'attn'):
                    hook = last_block.attn.register_forward_hook(hook_fn)
                    hooks.append(hook)
            
            model.eval()
            with torch.no_grad():
                # Forward pass to trigger hook
                _ = actual_model.forward_features(images[:4])
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                
                # If we didn't capture attention, just visualize input images
                if not attention_weights:
                    logger.info("Logging input images (attention extraction not available)")
                    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                    for idx in range(min(4, images.shape[0])):
                        img = images[idx].cpu().numpy().transpose(1, 2, 0)
                        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                        axes[idx].imshow(img)
                        axes[idx].set_title(f'Input {idx}')
                        axes[idx].axis('off')
                    plt.tight_layout()
                    wandb.log({"input_images": wandb.Image(fig)}, step=step)
                    plt.close()
                    model.train()
                    return
                
                # Use captured attention
                outputs = attention_weights[0]
                
                # Visualize
                fig, axes = plt.subplots(2, 4, figsize=(16, 8))
                
                for idx in range(min(4, images.shape[0])):
                    # Original image
                    img = images[idx].cpu().numpy().transpose(1, 2, 0)
                    img = (img - img.min()) / (img.max() - img.min())
                    axes[0, idx].imshow(img)
                    axes[0, idx].set_title(f'Image {idx}')
                    axes[0, idx].axis('off')
                    
                    # Attention map (average across heads)
                    attn = outputs[idx].mean(0).cpu().numpy()  # Average heads
                    attn_map = attn[0, 1:]  # CLS token attention to patches
                    h = w = int(np.sqrt(len(attn_map)))
                    attn_map = attn_map.reshape(h, w)
                    
                    axes[1, idx].imshow(attn_map, cmap='jet')
                    axes[1, idx].set_title(f'Attention {idx}')
                    axes[1, idx].axis('off')
                
                plt.tight_layout()
                
                # Log to wandb
                wandb.log({"attention_maps": wandb.Image(fig)}, step=step)
                plt.close()
                
            model.train()
            
        except Exception as e:
            logger.warning(f"Failed to log attention maps: {e}")
    
    def log_feature_pca(self, features, labels, step: int):
        """Log PCA visualization of features."""
        if not self.enabled:
            return
        
        try:
            from sklearn.decomposition import PCA
            import matplotlib.pyplot as plt
            
            # Compute PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features.cpu().numpy())
            
            # Plot
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                                c=labels, cmap='tab20', alpha=0.6)
            plt.colorbar(scatter)
            plt.title(f'Feature PCA at step {step}')
            plt.xlabel('PC1')
            plt.ylabel('PC2')
            
            wandb.log({"feature_pca": wandb.Image(plt)}, step=step)
            plt.close()
            
        except Exception as e:
            logger.warning(f"Failed to log PCA: {e}")
    
    def log_histogram(self, name: str, values: torch.Tensor, step: int):
        """Log histogram of values."""
        if not self.enabled:
            return
        
        wandb.log({f"{name}_histogram": wandb.Histogram(values.cpu().numpy())}, step=step)
    
    def log_gradients(self, model, step: int):
        """Log gradient statistics."""
        if not self.enabled:
            return
        
        if step % (self.log_interval * 10) != 0:  # Less frequent
            return
        
        grad_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_dict[f"gradients/{name}_mean"] = param.grad.mean().item()
                grad_dict[f"gradients/{name}_std"] = param.grad.std().item()
                grad_dict[f"gradients/{name}_max"] = param.grad.max().item()
        
        wandb.log(grad_dict, step=step)
    
    def log_model_checkpoint(self, checkpoint_path: str, step: int):
        """Log model checkpoint to wandb (optional - large files)."""
        if not self.enabled:
            return
        
        # Create artifact
        artifact = wandb.Artifact(
            name=f"model-checkpoint-{step}",
            type="model",
            description=f"Model checkpoint at step {step}"
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
    
    def finish(self):
        """Finish wandb run."""
        if self.enabled:
            wandb.finish()


def create_wandb_logger(cfg):
    """Factory function to create wandb logger from config."""
    enabled = cfg.get('wandb', {}).get('enabled', False)
    return WandbLogger(cfg, enabled=enabled)

