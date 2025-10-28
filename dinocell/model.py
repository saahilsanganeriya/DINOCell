"""
DINOCell Model Architecture
============================

DINOCell uses a DINOv3 Vision Transformer as the backbone encoder with a U-Net style
decoder for distance map prediction.

The model consists of:
1. DINOv3 backbone (frozen or fine-tunable) - extracts multi-scale features
2. U-Net decoder - fuses features and upsamples to distance map prediction
3. Distance map head - final conv layer outputting single-channel distance map

Based on SAMCell's distance map regression approach but using DINOv3 instead of SAM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """
    Convolutional block with Conv2d -> BatchNorm -> ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class UpConvBlock(nn.Module):
    """
    Upsampling block with transposed convolution.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = ConvBlock(out_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
    
    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            # Resize skip connection to match x if needed
            if x.shape != skip.shape:
                skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.conv1(x)
        x = self.conv2(x)
        return x


class DINOv3UNetDecoder(nn.Module):
    """
    U-Net style decoder that takes multi-scale DINOv3 features and produces distance maps.
    
    Architecture:
    - Takes 4 intermediate layers from DINOv3 (like depth/segmentation tasks in official repo)
    - Progressively upsamples and fuses features
    - Final 1x1 conv to produce single-channel distance map
    """
    def __init__(self, embed_dims=[384, 384, 384, 384], decoder_channels=[256, 128, 64, 32]):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.decoder_channels = decoder_channels
        
        # Lateral connections to reduce feature dimensions
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, dec_ch, kernel_size=1)
            for embed_dim, dec_ch in zip(embed_dims, decoder_channels)
        ])
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList([
            UpConvBlock(decoder_channels[i], decoder_channels[i+1])
            for i in range(len(decoder_channels)-1)
        ])
        
        # Final upsampling to full resolution (256x256)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(decoder_channels[-1], 16, kernel_size=2, stride=2),
            ConvBlock(16, 16),
            nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2),
            ConvBlock(8, 8),
        )
        
        # Distance map head
        self.dist_map_head = nn.Conv2d(8, 1, kernel_size=1)
    
    def forward(self, features):
        """
        Args:
            features: List of 4 feature maps from DINOv3, from shallow to deep
                     Each has shape (B, C, H, W)
        
        Returns:
            Distance map prediction (B, 1, 256, 256)
        """
        # Reverse features to go from deep to shallow
        features = features[::-1]
        
        # Apply lateral convolutions
        x = self.lateral_convs[0](features[0])  # Deepest features
        
        # Progressive upsampling and fusion
        for i in range(len(self.up_blocks)):
            skip = self.lateral_convs[i+1](features[i+1]) if i+1 < len(features) else None
            x = self.up_blocks[i](x, skip)
        
        # Final upsampling to 256x256
        x = self.final_up(x)
        
        # Generate distance map
        dist_map = self.dist_map_head(x)
        
        return dist_map


class DINOCell(nn.Module):
    """
    Main DINOCell model combining DINOv3 backbone with U-Net decoder.
    
    Parameters
    ----------
    backbone_name : str
        DINOv3 model name (e.g., 'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16')
    backbone_weights : str, optional
        Path or URL to pretrained DINOv3 weights
    freeze_backbone : bool
        Whether to freeze the DINOv3 backbone during training
    pretrained : bool
        Whether to load pretrained DINOv3 weights
    intermediate_layers : list
        Which intermediate layers to extract from DINOv3 (default: [2, 5, 8, 11] for ViT-S/B)
    """
    def __init__(self, 
                 backbone_name='dinov3_vits16',
                 backbone_weights=None,
                 freeze_backbone=False,
                 pretrained=True,
                 intermediate_layers=None):
        super().__init__()
        
        self.backbone_name = backbone_name
        self.freeze_backbone = freeze_backbone
        
        # Load DINOv3 backbone
        self.backbone = self._load_backbone(backbone_name, backbone_weights, pretrained)
        
        # Determine intermediate layers based on backbone architecture
        if intermediate_layers is None:
            n_blocks = self.backbone.n_blocks
            if n_blocks == 12:  # ViT-S/B
                intermediate_layers = [2, 5, 8, 11]
            elif n_blocks == 24:  # ViT-L
                intermediate_layers = [4, 11, 17, 23]
            elif n_blocks == 40:  # ViT-7B
                intermediate_layers = [9, 19, 29, 39]
            else:
                # Default: evenly spaced
                intermediate_layers = [n_blocks // 4 - 1, n_blocks // 2 - 1, 
                                      3 * n_blocks // 4 - 1, n_blocks - 1]
        
        self.intermediate_layers = intermediate_layers
        
        # Get embedding dimensions
        embed_dim = self.backbone.embed_dim
        if hasattr(self.backbone, 'embed_dims'):
            # Use per-layer embed dims if available
            embed_dims = [self.backbone.embed_dims[i] for i in intermediate_layers]
        else:
            embed_dims = [embed_dim] * len(intermediate_layers)
        
        logger.info(f"Using DINOv3 {backbone_name} with {n_blocks} blocks")
        logger.info(f"Intermediate layers: {intermediate_layers}")
        logger.info(f"Embedding dimensions: {embed_dims}")
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Backbone frozen")
        
        # Create U-Net decoder
        self.decoder = DINOv3UNetDecoder(embed_dims=embed_dims)
        
    def _load_backbone(self, backbone_name, backbone_weights, pretrained):
        """Load DINOv3 backbone from torch hub or local path."""
        try:
            # Get the dinov3_modified repository path (parent directory)
            dinov3_repo_path = Path(__file__).parent.parent / 'dinov3_modified' / 'dinov3'
            
            if not dinov3_repo_path.exists():
                raise FileNotFoundError(f"DINOv3 repository not found at {dinov3_repo_path}")
            
            logger.info(f"Loading DINOv3 from local repository: {dinov3_repo_path}")
            
            # Load backbone using torch.hub from local path
            if backbone_weights is not None:
                # Load with custom weights
                backbone = torch.hub.load(
                    str(dinov3_repo_path),
                    backbone_name,
                    source='local',
                    weights=backbone_weights,
                    pretrained=pretrained
                )
            else:
                # Load with default pretrained weights
                backbone = torch.hub.load(
                    str(dinov3_repo_path),
                    backbone_name,
                    source='local',
                    pretrained=pretrained
                )
            
            logger.info(f"Successfully loaded {backbone_name}")
            return backbone
            
        except Exception as e:
            logger.error(f"Error loading DINOv3 backbone: {e}")
            raise
    
    def forward(self, x):
        """
        Forward pass through DINOCell.
        
        Args:
            x: Input images (B, 3, H, W), should be normalized properly for DINOv3
        
        Returns:
            Distance map predictions (B, 1, 256, 256)
        """
        B, C, H, W = x.shape
        
        # Extract intermediate features from DINOv3
        # DINOv3 expects specific normalization
        with torch.set_grad_enabled(not self.freeze_backbone):
            features = self.backbone.get_intermediate_layers(
                x,
                n=self.intermediate_layers,
                reshape=True,  # Return as (B, C, H, W)
                return_class_token=False,
                norm=True
            )
        
        # Features is a tuple of tensors (B, C, H_i, W_i)
        # Decode to distance map
        dist_map = self.decoder(list(features))
        
        return dist_map
    
    def get_model(self):
        """Return self for compatibility with SAMCell interface."""
        return self
    
    def load_weights(self, weight_path, map_location=None):
        """
        Load model weights from a checkpoint file.
        
        Parameters
        ----------
        weight_path : str
            Path to the weights file (.pt or .pth)
        map_location : torch.device, optional
            Device to load weights to
        """
        if map_location is None:
            map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            state_dict = torch.load(weight_path, map_location=map_location)
            
            # Handle different checkpoint formats
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            elif 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            self.load_state_dict(state_dict)
            logger.info(f"Successfully loaded weights from {weight_path}")
            
        except Exception as e:
            logger.error(f"Error loading weights: {e}")
            raise
    
    def save_weights(self, save_path):
        """
        Save model weights to a checkpoint file.
        
        Parameters
        ----------
        save_path : str
            Path to save the weights file
        """
        try:
            save_dict = {
                'model_state_dict': self.state_dict(),
                'backbone_name': self.backbone_name,
                'intermediate_layers': self.intermediate_layers,
                'freeze_backbone': self.freeze_backbone,
            }
            torch.save(save_dict, save_path)
            logger.info(f"Successfully saved weights to {save_path}")
            
        except Exception as e:
            logger.error(f"Error saving weights: {e}")
            raise


def create_dinocell_model(model_size='small', freeze_backbone=False, pretrained=True, 
                          backbone_weights=None):
    """
    Factory function to create DINOCell models of different sizes.
    
    Parameters
    ----------
    model_size : str
        Model size: 'small', 'base', 'large', or '7b'
    freeze_backbone : bool
        Whether to freeze the backbone during training
    pretrained : bool
        Whether to use pretrained DINOv3 weights
    backbone_weights : str, optional
        Path or URL to custom backbone weights
    
    Returns
    -------
    DINOCell
        Initialized DINOCell model
    """
    model_configs = {
        'small': 'dinov3_vits16',
        'base': 'dinov3_vitb16',
        'large': 'dinov3_vitl16',
        '7b': 'dinov3_vit7b16'
    }
    
    if model_size not in model_configs:
        raise ValueError(f"Unknown model_size: {model_size}. Options: {list(model_configs.keys())}")
    
    backbone_name = model_configs[model_size]
    
    return DINOCell(
        backbone_name=backbone_name,
        backbone_weights=backbone_weights,
        freeze_backbone=freeze_backbone,
        pretrained=pretrained
    )


if __name__ == "__main__":
    # Test model creation
    print("Testing DINOCell model creation...")
    
    # Test with small model
    try:
        model = create_dinocell_model(model_size='small', freeze_backbone=True, pretrained=False)
        print(f"✓ Created DINOCell-Small")
        print(f"  Backbone blocks: {model.backbone.n_blocks}")
        print(f"  Embedding dim: {model.backbone.embed_dim}")
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        print(f"  Output shape: {output.shape}")
        assert output.shape == (2, 1, 256, 256), f"Expected (2, 1, 256, 256), got {output.shape}"
        print("✓ Forward pass successful")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

