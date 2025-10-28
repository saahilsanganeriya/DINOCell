# DINOCell Setup Guide

## Quick Start

### Using Conda (Recommended)

```bash
# Create environment from environment.yml
conda env create -f environment.yml

# Activate environment
conda activate dinocell

# Install dinov3 package
cd dinov3_modified/dinov3
pip install -e .
cd ../..

# Install DINOCell package
pip install -e .
```

### Using pip + venv

```bash
# Requires Python 3.11+
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install dinov3 package
cd dinov3_modified/dinov3
pip install -e .
cd ../..

# Install DINOCell package
pip install -e .
```

## Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from dinov3.data.datasets import JUMPCellPainting; print('✓ DINOv3 datasets loaded')"
python -c "import wandb; print('✓ Wandb installed')"
```

## GPU Requirements

- **For SSL Pretraining**: NVIDIA GPU with 40GB+ VRAM (A100 recommended)
- **For Fine-tuning**: NVIDIA GPU with 16GB+ VRAM
- **CUDA**: 11.8 or later

## Storage Requirements

- **With S3 Streaming**: ~10GB (code + cache)
- **Local JUMP Dataset**: ~510GB
- **Checkpoints**: ~5-10GB per training run

## Next Steps

See [START_HERE.md](START_HERE.md) for usage instructions.

