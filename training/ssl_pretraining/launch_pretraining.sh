#!/bin/bash
# ============================================================================
# DINOv3 Pretraining Launch Script for JUMP Cell Painting
# ============================================================================
#
# This script launches DINOv3 self-supervised pretraining on JUMP dataset
# with patch size 8 for higher resolution cell features.
#
# Hardware: Single A100 80GB GPU
# Time: 24-48 hours
# Dataset: ~3M JUMP Cell Painting images
#
# Usage:
#   bash launch_pretraining.sh
#
# Or with custom parameters:
#   bash launch_pretraining.sh --epochs 100 --batch-size 32
# ============================================================================

# Configuration
DINOV3_ROOT="../../dinov3"
CONFIG_FILE="../DINOCell/training/configs/dinov3_vits8_jump_pretrain.yaml"
OUTPUT_DIR="../DINOCell/checkpoints/dinov3_vits8_jump_pretrained"
DATASET_ROOT="../../2024_Chandrasekaran_NatureMethods_CPJUMP1"

# Parse command line arguments (optional)
EXTRA_ARGS="$@"

# Print configuration
echo "========================================================================"
echo "DINOv3 Pretraining on JUMP Cell Painting"
echo "========================================================================"
echo "DINOv3 Root: $DINOV3_ROOT"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Dataset: $DATASET_ROOT"
echo "Extra args: $EXTRA_ARGS"
echo "========================================================================"
echo ""

# Check if DINOv3 exists
if [ ! -d "$DINOV3_ROOT" ]; then
    echo "ERROR: DINOv3 repository not found at $DINOV3_ROOT"
    echo "Please clone it first:"
    echo "  cd ../.. && git clone https://github.com/facebookresearch/dinov3.git"
    exit 1
fi

# Check if dataset exists
if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: JUMP dataset not found at $DATASET_ROOT"
    echo "Please ensure the dataset is downloaded"
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Activate environment (if using conda/mamba)
if command -v micromamba &> /dev/null; then
    echo "Activating dinov3 environment with micromamba..."
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate dinov3
elif command -v conda &> /dev/null; then
    echo "Activating dinov3 environment with conda..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate dinov3
fi

# Navigate to DINOv3
cd $DINOV3_ROOT

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch training
echo ""
echo "========================================================================"
echo "Starting DINOv3 pretraining..."
echo "========================================================================"
echo "This will take 24-48 hours. Monitor with:"
echo "  tail -f $OUTPUT_DIR/logs/log.txt"
echo ""
echo "To resume if interrupted:"
echo "  The script will automatically resume from last checkpoint"
echo "========================================================================"
echo ""

# Run training
PYTHONPATH=. python dinov3/train/train.py \
    --config-file "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    $EXTRA_ARGS

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ Pretraining completed successfully!"
    echo "========================================================================"
    echo "Checkpoints saved to: $OUTPUT_DIR"
    echo "Teacher checkpoint: $OUTPUT_DIR/eval/final/teacher_checkpoint.pth"
    echo ""
    echo "Next steps:"
    echo "1. Evaluate pretrained backbone"
    echo "2. Fine-tune DINOCell with these weights"
    echo "3. Compare with standard pretrained DINOv3"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "❌ Pretraining failed or was interrupted"
    echo "========================================================================"
    echo "Check logs at: $OUTPUT_DIR/logs/log.txt"
    echo "To resume, run this script again (it will auto-resume from checkpoint)"
    echo "========================================================================"
    exit 1
fi


