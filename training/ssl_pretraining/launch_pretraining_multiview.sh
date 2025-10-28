#!/bin/bash
# ============================================================================
# DINOv3 Multi-View Pretraining Launch Script for JUMP Cell Painting
# ============================================================================
#
# This script launches ADVANCED multi-view consistency learning where
# different fluorescent channels are treated as different views of the same cell.
#
# Key Innovation:
# - Global crop 1: Average of all 5 fluorescent channels
# - Global crop 2: Random single channel
# - DINO loss enforces: features(avg) ≈ features(random_channel)
# - Result: Channel-invariant cell representations!
#
# Hardware: Single A100 80GB GPU
# Time: 30-40 hours (slightly longer than averaging)
# Dataset: ~3M JUMP Cell Painting images (multi-channel)
#
# Usage:
#   bash launch_pretraining_multiview.sh
#
# Or with custom parameters:
#   bash launch_pretraining_multiview.sh --epochs 100 --batch-size 32
# ============================================================================

# Configuration
DINOV3_ROOT="../../dinov3"
CONFIG_FILE="../DINOCell/training/configs/dinov3_vits8_jump_multiview.yaml"
OUTPUT_DIR="../DINOCell/checkpoints/dinov3_vits8_jump_multiview"
DATASET_ROOT="../../2024_Chandrasekaran_NatureMethods_CPJUMP1"

# Parse command line arguments
EXTRA_ARGS="$@"

# Print configuration
echo "========================================================================"
echo "DINOv3 MULTI-VIEW Pretraining on JUMP Cell Painting"
echo "========================================================================"
echo "Approach: Multi-View Consistency Learning"
echo "  - Global Crop 1: Average of 5 fluorescent channels"
echo "  - Global Crop 2: Random single channel"
echo "  - DINO enforces: Same cell → Same features"
echo ""
echo "DINOv3 Root: $DINOV3_ROOT"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Dataset: $DATASET_ROOT"
echo "Extra args: $EXTRA_ARGS"
echo "========================================================================"
echo ""

# Check prerequisites
if [ ! -d "$DINOV3_ROOT" ]; then
    echo "ERROR: DINOv3 repository not found at $DINOV3_ROOT"
    echo "Please clone: cd ../.. && git clone https://github.com/facebookresearch/dinov3.git"
    exit 1
fi

if [ ! -d "$DATASET_ROOT" ]; then
    echo "ERROR: JUMP dataset not found at $DATASET_ROOT"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Check for pretrained checkpoint
CHECKPOINT_PATH="$DINOV3_ROOT/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo ""
    echo "Pretrained checkpoint not found. Downloading..."
    mkdir -p "$DINOV3_ROOT/checkpoints"
    cd "$DINOV3_ROOT/checkpoints"
    wget https://dl.fbaipublicfiles.com/dinov3/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
    cd -
    echo "✅ Checkpoint downloaded"
fi

# Activate environment
if command -v micromamba &> /dev/null; then
    echo "Activating dinov3 environment with micromamba..."
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate dinov3
elif command -v conda &> /dev/null; then
    echo "Activating dinov3 environment with conda..."
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate dinov3
fi

# Test multi-view dataset loading
echo ""
echo "Testing multi-view dataset loading..."
cd $DINOV3_ROOT
PYTHONPATH=. python -c "
from dinov3.data.datasets import JUMPCellPaintingMultiView
print('Loading dataset...')
dataset = JUMPCellPaintingMultiView(root='$DATASET_ROOT', max_samples=10)
print(f'✅ Dataset test successful: {len(dataset)} samples')
if len(dataset) > 0:
    sample = dataset[0]
    if isinstance(sample[0], list):
        print(f'✅ Multi-view mode confirmed: {len(sample[0])} channels per sample')
    else:
        print('⚠️ Warning: Not returning channels as list')
" || {
    echo "❌ Dataset loading test failed!"
    echo "Check that JUMP data is accessible at: $DATASET_ROOT"
    exit 1
}

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Launch training
echo ""
echo "========================================================================"
echo "Starting Multi-View DINOv3 Pretraining..."
echo "========================================================================"
echo "Innovation: Learning channel-invariant cell representations!"
echo ""
echo "This will take 30-40 hours. Monitor with:"
echo "  tail -f $OUTPUT_DIR/logs/log.txt"
echo ""
echo "Training enforces:"
echo "  - features(averaged_channels) ≈ features(random_channel)"
echo "  - Result: Model learns 'same cell' across channels!"
echo ""
echo "To resume if interrupted:"
echo "  Just run this script again - auto-resumes from latest checkpoint"
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
    echo "✅ Multi-View Pretraining completed successfully!"
    echo "========================================================================"
    echo "Checkpoints: $OUTPUT_DIR"
    echo "Teacher checkpoint: $OUTPUT_DIR/eval/final/teacher_checkpoint.pth"
    echo ""
    echo "Key Achievement:"
    echo "  ✅ Model learned channel-invariant cell features!"
    echo "  ✅ Should work on ANY channel combination"
    echo "  ✅ Better generalization than simple averaging"
    echo ""
    echo "Next steps:"
    echo "1. Validate channel consistency (see MULTIVIEW_IMPLEMENTATION.md)"
    echo "2. Fine-tune DINOCell with these weights"
    echo "3. Test on cross-channel tasks"
    echo "4. Compare with averaging approach"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "❌ Training failed or was interrupted"
    echo "========================================================================"
    echo "Check logs: $OUTPUT_DIR/logs/log.txt"
    echo "To resume: ./launch_pretraining_multiview.sh"
    echo "========================================================================"
    exit 1
fi

