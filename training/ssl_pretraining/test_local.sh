#!/bin/bash
# Quick Test Script - Verify pipeline works before full S3 training
# ===================================================================

echo "========================================================================"
echo "🧪 DINOv3 Quick Test - Local Example Images"
echo "========================================================================"
echo "Purpose: Verify pipeline works before 30-40hr S3 training"
echo "Dataset: /home/shadeform/example_images (10 fields, 80 images)"
echo "Iterations: ~50 (should complete in 5-10 minutes)"
echo "========================================================================"
echo ""

# Configuration
DINOV3_ROOT="../../dinov3_modified/dinov3"
CONFIG_FILE="$(pwd)/configs/test_local_images.yaml"
OUTPUT_DIR="$(pwd)/test_output"
WANDB_PROJECT="dinocell-test"
WANDB_NAME="local-pipeline-test"

# Setup environment
export WANDB_DIR=/home/shadeform/wandb_cache
export WANDB_CACHE_DIR=/home/shadeform/wandb_cache
export WANDB_DATA_DIR=/home/shadeform/wandb_cache
export PATH="/home/shadeform/miniconda3/envs/dinocell/bin:$PATH"

# Check prerequisites
if [ ! -d "$DINOV3_ROOT" ]; then
    echo "❌ ERROR: DINOv3 not found"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERROR: Config not found at $CONFIG_FILE"
    exit 1
fi

if [ ! -d "/home/shadeform/example_images" ]; then
    echo "❌ ERROR: Example images not found at /home/shadeform/example_images"
    exit 1
fi

echo "✅ Config: $CONFIG_FILE"
echo "✅ Example images found: $(find /home/shadeform/example_images -name '*.tiff' | wc -l) images"
echo "✅ Output: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Navigate to dinov3
cd $DINOV3_ROOT

# Set wandb to online
wandb online 2>/dev/null

echo "========================================================================"
echo "🚀 Starting Quick Test..."
echo "========================================================================"
echo "Expected: 50 iterations in 5-10 minutes"
echo "Monitor: tail -f $OUTPUT_DIR/logs/log.txt"
echo "========================================================================"
echo ""

# Run training
PYTHONPATH=. WANDB_DIR=$WANDB_DIR WANDB_CACHE_DIR=$WANDB_CACHE_DIR WANDB_DATA_DIR=$WANDB_DATA_DIR \
python dinov3/train/train.py \
    --config-file "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR" \
    wandb.enabled=true \
    wandb.project=$WANDB_PROJECT \
    wandb.name=$WANDB_NAME

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✅ TEST PASSED! Pipeline works!"
    echo "========================================================================"
    echo "Now you can run the full S3 training with confidence:"
    echo "  bash launch_ssl_with_s3_wandb.sh"
    echo "========================================================================"
else
    echo ""
    echo "❌ Test failed - check logs at $OUTPUT_DIR/logs/log.txt"
    exit 1
fi

