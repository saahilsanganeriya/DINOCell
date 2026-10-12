#!/bin/bash
# ============================================================================
# Ultimate DINOv3 SSL Pretraining Script
# ============================================================================
#
# Features:
# ✅ S3 Streaming (no local storage needed!)
# ✅ Multi-View Consistency Learning
# ✅ Wandb Logging (metrics, attention maps, PCA)
# ✅ Auto-resume capability
# ✅ One-command launch
#
# Usage:
#   # Default: S3 streaming + multi-view + wandb
#   ./launch_ssl_with_s3_wandb.sh
#
#   # Custom wandb project
#   ./launch_ssl_with_s3_wandb.sh --wandb-project my-cells
#
#   # Disable wandb
#   ./launch_ssl_with_s3_wandb.sh --no-wandb
#
# Requirements:
#   pip install boto3 smart-open wandb
# ============================================================================

# Parse arguments
USE_S3=true
USE_WANDB=true
WANDB_PROJECT="dinocell-ssl-pretraining"
WANDB_NAME="vits8-jump-multiview-s3"

while [[ $# -gt 0 ]]; do
    case $1 in
        --no-s3)
            USE_S3=false
            shift
            ;;
        --no-wandb)
            USE_WANDB=false
            shift
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-name)
            WANDB_NAME="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Configuration
DINOV3_ROOT="../../dinov3_modified/dinov3"
OUTPUT_DIR="./output"

# Select config based on S3 usage (absolute paths)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$USE_S3" = true ]; then
    CONFIG_FILE="$SCRIPT_DIR/configs/dinov3_vits8_jump_s3_multiview.yaml"
    DATASET_INFO="S3 Streaming (no local storage)"
else
    CONFIG_FILE="$SCRIPT_DIR/configs/dinov3_vits8_jump_multiview.yaml"
    DATASET_INFO="Local files"
fi

# Print configuration
echo "========================================================================"
echo "🚀 DINOv3 SSL Pretraining - Ultimate Configuration"
echo "========================================================================"
echo "Features:"
echo "  ✅ Multi-View Consistency Learning"
echo "  ✅ Patch Size 8 (higher resolution)"
echo "  ✅ Continue from pretrained checkpoint"
if [ "$USE_S3" = true ]; then
    echo "  ✅ S3 Streaming (saves ~500GB local storage)"
else
    echo "  ⚠️  Local files (ensure JUMP dataset downloaded)"
fi
if [ "$USE_WANDB" = true ]; then
    echo "  ✅ Wandb Logging"
    echo "     Project: $WANDB_PROJECT"
    echo "     Run: $WANDB_NAME"
else
    echo "  ⚠️  Wandb disabled"
fi
echo ""
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "Extra args: $EXTRA_ARGS"
echo "========================================================================"
echo ""

# Check prerequisites
if [ ! -d "$DINOV3_ROOT" ]; then
    echo "❌ ERROR: DINOv3 not found at $DINOV3_ROOT"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ ERROR: Config not found at $CONFIG_FILE"
    exit 1
fi

# Check S3 dependencies if using S3
if [ "$USE_S3" = true ]; then
    python -c "import boto3, smart_open" 2>/dev/null || {
        echo "❌ ERROR: S3 streaming requires boto3 and smart-open"
        echo "Install with: pip install boto3 smart-open"
        exit 1
    }
    echo "✅ S3 dependencies installed"
fi

# Check wandb if enabled
if [ "$USE_WANDB" = true ]; then
    python -c "import wandb" 2>/dev/null || {
        echo "❌ ERROR: Wandb not installed"
        echo "Install with: pip install wandb"
        echo "Then login: wandb login"
        exit 1
    }
    echo "✅ Wandb installed"
    
    # Configure wandb to use main filesystem for cache/logs
    export WANDB_DIR="$HOME/wandb_cache"
    export WANDB_CACHE_DIR="$HOME/wandb_cache"
    export WANDB_DATA_DIR="$HOME/wandb_cache"
    mkdir -p $WANDB_DIR
    
    # Set wandb to online mode
    wandb online 2>/dev/null
    
    # Check if logged in
    wandb login --relogin 2>/dev/null || {
        echo "⚠️  Wandb not logged in. Logging in now..."
        wandb login
    }
    echo "✅ Wandb configured (cache: $WANDB_DIR)"
fi

# Download checkpoint if needed
CHECKPOINT="$DINOV3_ROOT/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
if [ ! -f "$CHECKPOINT" ]; then
    echo "Downloading pretrained checkpoint..."
    mkdir -p "$DINOV3_ROOT/checkpoints"
    cd "$DINOV3_ROOT/checkpoints"
    wget -q --show-progress https://dl.fbaipublicfiles.com/dinov3/dinov3_vits16/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
    cd -
    echo "✅ Checkpoint downloaded"
fi

# Test S3 access if using S3
if [ "$USE_S3" = true ]; then
    echo ""
    echo "Testing S3 access..."
    python -c "
import boto3
from botocore import UNSIGNED
from botocore.config import Config

s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
response = s3.list_objects_v2(
    Bucket='cellpainting-gallery',
    Prefix='cpg0000-jump-pilot/source_4/images/2020_11_04_CPJUMP1/images/',
    MaxKeys=1
)
print(f'✅ S3 access confirmed: {len(response.get(\"Contents\", []))} objects found')
" || {
        echo "❌ ERROR: Cannot access S3 bucket"
        echo "Check internet connection"
        exit 1
    }
fi

# Activate environment
if command -v micromamba &> /dev/null; then
    eval "$(micromamba shell hook --shell bash)"
    micromamba activate dinov3
elif command -v conda &> /dev/null; then
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate dinov3
fi

# Navigate to dinov3
cd $DINOV3_ROOT

# Create output directory next to this script
OUTPUT_DIR_FULL="$SCRIPT_DIR/output"
mkdir -p "$OUTPUT_DIR_FULL"

# Build wandb override args
WANDB_ARGS=""
if [ "$USE_WANDB" = true ]; then
    WANDB_ARGS="wandb.enabled=true wandb.project=$WANDB_PROJECT wandb.name=$WANDB_NAME"
else
    WANDB_ARGS="wandb.enabled=false"
fi

# Launch training
echo ""
echo "========================================================================"
echo "🚀 Launching SSL Pretraining..."
echo "========================================================================"
echo "Timeline: 30-40 hours"
echo ""
echo "Monitor with:"
echo "  tail -f $OUTPUT_DIR/logs/log.txt"
if [ "$USE_WANDB" = true ]; then
    echo "  https://wandb.ai/$WANDB_PROJECT"
fi
echo ""
echo "Features being learned:"
echo "  📊 Channel-invariant cell representations"
echo "  🔬 Multi-scale features (patch-8)"
echo "  🎯 Self-supervised on 3M images"
if [ "$USE_S3" = true ]; then
    echo "  ☁️  Streaming from S3 (no local storage!)"
fi
echo "========================================================================"
echo ""

# Run training with environment variables for wandb
PYTHONPATH=. WANDB_DIR=$WANDB_DIR WANDB_CACHE_DIR=$WANDB_CACHE_DIR WANDB_DATA_DIR=$WANDB_DATA_DIR \
python dinov3/train/train.py \
    --config-file "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR_FULL" \
    $WANDB_ARGS \
    $EXTRA_ARGS

# Check result
if [ $? -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "🎉 SSL Pretraining Complete!"
    echo "========================================================================"
    echo "Checkpoint: $OUTPUT_DIR/eval/final/teacher_checkpoint.pth"
    if [ "$USE_WANDB" = true ]; then
        echo "Wandb: https://wandb.ai/$WANDB_PROJECT/runs/$WANDB_NAME"
    fi
    echo ""
    echo "Next: Fine-tune DINOCell with these weights"
    echo "========================================================================"
else
    echo "❌ Training failed - check logs"
    exit 1
fi

