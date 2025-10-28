#!/bin/bash
# Training Monitoring Script for DINOCell SSL Pretraining

echo "============================================================"
echo "DINOCell SSL Pretraining Monitor"
echo "============================================================"
echo ""

# Check if training is running
if ps aux | grep -q "[p]ython.*train.py"; then
    echo "✅ Training process is RUNNING"
    
    # Get process info
    echo ""
    echo "Process info:"
    ps aux | grep "[p]ython.*train.py" | awk '{print "  PID: " $2 ", CPU: " $3 "%, MEM: " $4 "%"}'
    
    # GPU status
    echo ""
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F, '{printf "  GPU %s: %s | Temp: %s°C | GPU Util: %s%% | Mem Util: %s%% | Mem: %s/%s MB\n", $1, $2, $3, $4, $5, $6, $7}'
    
    # Latest log lines
    echo ""
    echo "Latest training logs (last 20 lines):"
    echo "------------------------------------------------------------"
    tail -20 /home/shadeform/DINOCell/training/ssl_pretraining/pretraining_final.log
    echo "------------------------------------------------------------"
    
    # Check for iteration progress
    echo ""
    if grep -q "iteration:" /home/shadeform/DINOCell/training/ssl_pretraining/pretraining_final.log 2>/dev/null; then
        echo "Training iterations:"
        grep "iteration:" /home/shadeform/DINOCell/training/ssl_pretraining/pretraining_final.log | tail -5
    else
        echo "⏳ Still initializing... (this is normal, can take 10-20 mins for S3 discovery)"
    fi
    
    # Wandb status
    echo ""
    if [ -f /home/shadeform/wandb_cache/.wandb ]; then
        echo "Wandb:"
        cd /home/shadeform/DINOCell/training/ssl_pretraining
        export WANDB_DIR=/home/shadeform/wandb_cache
        /home/shadeform/miniconda3/envs/dinocell/bin/wandb status 2>/dev/null || echo "  Wandb syncing in background"
    fi
    
else
    echo "❌ Training process is NOT running"
    echo ""
    echo "Last 30 lines of log:"
    echo "------------------------------------------------------------"
    tail -30 /home/shadeform/DINOCell/training/ssl_pretraining/pretraining_final.log
    echo "------------------------------------------------------------"
fi

echo ""
echo "============================================================"
echo "Commands:"
echo "  Watch logs:  tail -f /home/shadeform/DINOCell/training/ssl_pretraining/pretraining_final.log"
echo "  GPU monitor: watch -n 1 nvidia-smi"
echo "  Re-run this: bash /home/shadeform/DINOCell/training/ssl_pretraining/monitor_training.sh"
echo "============================================================"

