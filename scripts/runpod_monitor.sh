#!/bin/bash
# RunPod monitoring script
# Shows GPU usage, running containers, and training progress

set -euo pipefail

while true; do
    clear
    echo "========================================="
    echo "UltraShape Training Monitor"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================="
    echo ""

    # GPU status
    echo "=== GPU Status ==="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits | \
            awk -F',' '{printf "GPU %s: %s | %s°C | %s%% util | %sMB / %sMB\n", $1, $2, $3, $4, $5, $6}'
    else
        echo "nvidia-smi not found"
    fi
    echo ""

    # Docker containers
    echo "=== Running Containers ==="
    if docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null | grep -v NAMES; then
        docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    else
        echo "No running containers"
    fi
    echo ""

    # Disk usage
    echo "=== Disk Usage ==="
    if [ -d "/workspace" ]; then
        df -h /workspace | tail -n1 | awk '{printf "Used: %s / %s (%s)\n", $3, $2, $5}'
    else
        df -h . | tail -n1 | awk '{printf "Used: %s / %s (%s)\n", $3, $2, $5}'
    fi
    echo ""

    # Latest checkpoint
    if [ -d "/workspace/UltraShape-Training/outputs" ]; then
        echo "=== Latest Checkpoint ==="
        find /workspace/UltraShape-Training/outputs -name "*.pt" -o -name "*.ckpt" 2>/dev/null | \
            xargs ls -lht 2>/dev/null | head -n1 | \
            awk '{printf "Modified: %s %s %s - %s\n", $6, $7, $8, $9}' || echo "No checkpoints yet"
        echo ""
    fi

    # Training logs (last 5 lines)
    if [ -d "/workspace/UltraShape-Training/logs" ]; then
        echo "=== Recent Log Entries ==="
        find /workspace/UltraShape-Training/logs -name "*.log" -type f 2>/dev/null | \
            xargs tail -n 5 2>/dev/null || echo "No logs yet"
        echo ""
    fi

    # WandB status
    if [ -n "${WANDB_API_KEY:-}" ]; then
        echo "=== WandB ==="
        echo "Status: Configured"
        echo "Project: ${WANDB_PROJECT:-ultrashape-collectibles}"
        echo "Dashboard: https://wandb.ai"
        echo ""
    fi

    echo "========================================="
    echo "Press Ctrl+C to exit | Refreshing in 5s..."
    echo "========================================="
    sleep 5
done
