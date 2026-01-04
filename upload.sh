#!/bin/bash
# Quick upload script - uses dataset/ folder by default
# Usage: bash upload.sh <runpod-ip> <ssh-port>

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash upload.sh <runpod-ip> <ssh-port>"
    echo ""
    echo "Example:"
    echo "  bash upload.sh 194.26.192.100 12345"
    echo ""
    echo "This will upload from: /Users/brunopapa/Documents/Projects/UltraShape-Training/dataset"
    exit 1
fi

RUNPOD_IP="$1"
SSH_PORT="$2"
DATASET_DIR="/Users/brunopapa/Documents/Projects/UltraShape-Training/dataset"

# Check if dataset directory exists
if [ ! -d "$DATASET_DIR" ]; then
    echo "Error: Dataset directory not found: $DATASET_DIR"
    echo ""
    echo "Please create the directory and add your models:"
    echo "  mkdir -p $DATASET_DIR"
    echo "  cp /path/to/your/models/*.stl $DATASET_DIR/"
    exit 1
fi

# Run the main upload script
bash scripts/upload_to_runpod.sh "$RUNPOD_IP" "$SSH_PORT" "$DATASET_DIR"
