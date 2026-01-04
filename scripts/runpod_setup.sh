#!/bin/bash
# RunPod instance setup script
# Run this once when starting a new RunPod instance

set -euo pipefail

echo "==================================="
echo "UltraShape RunPod Setup"
echo "==================================="

# Update system
echo "Updating system packages..."
apt-get update -qq
apt-get install -y -qq git vim tmux htop curl

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
else
    echo "Docker already installed: $(docker --version)"
fi

# Clone repository if not exists
if [ ! -d "/workspace/UltraShape-Training" ]; then
    echo "Cloning repository..."
    cd /workspace
    read -p "Enter repository URL: " REPO_URL
    git clone "$REPO_URL" UltraShape-Training
else
    echo "Repository already cloned at /workspace/UltraShape-Training"
fi

cd /workspace/UltraShape-Training

# Create necessary directories
echo "Creating directories..."
mkdir -p data/input data/output checkpoints outputs logs temp

# Build Docker images
echo ""
read -p "Build which container? (processing/training/inference/all): " CONTAINER

case "$CONTAINER" in
    processing)
        echo "Building processing container..."
        docker build -f docker/Dockerfile.processing -t ultrashape-processing .
        ;;
    training)
        echo "Building training container..."
        docker build -f docker/Dockerfile.training -t ultrashape-training .
        ;;
    inference)
        echo "Building inference container..."
        docker build -f docker/Dockerfile.inference -t ultrashape-inference .
        ;;
    all)
        echo "Building all containers..."
        docker build -f docker/Dockerfile.processing -t ultrashape-processing .
        docker build -f docker/Dockerfile.training -t ultrashape-training .
        docker build -f docker/Dockerfile.inference -t ultrashape-inference .
        ;;
    *)
        echo "Invalid option: $CONTAINER"
        exit 1
        ;;
esac

# Setup WandB
echo ""
read -p "Enter WandB API key (or press Enter to skip): " WANDB_KEY
if [ -n "$WANDB_KEY" ]; then
    echo "export WANDB_API_KEY=$WANDB_KEY" >> ~/.bashrc
    export WANDB_API_KEY="$WANDB_KEY"
    echo "WandB configured!"
else
    echo "Skipping WandB setup"
fi

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Upload your .obj files to: /workspace/UltraShape-Training/data/input/"
echo "2. Run data processing:"
echo "   docker run --rm -v /workspace/UltraShape-Training/data/input:/input \\"
echo "     -v /workspace/UltraShape-Training/data/output:/output \\"
echo "     ultrashape-processing:latest --input-dir /input --output-dir /output"
echo ""
echo "3. Download pretrained weights:"
echo "   docker run --rm -v /workspace/UltraShape-Training/checkpoints:/workspace/checkpoints \\"
echo "     ultrashape-training:latest python3 scripts/download_pretrained.py --output-dir /workspace/checkpoints"
echo ""
echo "4. Start training:"
echo "   docker run --gpus all --rm -v /workspace/UltraShape-Training/data/output:/workspace/data \\"
echo "     -v /workspace/UltraShape-Training/checkpoints:/workspace/checkpoints \\"
echo "     -v /workspace/UltraShape-Training/outputs:/workspace/outputs \\"
echo "     -e WANDB_API_KEY=\$WANDB_API_KEY -p 6006:6006 --shm-size=16g \\"
echo "     ultrashape-training:latest"
echo ""
echo "5. Monitor training:"
echo "   bash scripts/runpod_monitor.sh"
echo ""
