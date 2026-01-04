#!/bin/bash
# Upload models from macOS to RunPod via rsync
# Usage: bash scripts/upload_to_runpod.sh <runpod-ip> <ssh-port> <local-models-dir>

set -euo pipefail

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check arguments
if [ $# -lt 3 ]; then
    echo -e "${RED}Error: Missing arguments${NC}"
    echo ""
    echo "Usage: bash scripts/upload_to_runpod.sh <runpod-ip> <ssh-port> <local-models-dir>"
    echo ""
    echo "Example:"
    echo "  bash scripts/upload_to_runpod.sh 194.26.192.100 12345 /path/to/models"
    echo ""
    echo "Get RunPod connection info from:"
    echo "  RunPod Console → Your Pod → Connect → SSH"
    exit 1
fi

RUNPOD_IP="$1"
SSH_PORT="$2"
LOCAL_MODELS_DIR="$3"

# Validate local directory
if [ ! -d "$LOCAL_MODELS_DIR" ]; then
    echo -e "${RED}Error: Local directory not found: $LOCAL_MODELS_DIR${NC}"
    exit 1
fi

# Count files
MODEL_COUNT=$(find "$LOCAL_MODELS_DIR" -type f \( -iname "*.stl" -o -iname "*.obj" -o -iname "*.fbx" -o -iname "*.ply" \) | wc -l | tr -d ' ')
TOTAL_SIZE=$(du -sh "$LOCAL_MODELS_DIR" | awk '{print $1}')

echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}RunPod Upload Configuration${NC}"
echo -e "${YELLOW}========================================${NC}"
echo ""
echo "RunPod IP:       $RUNPOD_IP"
echo "SSH Port:        $SSH_PORT"
echo "Local Dir:       $LOCAL_MODELS_DIR"
echo "Model Count:     $MODEL_COUNT files"
echo "Total Size:      $TOTAL_SIZE"
echo "Remote Dir:      /workspace/UltraShape-Training/data/input/"
echo ""

# Test SSH connection
echo -e "${YELLOW}Testing SSH connection...${NC}"
if ! ssh -p "$SSH_PORT" -o ConnectTimeout=5 -o StrictHostKeyChecking=no root@"$RUNPOD_IP" "echo 'Connection OK'" 2>/dev/null; then
    echo -e "${RED}Error: Cannot connect to RunPod${NC}"
    echo ""
    echo "Please check:"
    echo "1. RunPod pod is running"
    echo "2. IP and port are correct"
    echo "3. SSH is enabled on the pod"
    exit 1
fi
echo -e "${GREEN}✓ SSH connection OK${NC}"
echo ""

# Confirm upload
echo -e "${YELLOW}Ready to upload $MODEL_COUNT models ($TOTAL_SIZE)${NC}"
read -p "Continue? (y/n): " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Upload cancelled"
    exit 0
fi

# Create remote directory
echo -e "${YELLOW}Creating remote directory...${NC}"
ssh -p "$SSH_PORT" root@"$RUNPOD_IP" "mkdir -p /workspace/UltraShape-Training/data/input/"

# Upload with rsync (with progress)
echo -e "${YELLOW}Uploading models...${NC}"
echo ""
rsync -avz --progress \
    -e "ssh -p $SSH_PORT" \
    --include="*.stl" \
    --include="*.STL" \
    --include="*.obj" \
    --include="*.OBJ" \
    --include="*.fbx" \
    --include="*.FBX" \
    --include="*.ply" \
    --include="*.PLY" \
    --exclude="*" \
    "$LOCAL_MODELS_DIR/" \
    root@"$RUNPOD_IP":/workspace/UltraShape-Training/data/input/

# Verify upload
echo ""
echo -e "${YELLOW}Verifying upload...${NC}"
REMOTE_COUNT=$(ssh -p "$SSH_PORT" root@"$RUNPOD_IP" "find /workspace/UltraShape-Training/data/input/ -type f | wc -l" | tr -d ' ')

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Upload Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Local files:     $MODEL_COUNT"
echo "Remote files:    $REMOTE_COUNT"
echo ""

if [ "$MODEL_COUNT" -eq "$REMOTE_COUNT" ]; then
    echo -e "${GREEN}✓ All files uploaded successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. SSH to RunPod: ssh -p $SSH_PORT root@$RUNPOD_IP"
    echo "2. cd /workspace/UltraShape-Training"
    echo "3. docker compose --profile processing run processing"
else
    echo -e "${YELLOW}⚠ Warning: File count mismatch${NC}"
    echo "Some files may not have uploaded. Check the rsync output above."
fi
