# UltraShape Docker Setup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create three Docker containers for UltraShape fine-tuning: data processing, training, and inference with REST API.

**Architecture:** Three separate Dockerfiles optimized for different purposes - CPU-focused data processing with Blender/PyMeshLab, GPU-accelerated training with DeepSpeed/WandB, and lightweight inference with FastAPI.

**Tech Stack:** Docker, CUDA 12.1, PyTorch 2.5.1, Blender, PyMeshLab, DeepSpeed, FastAPI, WandB, TensorBoard

---

## Implementation Progress

**Status:** 2 of 6 tasks completed

- ✅ **Task 1:** Data Processing Dockerfile - COMPLETED
  - All files created and tested
  - Fixed critical camera rotation bug in Blender rendering
  - Production-ready

- ✅ **Task 2:** Training Dockerfile - COMPLETED
  - All files created and tested
  - Fixed GPU detection logic and error handling
  - Production-ready

- ⏳ **Task 3:** Inference Dockerfile and REST API - NEXT
- ⏳ **Task 4:** Docker Compose - Pending
- ⏳ **Task 5:** RunPod Scripts - Pending
- ⏳ **Task 6:** README Updates - Pending

**Last Updated:** 2026-01-04 (Tasks 1-2 complete)

---

## Task 1: Create Data Processing Dockerfile

**Files:**
- Create: `docker/Dockerfile.processing`
- Create: `docker/.dockerignore`
- Create: `scripts/process_dataset.py`
- Create: `scripts/blender_render.py`
- Create: `scripts/watertight_mesh.py`

**Step 1: Create docker directory and .dockerignore**

```bash
mkdir -p docker
```

Create `docker/.dockerignore`:
```
.git
.github
docs/
*.md
checkpoints/
outputs/
__pycache__
*.pyc
.DS_Store
.vscode
.idea
```

**Step 2: Create data processing Dockerfile**

Create `docker/Dockerfile.processing`:
```dockerfile
FROM ubuntu:22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    wget \
    xz-utils \
    libglu1-mesa \
    libxi6 \
    libxrender1 \
    libxkbcommon0 \
    libsm6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Blender 3.6 LTS (headless)
WORKDIR /opt
RUN wget -q https://download.blender.org/release/Blender3.6/blender-3.6.5-linux-x64.tar.xz && \
    tar -xf blender-3.6.5-linux-x64.tar.xz && \
    rm blender-3.6.5-linux-x64.tar.xz && \
    ln -s /opt/blender-3.6.5-linux-x64/blender /usr/local/bin/blender

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    pymeshlab==2022.2.post3 \
    numpy==1.24.4 \
    trimesh==4.4.7 \
    Pillow==12.0.0 \
    tqdm==4.66.5 \
    PyYAML==6.0.2

# Copy scripts
COPY scripts/blender_render.py /workspace/scripts/
COPY scripts/watertight_mesh.py /workspace/scripts/
COPY scripts/process_dataset.py /workspace/scripts/
COPY scripts/sampling.py /workspace/scripts/

# Set entrypoint
ENTRYPOINT ["python3", "/workspace/scripts/process_dataset.py"]
```

**Step 3: Create watertight mesh processing script**

Create `scripts/watertight_mesh.py`:
```python
#!/usr/bin/env python3
"""
Watertight mesh processing using PyMeshLab.
Fills holes, fixes non-manifold geometry, ensures closed surfaces.
"""
import argparse
import pymeshlab as ml
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_watertight(input_path: str, output_path: str) -> bool:
    """
    Convert mesh to watertight format.

    Args:
        input_path: Path to input .obj file
        output_path: Path to output watertight .obj file

    Returns:
        True if successful, False otherwise
    """
    try:
        ms = ml.MeshSet()
        ms.load_new_mesh(input_path)

        # Remove duplicate vertices
        ms.meshing_remove_duplicate_vertices()

        # Remove unreferenced vertices
        ms.meshing_remove_unreferenced_vertices()

        # Fill holes
        ms.meshing_close_holes(maxholesize=30)

        # Remove duplicate faces
        ms.meshing_remove_duplicate_faces()

        # Fix non-manifold edges
        ms.meshing_repair_non_manifold_edges()

        # Fix non-manifold vertices
        ms.meshing_repair_non_manifold_vertices()

        # Re-orient faces consistently
        ms.meshing_re_orient_faces_coherently()

        # Save watertight mesh
        ms.save_current_mesh(output_path)

        logger.info(f"Successfully processed: {input_path} -> {output_path}")
        return True

    except Exception as e:
        logger.error(f"Failed to process {input_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Make mesh watertight")
    parser.add_argument("--input", required=True, help="Input mesh file")
    parser.add_argument("--output", required=True, help="Output mesh file")
    args = parser.parse_args()

    success = make_watertight(args.input, args.output)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
```

**Step 4: Create Blender rendering script**

Create `scripts/blender_render.py`:
```python
#!/usr/bin/env python3
"""
Blender headless rendering script for multi-view mesh rendering.
Usage: blender --background --python blender_render.py -- --mesh <path> --output <dir>
"""
import bpy
import sys
import argparse
import math
from pathlib import Path


def clear_scene():
    """Remove all default objects from scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def setup_camera(distance=2.5):
    """Create and setup camera."""
    bpy.ops.object.camera_add()
    camera = bpy.context.active_object
    camera.location = (0, -distance, 0)
    camera.rotation_euler = (math.radians(90), 0, 0)
    bpy.context.scene.camera = camera
    return camera


def setup_lighting():
    """Setup 3-point lighting."""
    # Key light
    bpy.ops.object.light_add(type='AREA', location=(3, -3, 3))
    key_light = bpy.context.active_object
    key_light.data.energy = 300

    # Fill light
    bpy.ops.object.light_add(type='AREA', location=(-2, -2, 2))
    fill_light = bpy.context.active_object
    fill_light.data.energy = 150

    # Back light
    bpy.ops.object.light_add(type='AREA', location=(0, 2, 2))
    back_light = bpy.context.active_object
    back_light.data.energy = 100


def load_mesh(mesh_path):
    """Load mesh file."""
    if mesh_path.endswith('.obj'):
        bpy.ops.import_scene.obj(filepath=mesh_path)
    elif mesh_path.endswith('.glb') or mesh_path.endswith('.gltf'):
        bpy.ops.import_scene.gltf(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_path}")

    # Get imported object and center it
    imported_obj = bpy.context.selected_objects[0]
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    imported_obj.location = (0, 0, 0)

    return imported_obj


def render_view(output_path, angle_deg=0):
    """Render single view at specified angle."""
    # Rotate object
    obj = bpy.context.scene.objects[0]
    obj.rotation_euler = (0, 0, math.radians(angle_deg))

    # Render
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def main():
    # Parse arguments after --
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="Input mesh file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--views", type=int, default=4, help="Number of views")
    parser.add_argument("--resolution", type=int, default=1024, help="Image resolution")
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup scene
    clear_scene()
    setup_camera()
    setup_lighting()

    # Configure render settings
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.render.resolution_x = args.resolution
    bpy.context.scene.render.resolution_y = args.resolution
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.cycles.samples = 128

    # Load mesh
    load_mesh(args.mesh)

    # Render multiple views
    angles = [i * (360 / args.views) for i in range(args.views)]
    for i, angle in enumerate(angles):
        output_path = str(output_dir / f"view_{i}.png")
        render_view(output_path, angle)
        print(f"Rendered view {i} at {angle}° -> {output_path}")


if __name__ == "__main__":
    main()
```

**Step 5: Create main dataset processing orchestration script**

Create `scripts/process_dataset.py`:
```python
#!/usr/bin/env python3
"""
Main dataset processing orchestration script.
Processes meshes through watertighting, rendering, and sampling pipeline.
"""
import argparse
import json
import logging
import multiprocessing as mp
import subprocess
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_single_model(args_tuple):
    """Process a single model through the pipeline."""
    model_path, output_dir, num_views = args_tuple
    model_id = model_path.stem

    try:
        # Step 1: Watertight processing
        watertight_dir = output_dir / "meshes"
        watertight_dir.mkdir(parents=True, exist_ok=True)
        watertight_path = watertight_dir / f"{model_id}.obj"

        cmd = [
            "python3", "/workspace/scripts/watertight_mesh.py",
            "--input", str(model_path),
            "--output", str(watertight_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Watertight failed for {model_id}: {result.stderr}")
            return None

        # Step 2: Blender rendering
        render_dir = output_dir / "renders" / model_id
        render_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "blender", "--background", "--python",
            "/workspace/scripts/blender_render.py", "--",
            "--mesh", str(watertight_path),
            "--output", str(render_dir),
            "--views", str(num_views),
            "--resolution", "1024"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Rendering failed for {model_id}: {result.stderr}")
            return None

        # Step 3: Point cloud sampling (placeholder - will use UltraShape's sampling.py)
        sample_dir = output_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Return metadata
        return {
            "model_id": model_id,
            "watertight_mesh": str(watertight_path),
            "renders": [str(render_dir / f"view_{i}.png") for i in range(num_views)],
            "sample": str(sample_dir / f"{model_id}.npz")
        }

    except Exception as e:
        logger.error(f"Error processing {model_id}: {e}")
        return None


def create_dataset_splits(metadata: List[Dict], output_dir: Path, train_ratio=0.9):
    """Create train/val splits and save JSON files."""
    import random
    random.shuffle(metadata)

    split_idx = int(len(metadata) * train_ratio)
    train_data = metadata[:split_idx]
    val_data = metadata[split_idx:]

    data_list_dir = output_dir / "data_list"
    data_list_dir.mkdir(parents=True, exist_ok=True)

    with open(data_list_dir / "train.json", "w") as f:
        json.dump([m["model_id"] for m in train_data], f, indent=2)

    with open(data_list_dir / "val.json", "w") as f:
        json.dump([m["model_id"] for m in val_data], f, indent=2)

    # Create render mapping
    render_mapping = {m["model_id"]: m["renders"] for m in metadata}
    with open(output_dir / "render.json", "w") as f:
        json.dump(render_mapping, f, indent=2)

    logger.info(f"Created splits: {len(train_data)} train, {len(val_data)} val")


def main():
    parser = argparse.ArgumentParser(description="Process dataset for UltraShape training")
    parser.add_argument("--input-dir", required=True, help="Directory with input .obj files")
    parser.add_argument("--output-dir", required=True, help="Output directory for processed data")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--num-views", type=int, default=4, help="Number of render views per model")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of models to process")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .obj files
    model_paths = list(input_dir.glob("*.obj"))
    if args.limit:
        model_paths = model_paths[:args.limit]

    logger.info(f"Found {len(model_paths)} models to process")

    # Process models in parallel
    process_args = [(p, output_dir, args.num_views) for p in model_paths]

    with mp.Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_model, process_args),
            total=len(process_args),
            desc="Processing models"
        ))

    # Filter out failed models
    metadata = [r for r in results if r is not None]
    logger.info(f"Successfully processed {len(metadata)}/{len(model_paths)} models")

    # Create dataset splits
    create_dataset_splits(metadata, output_dir)

    logger.info(f"Dataset processing complete. Output: {output_dir}")


if __name__ == "__main__":
    main()
```

**Step 6: Test the Dockerfile builds**

```bash
cd docker
docker build -f Dockerfile.processing -t ultrashape-processing:latest ..
```

Expected: Build completes successfully with no errors

**Step 7: Commit data processing container**

```bash
git add docker/Dockerfile.processing docker/.dockerignore
git add scripts/process_dataset.py scripts/blender_render.py scripts/watertight_mesh.py
git commit -m "feat: add data processing Docker container and scripts"
```

---

## Task 2: Create Training Dockerfile

**Files:**
- Create: `docker/Dockerfile.training`
- Create: `scripts/train_deepspeed.sh`
- Modify: `train.sh`
- Create: `scripts/download_pretrained.py`

**Step 1: Create training Dockerfile**

Create `docker/Dockerfile.training`:
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    build-essential \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch with CUDA 12.1
RUN pip3 install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install flash attention (requires CUDA)
RUN pip3 install --no-cache-dir flash-attn==2.8.3 --no-build-isolation

# Install other dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Install cubvh for MC acceleration
RUN pip3 install --no-cache-dir git+https://github.com/ashawkey/cubvh --no-build-isolation

# Install PyTorch3D
RUN pip3 install --no-cache-dir --no-build-isolation \
    "git+https://github.com/facebookresearch/pytorch3d.git@stable"

# Install torch_cluster
RUN pip3 install --no-cache-dir \
    https://data.pyg.org/whl/torch-2.5.0%2Bcu121/torch_cluster-1.6.3%2Bpt25cu121-cp310-cp310-linux_x86_64.whl

# Copy project files
COPY . /workspace/

# Create necessary directories
RUN mkdir -p /workspace/data /workspace/checkpoints /workspace/outputs /workspace/logs

# Set environment variables for training
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0+PTX"
ENV FORCE_CUDA=1

# Expose TensorBoard port
EXPOSE 6006

# Default command
CMD ["bash", "train.sh", "0"]
```

**Step 2: Create DeepSpeed training launcher script**

Create `scripts/train_deepspeed.sh`:
```bash
#!/bin/bash

# DeepSpeed multi-node training launcher
# Usage: bash scripts/train_deepspeed.sh <node_num> <node_rank> <num_gpu_per_node> <master_ip> <config> <output_dir>

node_num=$1
node_rank=$2
num_gpu_per_node=$3
master_ip=$4
config=$5
output_dir=$6

# Set distributed training environment variables
export MASTER_ADDR=${master_ip:-"localhost"}
export MASTER_PORT=${MASTER_PORT:-29500}
export NODE_RANK=$node_rank
export WORLD_SIZE=$((node_num * num_gpu_per_node))

echo "========================================"
echo "Training Configuration:"
echo "Node Number: $node_num"
echo "Node Rank: $node_rank"
echo "GPUs per Node: $num_gpu_per_node"
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "Config: $config"
echo "Output Dir: $output_dir"
echo "========================================"

# Create output directory
mkdir -p $output_dir

# Launch training with DeepSpeed
deepspeed --num_nodes=$node_num \
          --num_gpus=$num_gpu_per_node \
          --master_addr=$MASTER_ADDR \
          --master_port=$MASTER_PORT \
          --node_rank=$node_rank \
          main.py \
          --config $config \
          --output_dir $output_dir
```

**Step 3: Update main training script**

Modify `train.sh`:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export num_gpu_per_node=8

export node_num=1
export node_rank=${1:-0}
export master_ip=${MASTER_IP:-localhost}

# For single-node training, override to use available GPUs
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # Auto-detect available GPUs
    num_gpu_per_node=$(nvidia-smi --list-gpus | wc -l)
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((num_gpu_per_node - 1)))
fi

echo "Using GPUs: $CUDA_VISIBLE_DEVICES"

############## vae ##############
# export config=configs/train_vae_refine.yaml
# export output_dir=outputs/vae_ultrashape/exp1_token8192
# bash scripts/train_deepspeed.sh $node_num $node_rank $num_gpu_per_node $master_ip $config $output_dir

############## dit ##############
export config=configs/train_dit_refine.yaml
export output_dir=outputs/dit_ultrashape/exp1_token8192
bash scripts/train_deepspeed.sh $node_num $node_rank $num_gpu_per_node $master_ip $config $output_dir
```

**Step 4: Create pretrained model download script**

Create `scripts/download_pretrained.py`:
```python
#!/usr/bin/env python3
"""
Download pretrained UltraShape weights from Hugging Face.
"""
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_pretrained_weights(output_dir: str, model_type: str = "dit"):
    """
    Download pretrained weights from Hugging Face.

    Args:
        output_dir: Directory to save weights
        model_type: 'vae' or 'dit'
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    repo_id = "infinith/UltraShape"

    if model_type == "dit":
        filename = "ultrashape_v1.pt"
    elif model_type == "vae":
        filename = "vae_weights.pt"  # Update with actual filename
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Downloading {filename} from {repo_id}...")

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=output_dir,
        local_dir_use_symlinks=False
    )

    logger.info(f"Downloaded to: {local_path}")
    return local_path


def main():
    parser = argparse.ArgumentParser(description="Download pretrained UltraShape weights")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    parser.add_argument("--model-type", choices=["vae", "dit"], default="dit", help="Model type")
    args = parser.parse_args()

    download_pretrained_weights(args.output_dir, args.model_type)


if __name__ == "__main__":
    main()
```

**Step 5: Test training Dockerfile builds**

```bash
cd docker
docker build -f Dockerfile.training -t ultrashape-training:latest ..
```

Expected: Build completes successfully (may take 15-30 minutes due to PyTorch3D compilation)

**Step 6: Commit training container**

```bash
git add docker/Dockerfile.training scripts/train_deepspeed.sh scripts/download_pretrained.py
git add train.sh
git commit -m "feat: add training Docker container with DeepSpeed support"
```

---

## Task 3: Create Inference Dockerfile and REST API

**Files:**
- Create: `docker/Dockerfile.inference`
- Create: `api/main.py`
- Create: `api/models.py`
- Create: `api/inference.py`
- Create: `api/requirements.txt`

**Step 1: Create inference API requirements**

Create `api/requirements.txt`:
```
fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
pydantic==2.10.0
aiofiles==24.1.0
```

**Step 2: Create inference Dockerfile**

Create `docker/Dockerfile.inference`:
```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt .
COPY api/requirements.txt api_requirements.txt

# Install PyTorch with CUDA 12.1 (runtime only, smaller)
RUN pip3 install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

# Install core dependencies (minimal for inference)
RUN pip3 install --no-cache-dir \
    diffusers==0.30.0 \
    einops==0.8.1 \
    imageio==2.36.0 \
    numpy==1.24.4 \
    omegaconf==2.3.0 \
    opencv-python-headless==4.11.0.86 \
    Pillow==12.0.0 \
    safetensors==0.7.0 \
    scikit-image==0.24.0 \
    timm==1.0.22 \
    tqdm==4.66.5 \
    transformers==4.37.2 \
    trimesh==4.4.7

# Install API dependencies
RUN pip3 install --no-cache-dir -r api_requirements.txt

# Install cubvh for fast MC
RUN pip3 install --no-cache-dir git+https://github.com/ashawkey/cubvh --no-build-isolation

# Copy only necessary files for inference
COPY ultrashape/ /workspace/ultrashape/
COPY configs/ /workspace/configs/
COPY scripts/infer_dit_refine.py /workspace/scripts/
COPY api/ /workspace/api/

# Create directories
RUN mkdir -p /workspace/checkpoints /workspace/temp /workspace/logs

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')"

# Run API server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 3: Create API data models**

Create `api/models.py`:
```python
"""
Pydantic models for API request/response schemas.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RefineRequest(BaseModel):
    """Request model for mesh refinement."""
    num_steps: int = Field(default=50, ge=10, le=100, description="Number of diffusion steps")
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0, description="Classifier-free guidance scale")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")


class JobResponse(BaseModel):
    """Response model for job creation."""
    job_id: str
    status: JobStatus
    message: str


class JobStatusResponse(BaseModel):
    """Response model for job status query."""
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=100.0)
    message: Optional[str] = None
    error: Optional[str] = None
    result_path: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    gpu_available: bool
    model_loaded: bool
    version: str


class MetricsResponse(BaseModel):
    """Response model for system metrics."""
    gpu_memory_used: float
    gpu_memory_total: float
    gpu_utilization: float
    active_jobs: int
    completed_jobs: int
    failed_jobs: int
```

**Step 4: Create inference logic wrapper**

Create `api/inference.py`:
```python
"""
Inference wrapper for UltraShape refinement.
"""
import torch
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import subprocess
import json

logger = logging.getLogger(__name__)


class UltraShapeInference:
    """Wrapper for UltraShape inference pipeline."""

    def __init__(self, checkpoint_path: str, config_path: str, device: str = "cuda"):
        """
        Initialize inference pipeline.

        Args:
            checkpoint_path: Path to trained model checkpoint
            config_path: Path to inference config YAML
            device: Device to run inference on
        """
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path
        self.device = device
        self.model_loaded = False

        # Check if checkpoint exists
        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
        else:
            self.model_loaded = True
            logger.info(f"Model checkpoint ready: {checkpoint_path}")

    def refine_mesh(
        self,
        mesh_path: str,
        image_path: str,
        output_path: str,
        num_steps: int = 50,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Refine mesh using UltraShape pipeline.

        Args:
            mesh_path: Path to input coarse mesh
            image_path: Path to reference image
            output_path: Path to save refined mesh
            num_steps: Number of diffusion steps
            seed: Random seed

        Returns:
            Dictionary with inference results
        """
        try:
            # Build command
            cmd = [
                "python3",
                "scripts/infer_dit_refine.py",
                "--ckpt", self.checkpoint_path,
                "--image", image_path,
                "--mesh", mesh_path,
                "--config", self.config_path,
                "--output", output_path,
                "--num_steps", str(num_steps)
            ]

            if seed is not None:
                cmd.extend(["--seed", str(seed)])

            # Run inference
            logger.info(f"Running inference: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                raise RuntimeError(f"Inference failed: {result.stderr}")

            return {
                "success": True,
                "output_path": output_path,
                "message": "Refinement completed successfully"
            }

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def get_gpu_stats(self) -> Dict[str, float]:
        """Get GPU memory statistics."""
        if not torch.cuda.is_available():
            return {"available": False}

        return {
            "available": True,
            "memory_allocated": torch.cuda.memory_allocated() / 1e9,
            "memory_reserved": torch.cuda.memory_reserved() / 1e9,
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1e9
        }
```

**Step 5: Create FastAPI main application**

Create `api/main.py`:
```python
"""
FastAPI application for UltraShape inference service.
"""
import asyncio
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, Optional
import aiofiles

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import torch

from .models import (
    RefineRequest, JobResponse, JobStatusResponse,
    HealthResponse, MetricsResponse, JobStatus
)
from .inference import UltraShapeInference

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="UltraShape Inference API",
    description="REST API for UltraShape 3D mesh refinement",
    version="1.0.0"
)

# Global state
CHECKPOINT_PATH = os.getenv("CHECKPOINT_PATH", "/workspace/checkpoints/ultrashape_v1.pt")
CONFIG_PATH = os.getenv("CONFIG_PATH", "/workspace/configs/infer_dit_refine.yaml")
TEMP_DIR = Path("/workspace/temp")
TEMP_DIR.mkdir(exist_ok=True)

# Initialize inference engine
inference_engine = UltraShapeInference(CHECKPOINT_PATH, CONFIG_PATH)

# Job tracking
jobs: Dict[str, Dict] = {}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        gpu_available=torch.cuda.is_available(),
        model_loaded=inference_engine.model_loaded,
        version="1.0.0"
    )


@app.get("/status")
async def get_status():
    """Get system status."""
    gpu_stats = inference_engine.get_gpu_stats()
    return {
        "status": "running",
        "gpu_available": gpu_stats.get("available", False),
        "gpu_memory": gpu_stats,
        "active_jobs": len([j for j in jobs.values() if j["status"] == JobStatus.PROCESSING]),
        "total_jobs": len(jobs)
    }


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics."""
    gpu_stats = inference_engine.get_gpu_stats()

    return MetricsResponse(
        gpu_memory_used=gpu_stats.get("memory_allocated", 0.0),
        gpu_memory_total=gpu_stats.get("memory_total", 0.0),
        gpu_utilization=0.0,  # Placeholder
        active_jobs=len([j for j in jobs.values() if j["status"] == JobStatus.PROCESSING]),
        completed_jobs=len([j for j in jobs.values() if j["status"] == JobStatus.COMPLETED]),
        failed_jobs=len([j for j in jobs.values() if j["status"] == JobStatus.FAILED])
    )


async def process_refinement_job(
    job_id: str,
    mesh_path: Path,
    image_path: Path,
    output_path: Path,
    params: RefineRequest
):
    """Background task to process refinement job."""
    try:
        jobs[job_id]["status"] = JobStatus.PROCESSING
        jobs[job_id]["progress"] = 0.0

        # Run inference
        result = inference_engine.refine_mesh(
            str(mesh_path),
            str(image_path),
            str(output_path),
            num_steps=params.num_steps,
            seed=params.seed
        )

        if result["success"]:
            jobs[job_id]["status"] = JobStatus.COMPLETED
            jobs[job_id]["progress"] = 100.0
            jobs[job_id]["result_path"] = str(output_path)
            jobs[job_id]["message"] = "Refinement completed"
        else:
            jobs[job_id]["status"] = JobStatus.FAILED
            jobs[job_id]["error"] = result.get("error", "Unknown error")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs[job_id]["status"] = JobStatus.FAILED
        jobs[job_id]["error"] = str(e)


@app.post("/refine", response_model=JobResponse)
async def refine_mesh(
    background_tasks: BackgroundTasks,
    mesh_file: UploadFile = File(...),
    image_file: UploadFile = File(...),
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None
):
    """
    Submit mesh refinement job.

    Args:
        mesh_file: Input coarse mesh (.glb or .obj)
        image_file: Reference image (.png or .jpg)
        num_steps: Number of diffusion steps (10-100)
        guidance_scale: CFG scale (1.0-20.0)
        seed: Random seed for reproducibility
    """
    # Create job
    job_id = str(uuid.uuid4())
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir()

    # Save uploaded files
    mesh_path = job_dir / mesh_file.filename
    image_path = job_dir / image_file.filename
    output_path = job_dir / "refined.glb"

    async with aiofiles.open(mesh_path, 'wb') as f:
        await f.write(await mesh_file.read())

    async with aiofiles.open(image_path, 'wb') as f:
        await f.write(await image_file.read())

    # Create job record
    params = RefineRequest(num_steps=num_steps, guidance_scale=guidance_scale, seed=seed)
    jobs[job_id] = {
        "status": JobStatus.PENDING,
        "progress": 0.0,
        "mesh_path": str(mesh_path),
        "image_path": str(image_path),
        "output_path": str(output_path),
        "params": params.dict()
    }

    # Start background processing
    background_tasks.add_task(
        process_refinement_job,
        job_id, mesh_path, image_path, output_path, params
    )

    return JobResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message="Job submitted for processing"
    )


@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of a specific job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        progress=job.get("progress", 0.0),
        message=job.get("message"),
        error=job.get("error"),
        result_path=job.get("result_path")
    )


@app.get("/jobs/{job_id}/download")
async def download_result(job_id: str):
    """Download refined mesh result."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    result_path = job.get("result_path")
    if not result_path or not Path(result_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found")

    return FileResponse(result_path, filename="refined.glb")


@app.get("/logs")
async def get_logs(lines: int = 100):
    """Get recent log entries."""
    # Placeholder - implement actual log reading
    return {
        "lines": lines,
        "logs": ["Log reading not yet implemented"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

**Step 6: Test inference Dockerfile builds**

```bash
cd docker
docker build -f Dockerfile.inference -t ultrashape-inference:latest ..
```

Expected: Build completes successfully

**Step 7: Commit inference container and API**

```bash
git add docker/Dockerfile.inference
git add api/
git commit -m "feat: add inference Docker container with FastAPI REST API"
```

---

## Task 4: Create Docker Compose for Local Testing

**Files:**
- Create: `docker-compose.yml`
- Create: `README-docker.md`

**Step 1: Create docker-compose configuration**

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  # Data processing service (CPU-only, runs once)
  processing:
    build:
      context: .
      dockerfile: docker/Dockerfile.processing
    image: ultrashape-processing:latest
    volumes:
      - ./data/input:/input
      - ./data/output:/output
    command: >
      --input-dir /input
      --output-dir /output
      --num-workers 4
      --num-views 4
      --limit 10
    profiles:
      - processing

  # Training service (requires GPU)
  training:
    build:
      context: .
      dockerfile: docker/Dockerfile.training
    image: ultrashape-training:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - WANDB_API_KEY=${WANDB_API_KEY}
      - WANDB_PROJECT=ultrashape-collectibles
      - MASTER_IP=localhost
    volumes:
      - ./data/output:/workspace/data
      - ./checkpoints:/workspace/checkpoints
      - ./outputs:/workspace/outputs
      - ./logs:/workspace/logs
    ports:
      - "6006:6006"  # TensorBoard
    shm_size: '16gb'
    profiles:
      - training

  # Inference API service (requires GPU)
  inference:
    build:
      context: .
      dockerfile: docker/Dockerfile.inference
    image: ultrashape-inference:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - CHECKPOINT_PATH=/workspace/checkpoints/ultrashape_v1.pt
      - CONFIG_PATH=/workspace/configs/infer_dit_refine.yaml
    volumes:
      - ./checkpoints:/workspace/checkpoints
      - ./temp:/workspace/temp
    ports:
      - "8000:8000"  # API
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    profiles:
      - inference
```

**Step 2: Create Docker usage documentation**

Create `README-docker.md`:
```markdown
# UltraShape Docker Setup Guide

This guide explains how to use the Docker containers for UltraShape fine-tuning.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose v2+
- NVIDIA Docker runtime (for GPU containers)
- NVIDIA GPU with 24GB+ VRAM (A40/A5000 or better)

## Quick Start

### 1. Build All Containers

```bash
# Build all containers
docker compose build

# Or build individually
docker compose build processing
docker compose build training
docker compose build inference
```

### 2. Data Processing

Process your .obj models through the pipeline:

```bash
# Place your .obj files in data/input/
mkdir -p data/input

# Run processing container
docker compose --profile processing up processing

# Check output in data/output/
ls data/output/
```

### 3. Download Pretrained Weights

```bash
docker run --rm -v $(pwd)/checkpoints:/workspace/checkpoints \
    ultrashape-training:latest \
    python3 scripts/download_pretrained.py --output-dir /workspace/checkpoints
```

### 4. Training

```bash
# Set WandB API key
export WANDB_API_KEY=your_wandb_key_here

# Start training
docker compose --profile training up training

# Monitor with TensorBoard
open http://localhost:6006
```

### 5. Inference API

```bash
# Start inference server
docker compose --profile inference up inference

# Test API
curl http://localhost:8000/health

# View API docs
open http://localhost:8000/docs
```

## RunPod Deployment

### Data Processing on RunPod

```bash
# SSH into RunPod instance
ssh root@<runpod-ip>

# Clone repository
git clone <your-repo>
cd UltraShape-Training

# Build processing container
docker build -f docker/Dockerfile.processing -t ultrashape-processing .

# Run processing
docker run --rm \
    -v /workspace/models:/input \
    -v /workspace/data:/output \
    ultrashape-processing:latest \
    --input-dir /input \
    --output-dir /output \
    --num-workers 16 \
    --num-views 4
```

### Training on RunPod

```bash
# Build training container
docker build -f docker/Dockerfile.training -t ultrashape-training .

# Download pretrained weights
docker run --rm \
    -v /workspace/checkpoints:/workspace/checkpoints \
    ultrashape-training:latest \
    python3 scripts/download_pretrained.py

# Run training
docker run --gpus all --rm \
    -v /workspace/data:/workspace/data \
    -v /workspace/checkpoints:/workspace/checkpoints \
    -v /workspace/outputs:/workspace/outputs \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -p 6006:6006 \
    --shm-size=16g \
    ultrashape-training:latest
```

### Inference on RunPod

```bash
# Build inference container
docker build -f docker/Dockerfile.inference -t ultrashape-inference .

# Run inference API
docker run --gpus all -d \
    -v /workspace/checkpoints:/workspace/checkpoints \
    -p 8000:8000 \
    --name ultrashape-api \
    ultrashape-inference:latest
```

## Monitoring from macOS

### TensorBoard (SSH Tunnel)

```bash
# Create SSH tunnel
ssh -L 6006:localhost:6006 root@<runpod-ip>

# Open in browser
open http://localhost:6006
```

### WandB (Cloud)

Just visit your WandB dashboard:
```
https://wandb.ai/<username>/ultrashape-collectibles
```

### API Logs

```bash
# Get recent logs
curl http://<runpod-ip>:8000/logs?lines=100

# Get system status
curl http://<runpod-ip>:8000/status
```

## Troubleshooting

### Out of Memory Errors

Reduce batch size in `configs/train_dit_refine.yaml`:
```yaml
dataset:
  params:
    batch_size: 1  # Reduce if OOM
```

### Blender Rendering Fails

Check Blender is working:
```bash
docker run --rm ultrashape-processing:latest blender --version
```

### GPU Not Detected

Verify NVIDIA runtime:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## File Structure

```
.
├── docker/
│   ├── Dockerfile.processing    # Data processing container
│   ├── Dockerfile.training       # Training container
│   └── Dockerfile.inference      # Inference container
├── data/
│   ├── input/                    # Your .obj files
│   └── output/                   # Processed dataset
├── checkpoints/                  # Model weights
├── outputs/                      # Training outputs
└── temp/                         # Inference temp files
```
```

**Step 3: Test docker-compose configuration**

```bash
docker compose config
```

Expected: Valid YAML with no errors

**Step 4: Commit Docker Compose setup**

```bash
git add docker-compose.yml README-docker.md
git commit -m "feat: add Docker Compose configuration and documentation"
```

---

## Task 5: Create RunPod Setup Scripts

**Files:**
- Create: `scripts/runpod_setup.sh`
- Create: `scripts/runpod_monitor.sh`
- Create: `docs/RUNPOD-GUIDE.md`

**Step 1: Create RunPod initial setup script**

Create `scripts/runpod_setup.sh`:
```bash
#!/bin/bash
# RunPod instance setup script
# Run this once when starting a new RunPod instance

set -e

echo "==================================="
echo "UltraShape RunPod Setup"
echo "==================================="

# Update system
echo "Updating system packages..."
apt-get update
apt-get install -y git vim tmux htop

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
fi

# Clone repository if not exists
if [ ! -d "/workspace/UltraShape-Training" ]; then
    echo "Cloning repository..."
    cd /workspace
    read -p "Enter repository URL: " REPO_URL
    git clone $REPO_URL UltraShape-Training
fi

cd /workspace/UltraShape-Training

# Create necessary directories
echo "Creating directories..."
mkdir -p data/input data/output checkpoints outputs logs temp

# Build Docker images
echo "Building Docker images..."
read -p "Build which container? (processing/training/inference/all): " CONTAINER

case $CONTAINER in
    processing)
        docker build -f docker/Dockerfile.processing -t ultrashape-processing .
        ;;
    training)
        docker build -f docker/Dockerfile.training -t ultrashape-training .
        ;;
    inference)
        docker build -f docker/Dockerfile.inference -t ultrashape-inference .
        ;;
    all)
        docker build -f docker/Dockerfile.processing -t ultrashape-processing .
        docker build -f docker/Dockerfile.training -t ultrashape-training .
        docker build -f docker/Dockerfile.inference -t ultrashape-inference .
        ;;
esac

# Setup WandB
read -p "Enter WandB API key (or press Enter to skip): " WANDB_KEY
if [ ! -z "$WANDB_KEY" ]; then
    echo "export WANDB_API_KEY=$WANDB_KEY" >> ~/.bashrc
    export WANDB_API_KEY=$WANDB_KEY
fi

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Place your .obj files in /workspace/UltraShape-Training/data/input/"
echo "2. Run data processing: docker run ..."
echo "3. Download pretrained weights: docker run ..."
echo "4. Start training: docker run ..."
echo ""
```

**Step 2: Create RunPod monitoring script**

Create `scripts/runpod_monitor.sh`:
```bash
#!/bin/bash
# RunPod monitoring script
# Shows GPU usage, running containers, and training progress

while true; do
    clear
    echo "========================================="
    echo "UltraShape Training Monitor"
    echo "Time: $(date)"
    echo "========================================="
    echo ""

    # GPU status
    echo "=== GPU Status ==="
    nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits | \
        awk -F',' '{printf "GPU %s: %s | %s°C | %s%% util | %sMB / %sMB\n", $1, $2, $3, $4, $5, $6}'
    echo ""

    # Docker containers
    echo "=== Running Containers ==="
    docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    echo ""

    # Disk usage
    echo "=== Disk Usage ==="
    df -h /workspace | tail -n1 | awk '{printf "Used: %s / %s (%s)\n", $3, $2, $5}'
    echo ""

    # Latest checkpoint
    if [ -d "/workspace/UltraShape-Training/outputs" ]; then
        echo "=== Latest Checkpoint ==="
        ls -lht /workspace/UltraShape-Training/outputs/**/*.ckpt 2>/dev/null | head -n1 || echo "No checkpoints yet"
        echo ""
    fi

    # WandB status
    if [ ! -z "$WANDB_API_KEY" ]; then
        echo "=== WandB ==="
        echo "Logged in: Yes"
        echo "Dashboard: https://wandb.ai"
        echo ""
    fi

    echo "Press Ctrl+C to exit"
    sleep 5
done
```

**Step 3: Create RunPod deployment guide**

Create `docs/RUNPOD-GUIDE.md`:
```markdown
# RunPod Deployment Guide

Complete guide for deploying UltraShape training on RunPod.

## Phase 1: Testing Setup (A40/A5000)

### Step 1: Create RunPod Instance

1. Go to https://runpod.io
2. Select GPU: **A40 (48GB)** or **A5000 (24GB)**
3. Select Template: **PyTorch 2.1** or **RunPod PyTorch**
4. Storage: **500GB** Network Volume
5. Expose ports: **6006** (TensorBoard), **8000** (API)
6. Deploy instance

### Step 2: Initial Setup

SSH into your instance:
```bash
ssh root@<runpod-ip>
```

Run setup script:
```bash
cd /workspace
wget https://raw.githubusercontent.com/<your-repo>/main/scripts/runpod_setup.sh
bash runpod_setup.sh
```

### Step 3: Prepare Test Dataset

Upload 1000 test models:
```bash
# From your macOS:
rsync -avz --progress /path/to/1000/models/*.obj root@<runpod-ip>:/workspace/UltraShape-Training/data/input/
```

Or use RunPod's file manager to upload.

### Step 4: Process Data

```bash
cd /workspace/UltraShape-Training

# Start processing
docker run --rm \
    -v $(pwd)/data/input:/input \
    -v $(pwd)/data/output:/output \
    ultrashape-processing:latest \
    --input-dir /input \
    --output-dir /output \
    --num-workers 16 \
    --num-views 4

# Monitor progress
tail -f logs/processing.log
```

Expected: ~17 hours for 1000 models

### Step 5: Download Pretrained Weights

```bash
docker run --rm \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    ultrashape-training:latest \
    python3 scripts/download_pretrained.py \
    --output-dir /workspace/checkpoints \
    --model-type dit
```

### Step 6: Configure Training

Edit `configs/train_dit_refine.yaml`:
```yaml
dataset:
  params:
    training_data_list: /workspace/data/output/data_list
    sample_pcd_dir: /workspace/data/output/samples
    image_data_json: /workspace/data/output/render.json

model:
  params:
    ckpt_path: /workspace/checkpoints/ultrashape_v1.pt
    vae_config:
      from_pretrained: /workspace/checkpoints/vae_weights.pt
```

### Step 7: Start Training

In tmux session (so it persists):
```bash
tmux new -s training

docker run --gpus all --rm \
    -v $(pwd)/data/output:/workspace/data \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    -v $(pwd)/outputs:/workspace/outputs \
    -v $(pwd)/logs:/workspace/logs \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_PROJECT=ultrashape-test \
    -p 6006:6006 \
    --shm-size=16g \
    ultrashape-training:latest

# Detach: Ctrl+B, then D
```

### Step 8: Monitor from macOS

**Option 1: TensorBoard**
```bash
# Create SSH tunnel
ssh -L 6006:localhost:6006 root@<runpod-ip>

# Open browser
open http://localhost:6006
```

**Option 2: WandB**
```
https://wandb.ai/<your-username>/ultrashape-test
```

**Option 3: SSH + Monitor Script**
```bash
ssh root@<runpod-ip>
cd /workspace/UltraShape-Training
bash scripts/runpod_monitor.sh
```

### Step 9: Test Inference

After 10k-20k training steps:

```bash
# Stop training (Ctrl+C in tmux)
tmux attach -t training

# Start inference API
docker run --gpus all -d \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    -v $(pwd)/outputs:/workspace/checkpoints \
    -v $(pwd)/temp:/workspace/temp \
    -p 8000:8000 \
    --name ultrashape-api \
    ultrashape-inference:latest

# Test API
curl http://localhost:8000/health
```

Access API from macOS:
```bash
# Create SSH tunnel
ssh -L 8000:localhost:8000 root@<runpod-ip>

# Open API docs
open http://localhost:8000/docs
```

## Phase 2: Production Training (H100)

### Step 1: Migrate to H100

1. Stop A40/A5000 instance
2. Create new H100 instance (80GB)
3. Attach same Network Volume
4. Run setup script

### Step 2: Process Full Dataset

Upload all 30,000 models and process:
```bash
docker run --rm \
    -v $(pwd)/data/input:/input \
    -v $(pwd)/data/output:/output \
    ultrashape-processing:latest \
    --input-dir /input \
    --output-dir /output \
    --num-workers 32 \
    --num-views 4
```

Expected: ~21 days (parallelize across multiple CPU instances if needed)

### Step 3: Full Training

Update config for larger batch size (H100 has more memory):
```yaml
dataset:
  params:
    batch_size: 2  # H100 can handle larger batches
```

Start training:
```bash
tmux new -s production-training

docker run --gpus all --rm \
    -v $(pwd)/data/output:/workspace/data \
    -v $(pwd)/checkpoints:/workspace/checkpoints \
    -v $(pwd)/outputs:/workspace/outputs \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_PROJECT=ultrashape-production \
    -p 6006:6006 \
    --shm-size=32g \
    ultrashape-training:latest
```

Expected: 50k-100k steps, ~3-6 days on H100

## Troubleshooting

### Container Exits Immediately

Check logs:
```bash
docker logs <container-id>
```

### CUDA Out of Memory

Reduce batch size or enable gradient checkpointing:
```yaml
dataset:
  params:
    batch_size: 1

training:
  accumulate_grad_batches: 8  # Increase this
```

### Data Processing Slow

Parallelize across multiple CPU instances:
```bash
# Split your dataset into chunks
# Process each chunk on separate instance
# Combine results
```

### Can't Access TensorBoard

Check port forwarding:
```bash
# On RunPod, ensure port 6006 is exposed
# On macOS, verify SSH tunnel is active
netstat -an | grep 6006
```

## Cost Optimization

### A40 (~$0.34/hr)
- Testing: ~10 hours = ~$3.40
- Data processing: ~20 hours = ~$6.80

### H100 (~$2.50/hr)
- Training: ~150 hours = ~$375

**Total estimated cost for full pipeline: ~$385**

### Savings Tips
1. Use Spot instances (50% discount, but can be interrupted)
2. Process data on CPU-only instances (cheaper)
3. Stop instances when not training
4. Use network volumes to persist data across instance changes
```

**Step 4: Make scripts executable**

```bash
chmod +x scripts/runpod_setup.sh scripts/runpod_monitor.sh
```

**Step 5: Commit RunPod setup**

```bash
git add scripts/runpod_setup.sh scripts/runpod_monitor.sh docs/RUNPOD-GUIDE.md
git commit -m "feat: add RunPod deployment scripts and documentation"
```

---

## Task 6: Update Main README

**Files:**
- Modify: `README.md`

**Step 1: Add Docker section to main README**

Add to `README.md` after installation section:
```markdown
## 🐳 Docker Setup (Recommended)

We provide three Docker containers for the complete fine-tuning pipeline:

### Quick Start

```bash
# 1. Build containers
docker compose build

# 2. Process data
docker compose --profile processing up processing

# 3. Train model
export WANDB_API_KEY=your_key
docker compose --profile training up training

# 4. Run inference
docker compose --profile inference up inference
```

### Detailed Documentation

- **Docker Usage**: See [README-docker.md](README-docker.md)
- **RunPod Deployment**: See [docs/RUNPOD-GUIDE.md](docs/RUNPOD-GUIDE.md)
- **Design Document**: See [docs/plans/2026-01-04-ultrashape-finetuning-design.md](docs/plans/2026-01-04-ultrashape-finetuning-design.md)

### Container Purposes

1. **Data Processing** (`Dockerfile.processing`)
   - Watertight mesh conversion
   - Multi-view rendering with Blender
   - Point cloud sampling
   - Dataset organization

2. **Training** (`Dockerfile.training`)
   - Fine-tune DiT model
   - DeepSpeed optimization
   - WandB + TensorBoard monitoring
   - Multi-GPU support

3. **Inference** (`Dockerfile.inference`)
   - REST API for mesh refinement
   - Lightweight, optimized for speed
   - Health monitoring and metrics

### RunPod Deployment

Deploy on cloud GPUs easily:

```bash
# SSH to RunPod instance
ssh root@<runpod-ip>

# Run setup
wget <repo-url>/scripts/runpod_setup.sh
bash runpod_setup.sh
```

Monitor from your macOS:
- TensorBoard: `ssh -L 6006:localhost:6006 root@<runpod-ip>`
- WandB: https://wandb.ai/<username>/<project>
- API: `ssh -L 8000:localhost:8000 root@<runpod-ip>`
```

**Step 2: Commit README updates**

```bash
git add README.md
git commit -m "docs: add Docker and RunPod sections to README"
```

---

## Verification Steps

After completing all tasks, verify the setup:

**Step 1: Check all files created**

```bash
# Verify directory structure
tree -L 2 docker/ api/ scripts/

# Expected output:
# docker/
# ├── Dockerfile.processing
# ├── Dockerfile.training
# ├── Dockerfile.inference
# └── .dockerignore
#
# api/
# ├── main.py
# ├── models.py
# ├── inference.py
# └── requirements.txt
#
# scripts/
# ├── process_dataset.py
# ├── blender_render.py
# ├── watertight_mesh.py
# ├── download_pretrained.py
# ├── train_deepspeed.sh
# ├── runpod_setup.sh
# └── runpod_monitor.sh
```

**Step 2: Validate Docker configurations**

```bash
# Validate docker-compose
docker compose config

# Validate Dockerfiles (dry run)
docker build -f docker/Dockerfile.processing --target=0 . 2>&1 | grep -i "error" && echo "FAIL" || echo "PASS"
docker build -f docker/Dockerfile.training --target=0 . 2>&1 | grep -i "error" && echo "FAIL" || echo "PASS"
docker build -f docker/Dockerfile.inference --target=0 . 2>&1 | grep -i "error" && echo "FAIL" || echo "PASS"
```

**Step 3: Check Python syntax**

```bash
# Check all Python files for syntax errors
python3 -m py_compile scripts/*.py
python3 -m py_compile api/*.py
```

**Step 4: Review documentation**

```bash
# Ensure all documentation files exist
ls -l README.md README-docker.md docs/RUNPOD-GUIDE.md docs/plans/*.md
```

**Step 5: Final commit**

```bash
git status
git log --oneline -10

# Should show commits for:
# - Data processing container
# - Training container
# - Inference container
# - Docker Compose
# - RunPod scripts
# - README updates
```

---

## Success Criteria

✅ All Docker containers build successfully
✅ Docker Compose configuration valid
✅ All Python scripts syntax-valid
✅ Documentation complete and accurate
✅ Scripts executable and functional
✅ Git history clean with descriptive commits

---

## Next Steps After Implementation

1. **Test data processing locally** (small subset)
2. **Build containers on RunPod** (test environment)
3. **Process 1000 model subset**
4. **Run test training** (10k steps)
5. **Validate inference API**
6. **Scale to production** (H100 + full dataset)

---

**Plan Status:** Ready for execution
**Estimated Time:** 3-4 hours for implementation
**Prerequisites:** Docker installed, basic understanding of containers
