# UltraShape Docker Setup Guide

This guide explains how to use the Docker containers for UltraShape fine-tuning on your custom dataset.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose v2+
- NVIDIA Docker runtime (for GPU containers)
- NVIDIA GPU with 24GB+ VRAM (A40/A5000 minimum, H100 recommended)
- WandB account (optional, for training monitoring)

## Container Overview

This project provides three specialized Docker containers:

1. **Processing Container** (`ultrashape-processing`) - CPU-only
   - Converts raw 3D models to watertight meshes
   - Generates multi-view renders using Blender
   - Creates training/validation splits

2. **Training Container** (`ultrashape-training`) - GPU required
   - Fine-tunes UltraShape DiT model on your dataset
   - Supports multi-GPU training with DeepSpeed
   - Includes TensorBoard and WandB monitoring

3. **Inference Container** (`ultrashape-inference`) - GPU required
   - Serves trained model via REST API
   - Lightweight inference-only dependencies
   - Health checks and monitoring endpoints

## Quick Start

### 1. Prepare Your Data

Place your 3D models (OBJ format) in `data/input/`:

```bash
mkdir -p data/input
cp /path/to/your/models/*.obj data/input/
```

### 2. Process Dataset

Run the processing container to prepare training data:

```bash
# Build the processing container
docker compose build processing

# Run data processing (adjust --limit for testing)
docker compose --profile processing run processing \
  --input-dir /input \
  --output-dir /output \
  --num-workers 4 \
  --num-views 4 \
  --limit 100
```

This will create:
- `data/output/watertight/` - Processed meshes
- `data/output/renders/` - Multi-view images
- `data/output/data_list/` - Train/val splits
- `data/output/render.json` - Render mappings

### 3. Download Pretrained Weights

The training container can download pretrained VAE weights:

```bash
docker compose --profile training run training \
  python3 scripts/download_pretrained.py \
  --output-dir /workspace/checkpoints \
  --model-type vae
```

### 4. Train the Model

Set up WandB (optional but recommended):

```bash
export WANDB_API_KEY=your_api_key_here
```

Start training:

```bash
# Build training container
docker compose build training

# Start training
docker compose --profile training up training
```

Monitor training:
- **TensorBoard**: http://localhost:6006
- **WandB**: Check your WandB dashboard

Training outputs:
- Checkpoints: `outputs/dit_ultrashape/exp1_token8192/`
- Logs: `logs/`

### 5. Run Inference

Once training is complete, use the inference API:

```bash
# Build inference container
docker compose build inference

# Start inference API
docker compose --profile inference up -d inference

# Check health
curl http://localhost:8000/health

# Run inference
curl -X POST http://localhost:8000/infer \
  -F "image=@inputs/image/test.png" \
  -F "coarse_mesh=@inputs/coarse_mesh/test.glb" \
  -o refined_mesh.glb
```

API endpoints:
- **GET /health** - Health check
- **POST /infer** - Mesh refinement (accepts image + coarse mesh, returns refined mesh)

## Configuration

### Environment Variables

**Training Container:**
- `WANDB_API_KEY` - WandB API key for experiment tracking
- `WANDB_PROJECT` - WandB project name (default: ultrashape-collectibles)
- `MASTER_IP` - Master node IP for multi-node training (default: localhost)
- `CUDA_VISIBLE_DEVICES` - GPU selection (auto-detected if not set)

**Inference Container:**
- `CHECKPOINT_PATH` - Path to model checkpoint (default: /workspace/checkpoints/ultrashape_v1.pt)
- `CONFIG_PATH` - Path to config file (default: /workspace/configs/infer_dit_refine.yaml)
- `INFERENCE_TIMEOUT` - Max inference time in seconds (default: 300)
- `PORT` - API port (default: 8000)
- `TEMP_DIR` - Temporary file directory (default: /tmp/ultrashape_inference)

### Volume Mounts

Key directories mounted in containers:

```yaml
processing:
  - ./data/input:/input           # Raw model files
  - ./data/output:/output         # Processed training data

training:
  - ./data/output:/workspace/data # Training data
  - ./checkpoints:/workspace/checkpoints  # Model weights
  - ./outputs:/workspace/outputs  # Training outputs
  - ./logs:/workspace/logs        # Training logs

inference:
  - ./checkpoints:/workspace/checkpoints  # Model weights
  - ./temp:/workspace/temp        # Temporary files
```

## RunPod Deployment

These containers are designed for RunPod deployment:

1. **Push to Registry**:
   ```bash
   # Tag and push containers
   docker tag ultrashape-training:latest your-registry/ultrashape-training:latest
   docker push your-registry/ultrashape-training:latest
   ```

2. **Create RunPod Template**:
   - Use custom container: `your-registry/ultrashape-training:latest`
   - GPU: A40/A5000 for testing, H100 for production
   - Volume: Mount persistent storage for checkpoints/data
   - Ports: Expose 6006 (TensorBoard), 8000 (Inference API)

3. **Environment Setup**:
   - Set `WANDB_API_KEY` in RunPod environment variables
   - Mount data volume to `/workspace/data`
   - Mount checkpoint volume to `/workspace/checkpoints`

See `docs/runpod-setup.md` for detailed RunPod deployment instructions.

## Troubleshooting

### GPU Not Detected

Ensure NVIDIA Docker runtime is installed:

```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If this fails, install NVIDIA Container Toolkit:
- https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

### Out of Memory Errors

Reduce batch size or gradient accumulation:
- Edit `configs/train_dit_refine.yaml`
- Decrease `per_device_train_batch_size`
- Increase `gradient_accumulation_steps`

### Blender Rendering Fails

If rendering hangs or fails:
- Reduce `--num-workers` (try 2 or 1)
- Check input OBJ files are valid
- Ensure enough disk space in `data/output/`

### Training Crashes

Common issues:
1. **CUDA OOM**: Reduce batch size in config
2. **DeepSpeed errors**: Check GPU memory, try ZeRO stage 2 instead of 3
3. **Data loading errors**: Verify `data_list/train.json` paths are correct

## Advanced Usage

### Multi-GPU Training

To use multiple GPUs on a single machine:

```bash
# Auto-detect all GPUs
docker compose --profile training up training

# Or specify GPUs manually
CUDA_VISIBLE_DEVICES=0,1,2,3 docker compose --profile training up training
```

### Custom Configs

To use custom training configurations:

1. Copy config: `cp configs/train_dit_refine.yaml configs/my_config.yaml`
2. Edit settings in `my_config.yaml`
3. Update `train.sh` to use your config:
   ```bash
   export config=configs/my_config.yaml
   ```
4. Rebuild training container

### Debugging

Run containers in interactive mode:

```bash
# Processing (CPU)
docker compose --profile processing run --rm processing bash

# Training (GPU)
docker compose --profile training run --rm --entrypoint bash training

# Inference (GPU)
docker compose --profile inference run --rm --entrypoint bash inference
```

## Performance Tips

1. **Data Processing**:
   - Use `--num-workers` = number of CPU cores
   - Process in batches using `--limit` for large datasets
   - Use SSDs for `data/output` to speed up rendering

2. **Training**:
   - Start with small `--limit` (100-1000 models) to test pipeline
   - Monitor GPU utilization via `nvidia-smi` or WandB
   - Use mixed precision training (enabled by default)

3. **Inference**:
   - Keep model loaded (API server does this automatically)
   - Use batch inference for multiple models
   - Monitor temp directory disk usage

## Support

For issues or questions:
- Check logs: `docker compose logs <service>`
- Review configuration: `docker compose config`
- Verify GPU: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

## Next Steps

After completing local testing, proceed to:
1. `docs/runpod-setup.md` - Deploy to RunPod for production training
2. Scale up dataset processing
3. Monitor training metrics and tune hyperparameters
4. Deploy inference API for production use
