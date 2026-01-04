# RunPod Deployment Guide

Complete guide for deploying UltraShape training on RunPod with your custom 3D model dataset.

## Table of Contents

1. [Phase 1: Testing Setup (A40/A5000)](#phase-1-testing-setup-a40a5000)
2. [Phase 2: Production Training (H100)](#phase-2-production-training-h100)
3. [Instance Configuration](#instance-configuration)
4. [Data Upload](#data-upload)
5. [Training Workflow](#training-workflow)
6. [Monitoring](#monitoring)
7. [Cost Optimization](#cost-optimization)
8. [Troubleshooting](#troubleshooting)

---

## Phase 1: Testing Setup (A40/A5000)

Start with budget-friendly GPUs to validate your pipeline before scaling up.

### Recommended Instance

- **GPU**: NVIDIA A40 (48GB) or A5000 (24GB)
- **Storage**: 200GB+ persistent volume
- **Cost**: ~$0.50-0.80/hour

### Why Start Here?

1. **Validate Pipeline**: Test data processing, training config, and monitoring
2. **Debug Issues**: Cheaper to troubleshoot on budget GPUs
3. **Estimate Costs**: Understand full training time and costs before scaling
4. **Quick Iteration**: Faster to restart if something goes wrong

### Setup Steps

1. **Create RunPod Instance**:
   - Select GPU: A40 or A5000
   - Select Template: "RunPod PyTorch" or "NVIDIA CUDA"
   - Storage: 200GB persistent volume
   - Expose ports: 6006 (TensorBoard), 8000 (API)

2. **SSH into Instance**:
   ```bash
   ssh root@<runpod-ip> -p <ssh-port>
   ```

3. **Run Setup Script**:
   ```bash
   # Download and run setup script
   cd /workspace
   git clone <your-repo-url> UltraShape-Training
   cd UltraShape-Training
   bash scripts/runpod_setup.sh
   ```

   The script will:
   - Install Docker if needed
   - Create necessary directories
   - Build Docker containers
   - Configure WandB (optional)

4. **Upload Test Data**:
   ```bash
   # From your macOS machine
   scp -P <ssh-port> data/input/*.obj root@<runpod-ip>:/workspace/UltraShape-Training/data/input/
   ```

5. **Process Data**:
   ```bash
   docker run --rm \
     -v /workspace/UltraShape-Training/data/input:/input \
     -v /workspace/UltraShape-Training/data/output:/output \
     ultrashape-processing:latest \
     --input-dir /input \
     --output-dir /output \
     --num-workers 16 \
     --num-views 4 \
     --limit 100
   ```

6. **Download Pretrained Weights**:
   ```bash
   docker run --rm \
     -v /workspace/UltraShape-Training/checkpoints:/workspace/checkpoints \
     ultrashape-training:latest \
     python3 scripts/download_pretrained.py \
     --output-dir /workspace/checkpoints \
     --model-type vae
   ```

7. **Start Training**:
   ```bash
   docker run --gpus all --rm \
     -v /workspace/UltraShape-Training/data/output:/workspace/data \
     -v /workspace/UltraShape-Training/checkpoints:/workspace/checkpoints \
     -v /workspace/UltraShape-Training/outputs:/workspace/outputs \
     -e WANDB_API_KEY=$WANDB_API_KEY \
     -p 6006:6006 \
     --shm-size=16g \
     ultrashape-training:latest
   ```

8. **Monitor Training**:
   ```bash
   # In a tmux session
   tmux new -s monitor
   bash scripts/runpod_monitor.sh
   ```

### Expected Results

- **Processing**: ~100 models in 30-60 minutes
- **Training**: ~1 epoch in 1-2 hours (depending on dataset size)
- **GPU Memory**: 18-22GB for batch_size=1

---

## Phase 2: Production Training (H100)

Once pipeline is validated, scale to H100 for full dataset training.

### Recommended Instance

- **GPU**: NVIDIA H100 (80GB)
- **Storage**: 500GB+ persistent volume
- **Cost**: ~$2.50-4.00/hour

### Why H100?

1. **Faster Training**: 3-5x faster than A40/A5000
2. **Larger Batches**: 80GB VRAM allows batch_size=4-8
3. **Better Quality**: More steps per epoch, better convergence
4. **Cost Effective**: Despite higher hourly cost, total cost is lower due to speed

### Migration from A40/A5000

1. **Save Checkpoints**: Ensure your test run checkpoints are saved
2. **Upload Full Dataset**: Upload all 30,000 models
3. **Create H100 Instance** with same setup
4. **Process Full Dataset**:
   ```bash
   docker run --rm \
     -v /workspace/UltraShape-Training/data/input:/input \
     -v /workspace/UltraShape-Training/data/output:/output \
     ultrashape-processing:latest \
     --input-dir /input \
     --output-dir /output \
     --num-workers 32 \
     --num-views 4
   ```

5. **Adjust Training Config**:
   Edit `configs/train_dit_refine.yaml`:
   ```yaml
   dataset:
     params:
       batch_size: 4  # Increase for H100

   training:
     gradient_accumulation_steps: 2
     max_epochs: 100
   ```

6. **Resume or Start Fresh**:
   ```bash
   # Start fresh training
   docker run --gpus all --rm \
     -v /workspace/UltraShape-Training/data/output:/workspace/data \
     -v /workspace/UltraShape-Training/checkpoints:/workspace/checkpoints \
     -v /workspace/UltraShape-Training/outputs:/workspace/outputs \
     -e WANDB_API_KEY=$WANDB_API_KEY \
     -p 6006:6006 \
     --shm-size=32g \
     ultrashape-training:latest
   ```

---

## Instance Configuration

### Storage Setup

**Persistent Volume** (recommended):
- Mount at `/workspace`
- Survives instance stops
- Keeps checkpoints, data, outputs

**Network Volume** (for large datasets):
- Faster I/O
- Can be shared across instances
- Good for multi-GPU setups

### Environment Variables

Create `/workspace/UltraShape-Training/.env`:

```bash
# WandB
WANDB_API_KEY=your_key_here
WANDB_PROJECT=ultrashape-collectibles

# Training
CUDA_VISIBLE_DEVICES=0
MASTER_IP=localhost

# Inference
CHECKPOINT_PATH=/workspace/checkpoints/ultrashape_v1.pt
CONFIG_PATH=/workspace/configs/infer_dit_refine.yaml
```

Load with: `source .env`

### Port Forwarding

For monitoring from your macOS machine:

```bash
# TensorBoard
ssh -L 6006:localhost:6006 root@<runpod-ip> -p <ssh-port>
# Open: http://localhost:6006

# Inference API
ssh -L 8000:localhost:8000 root@<runpod-ip> -p <ssh-port>
# Open: http://localhost:8000/health
```

---

## Data Upload

### Option 1: SCP (Small Datasets <10GB)

```bash
# From macOS
scp -P <ssh-port> -r data/input/*.obj root@<runpod-ip>:/workspace/UltraShape-Training/data/input/
```

### Option 2: rsync (Large Datasets)

```bash
# From macOS - supports resume
rsync -avz --progress -e "ssh -p <ssh-port>" \
  data/input/ \
  root@<runpod-ip>:/workspace/UltraShape-Training/data/input/
```

### Option 3: Cloud Storage

```bash
# Upload to S3/GCS from macOS
aws s3 sync data/input/ s3://your-bucket/models/

# Download on RunPod
aws s3 sync s3://your-bucket/models/ /workspace/UltraShape-Training/data/input/
```

---

## Training Workflow

### Full Pipeline

```bash
# 1. Setup (once per instance)
bash scripts/runpod_setup.sh

# 2. Upload data (from macOS)
scp -P <ssh-port> -r data/input/*.obj root@<runpod-ip>:/workspace/UltraShape-Training/data/input/

# 3. Process data
docker run --rm \
  -v /workspace/UltraShape-Training/data/input:/input \
  -v /workspace/UltraShape-Training/data/output:/output \
  ultrashape-processing:latest \
  --input-dir /input --output-dir /output

# 4. Download pretrained weights
docker run --rm \
  -v /workspace/UltraShape-Training/checkpoints:/workspace/checkpoints \
  ultrashape-training:latest \
  python3 scripts/download_pretrained.py --output-dir /workspace/checkpoints

# 5. Start training in tmux
tmux new -s training
docker run --gpus all --rm \
  -v /workspace/UltraShape-Training/data/output:/workspace/data \
  -v /workspace/UltraShape-Training/checkpoints:/workspace/checkpoints \
  -v /workspace/UltraShape-Training/outputs:/workspace/outputs \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -p 6006:6006 --shm-size=16g \
  ultrashape-training:latest

# Detach: Ctrl+B, D
# Reattach: tmux attach -t training

# 6. Monitor (separate terminal)
bash scripts/runpod_monitor.sh
```

### Resume Training

If training stops, resume from latest checkpoint:

```bash
# Find latest checkpoint
ls -lht /workspace/UltraShape-Training/outputs/dit_ultrashape/exp1_token8192/*.pt | head -n1

# Update train.sh or config to load checkpoint, then restart training
```

---

## Monitoring

### Local Monitoring (on RunPod instance)

```bash
# GPU usage
nvidia-smi -l 5

# Training monitor script
bash scripts/runpod_monitor.sh

# Docker logs
docker logs -f <container-id>
```

### Remote Monitoring (from macOS)

**TensorBoard**:
```bash
# SSH tunnel
ssh -L 6006:localhost:6006 root@<runpod-ip> -p <ssh-port>
# Open: http://localhost:6006
```

**WandB Dashboard**:
```
https://wandb.ai/<username>/ultrashape-collectibles
```

**Logs via SSH**:
```bash
ssh root@<runpod-ip> -p <ssh-port> "tail -f /workspace/UltraShape-Training/logs/*.log"
```

---

## Cost Optimization

### Estimate Costs

**A40 (48GB) - Testing**:
- Processing: 100 models × 0.5hr × $0.70/hr = $0.35
- Training: 10 epochs × 2hr × $0.70/hr = $14.00
- **Total**: ~$15 for pipeline testing

**H100 (80GB) - Production**:
- Processing: 30,000 models × 10hr × $3.00/hr = $30.00
- Training: 100 epochs × 50hr × $3.00/hr = $150.00
- **Total**: ~$180 for full training

### Cost-Saving Tips

1. **Use Spot Instances**: 50-70% cheaper, can be interrupted
2. **Stop When Idle**: Stop instance between processing/training steps
3. **Persistent Volumes**: Don't lose data when stopping
4. **Monitor Costs**: Set WandB/email alerts for training completion
5. **Batch Upload**: Upload all data at once to minimize transfer time

---

## Troubleshooting

### GPU Not Detected

```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, check RunPod instance has GPU enabled
```

### Out of Memory (OOM)

```bash
# Reduce batch size in config
vim configs/train_dit_refine.yaml
# Set: batch_size: 1

# Or use gradient accumulation
# Set: gradient_accumulation_steps: 4
```

### Data Processing Hangs

```bash
# Reduce workers
docker run ... --num-workers 4  # Instead of 16

# Check Blender
docker run --rm ultrashape-processing:latest blender --version
```

### Training Diverges

Check WandB/TensorBoard for:
- Loss spiking: Reduce learning rate
- NaN values: Check data quality, reduce batch size
- No improvement: Increase training steps, check data splits

### Network Issues

```bash
# Test internet from container
docker run --rm ultrashape-training:latest curl -I https://huggingface.co

# Check RunPod firewall settings
```

---

## Best Practices

1. **Use tmux**: Keep training running when SSH disconnects
2. **Monitor Costs**: Track spending in RunPod dashboard
3. **Save Checkpoints**: Configure checkpoint saving every N steps
4. **Test First**: Always validate on small subset before full training
5. **Version Control**: Commit config changes before training
6. **Backup Checkpoints**: Copy important checkpoints to cloud storage
7. **Document Settings**: Note hyperparameters in WandB config

---

## Support

- **RunPod Docs**: https://docs.runpod.io
- **UltraShape Issues**: See main README
- **WandB Docs**: https://docs.wandb.ai

## Next Steps

After training completes:
1. Evaluate model quality on validation set
2. Deploy inference API: `docker run ... ultrashape-inference`
3. Test inference with sample inputs
4. Consider fine-tuning hyperparameters for another run
