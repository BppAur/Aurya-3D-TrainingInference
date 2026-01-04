# UltraShape Fine-Tuning: Step-by-Step Guide

Complete walkthrough for training UltraShape on your custom 3D model dataset using Docker and RunPod.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Phase 1: Local Preparation](#phase-1-local-preparation-macos)
- [Phase 2: RunPod Testing Setup](#phase-2-runpod-testing-setup-a40a5000)
- [Phase 3: Production Training](#phase-3-production-training-h100)
- [Phase 4: Inference Deployment](#phase-4-inference-deployment)
- [Monitoring & Troubleshooting](#monitoring--troubleshooting)

---

## Prerequisites

### On Your Local Machine (macOS)

- [ ] Git installed
- [ ] Docker Desktop installed and running
- [ ] SSH client (built into macOS)
- [ ] Your 3D models ready (OBJ format recommended)
- [ ] WandB account created (optional but recommended) - https://wandb.ai

### For RunPod

- [ ] RunPod account created - https://runpod.io
- [ ] Credit added to account
- [ ] SSH keys generated (if not already done)

**Generate SSH keys if needed:**
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub  # Copy this to RunPod SSH keys
```

---

## Phase 1: Local Preparation (macOS)

### Step 1.1: Clone and Setup Repository

```bash
# Clone the repository
cd ~/Documents/Projects/
git clone <your-repo-url> UltraShape-Training
cd UltraShape-Training

# Create local data directories
mkdir -p data/input
```

**✓ Checkpoint:** You should see the project structure with `docker/`, `scripts/`, `configs/` directories.

### Step 1.2: Prepare Your 3D Models

```bash
# Option A: Copy from your local directory
cp /path/to/your/models/*.obj data/input/

# Option B: Export from ZBrush
# 1. In ZBrush, select your model
# 2. Go to Tool → Export
# 3. Choose OBJ format
# 4. Save to data/input/ directory
# 5. Repeat for each model
```

**Model Requirements:**
- Format: OBJ, STL, or GLB (OBJ recommended)
- Quality: High-resolution exports from ZBrush work best
- Quantity: Start with 100 models for testing, then scale to your full 30,000

**✓ Checkpoint:** `ls data/input/*.obj` should show your model files.

### Step 1.3: Test Docker Setup Locally

```bash
# Verify Docker is running
docker --version
docker compose --version

# Validate Docker Compose configuration
docker compose config

# Build the processing container (CPU-only, safe on macOS)
docker compose build processing
```

**✓ Checkpoint:** Build completes without errors. You'll see "Successfully tagged ultrashape-processing:latest"

**Note:** GPU containers (training/inference) will build but cannot run on macOS. We'll test those on RunPod.

### Step 1.4: Create Environment Configuration

```bash
# Create your environment file from template
cp .env.example .env

# Edit with your details
nano .env
```

Add your WandB API key:
```bash
WANDB_API_KEY=your_actual_api_key_here
WANDB_PROJECT=ultrashape-collectibles
```

**Get your WandB API key:**
1. Go to https://wandb.ai/settings
2. Scroll to "API keys"
3. Copy your key

**✓ Checkpoint:** `cat .env` shows your configuration.

---

## Phase 2: RunPod Testing Setup (A40/A5000)

**Goal:** Validate the complete pipeline with a small subset (~100 models) before spending money on full training.

**Estimated Cost:** ~$15-20 total

### Step 2.1: Launch RunPod Instance

1. **Go to RunPod:** https://runpod.io/console/pods
2. **Click:** "Deploy" → "GPU Pod"
3. **Select GPU:**
   - **Recommended:** NVIDIA A40 (48GB)
   - **Alternative:** NVIDIA A5000 (24GB)
4. **Select Template:**
   - Choose "RunPod PyTorch" or "NVIDIA CUDA 12.1"
5. **Configure Storage:**
   - Container Disk: 50GB (minimum)
   - Volume Disk: 200GB (persistent storage)
   - Volume Mount Path: `/workspace`
6. **Expose Ports:**
   - Add port: `6006` (TensorBoard)
   - Add port: `8000` (Inference API)
7. **Add SSH Key:**
   - Paste your public key from `~/.ssh/id_ed25519.pub`
8. **Click:** "Deploy"

**Wait for:** Instance to start (Status: Running)

**✓ Checkpoint:** Instance shows "Running" status with SSH connection details.

### Step 2.2: Connect to RunPod Instance

```bash
# Copy SSH command from RunPod console (looks like this):
ssh root@<ip-address> -p <port> -i ~/.ssh/id_ed25519

# You should see the RunPod welcome message
```

**✓ Checkpoint:** You're logged into the RunPod instance, prompt shows `root@<hostname>`.

### Step 2.3: Run Automated Setup

```bash
# On RunPod instance:
cd /workspace
git clone <your-repo-url> UltraShape-Training
cd UltraShape-Training

# Run automated setup script
bash scripts/runpod_setup.sh
```

**Script will prompt you for:**
1. "Build which container?" → Type: `all` (builds all three containers)
2. "Enter WandB API key" → Paste your key (or press Enter to skip)

**This will take 15-30 minutes** to build all containers.

**✓ Checkpoint:** Script completes with "Setup Complete!" message and next steps shown.

### Step 2.4: Upload Test Dataset

**From your macOS terminal (new window):**

```bash
# Navigate to your local project
cd ~/Documents/Projects/UltraShape-Training

# Upload first 100 models for testing
# Replace <runpod-ip> and <ssh-port> with your instance details
scp -P <ssh-port> data/input/*.obj root@<runpod-ip>:/workspace/UltraShape-Training/data/input/ | head -n 100

# For large datasets, use rsync instead:
rsync -avz --progress -e "ssh -p <ssh-port>" \
  data/input/*.obj \
  root@<runpod-ip>:/workspace/UltraShape-Training/data/input/ \
  --max-size=100
```

**Alternative:** Upload via RunPod web interface if available.

**✓ Checkpoint:** On RunPod: `ls /workspace/UltraShape-Training/data/input/*.obj | wc -l` shows ~100 files.

### Step 2.5: Process Test Dataset

**Back on RunPod instance:**

```bash
cd /workspace/UltraShape-Training

# Start a tmux session (so processing continues if you disconnect)
tmux new -s processing

# Run data processing
docker run --rm \
  -v /workspace/UltraShape-Training/data/input:/input \
  -v /workspace/UltraShape-Training/data/output:/output \
  ultrashape-processing:latest \
  --input-dir /input \
  --output-dir /output \
  --num-workers 16 \
  --num-views 4 \
  --limit 100

# Detach from tmux: Press Ctrl+B, then D
# Reattach later: tmux attach -t processing
```

**This will take 30-60 minutes** for 100 models.

**✓ Checkpoint:** Processing completes. Check output:
```bash
ls data/output/watertight/*.glb | wc -l  # Should show ~100
ls data/output/renders/*/view_*.png | wc -l  # Should show ~400 (4 views × 100 models)
ls data/output/data_list/*.json  # Should show train.json and val.json
```

### Step 2.6: Download Pretrained Weights

```bash
# Download pretrained VAE (required for training)
docker run --rm \
  -v /workspace/UltraShape-Training/checkpoints:/workspace/checkpoints \
  ultrashape-training:latest \
  python3 scripts/download_pretrained.py \
  --output-dir /workspace/checkpoints \
  --model-type vae
```

**This takes 5-10 minutes** depending on internet speed.

**✓ Checkpoint:** `ls checkpoints/*.pt` shows downloaded weights.

### Step 2.7: Start Test Training

```bash
# Start training in a tmux session
tmux new -s training

# Load environment variables
source ~/.bashrc  # This loads WANDB_API_KEY from setup

# Start training
docker run --gpus all --rm \
  -v /workspace/UltraShape-Training/data/output:/workspace/data \
  -v /workspace/UltraShape-Training/checkpoints:/workspace/checkpoints \
  -v /workspace/UltraShape-Training/outputs:/workspace/outputs \
  -v /workspace/UltraShape-Training/logs:/workspace/logs \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e WANDB_PROJECT=ultrashape-test \
  -p 6006:6006 \
  --shm-size=16g \
  ultrashape-training:latest

# Detach: Ctrl+B, then D
```

**Training starts immediately.** You should see:
- GPU detection messages
- Dataset loading
- Training progress with loss values

**✓ Checkpoint:** Training is running without errors. You see regular log output with decreasing loss values.

### Step 2.8: Monitor Training

**Option 1: Real-time Monitor (on RunPod)**

```bash
# In a new tmux window or SSH session
tmux new -s monitor
bash scripts/runpod_monitor.sh

# Shows: GPU usage, containers, disk space, checkpoints, logs
# Refreshes every 5 seconds
```

**Option 2: TensorBoard (from macOS)**

```bash
# On your macOS, create SSH tunnel
ssh -L 6006:localhost:6006 root@<runpod-ip> -p <ssh-port>

# Keep this terminal open, then open browser:
# http://localhost:6006
```

**Option 3: WandB Dashboard (from anywhere)**

Go to: https://wandb.ai/your-username/ultrashape-test

You'll see:
- Loss curves
- Learning rate schedule
- GPU utilization
- Training images (if configured)

**✓ Checkpoint:** You can see training metrics updating in real-time via at least one monitoring method.

### Step 2.9: Evaluate Test Results

**Let training run for 2-3 hours** (or ~10 epochs), then evaluate:

```bash
# Check latest checkpoint
ls -lht outputs/dit_ultrashape/exp1_token8192/*.pt | head -n 5

# Check training logs
tail -n 50 logs/training.log

# Check WandB for:
# - Decreasing loss (should trend downward)
# - No NaN values
# - Reasonable GPU utilization (>70%)
```

**Expected Results:**
- Training loss decreases from ~0.5 to ~0.1-0.2
- No crashes or OOM errors
- Checkpoints saved every N steps
- TensorBoard shows smooth loss curves

**If everything looks good:** Pipeline is validated! ✅

**If there are issues:** See [Troubleshooting](#troubleshooting) section.

### Step 2.10: Stop Test Instance

Once you've validated the pipeline:

```bash
# On RunPod instance, stop training
tmux attach -t training
# Press Ctrl+C to stop training gracefully

# Exit SSH
exit
```

**On RunPod Console:**
1. Go to your pods
2. Click "Stop" (not Terminate - keeps your volume)
3. Volume with processed data and checkpoints is preserved

**Cost so far:** ~$10-15 for testing

---

## Phase 3: Production Training (H100)

**Goal:** Train on your full dataset (30,000 models) for production-quality results.

**Estimated Cost:** ~$180-200

### Step 3.1: Launch H100 Instance

1. **Go to RunPod:** https://runpod.io/console/pods
2. **Click:** "Deploy" → "GPU Pod"
3. **Select GPU:** NVIDIA H100 (80GB)
4. **Select Template:** Same as before
5. **Configure Storage:**
   - Container Disk: 50GB
   - **Volume Disk: 500GB** (for full dataset)
   - Volume Mount Path: `/workspace`
6. **Expose Ports:** 6006, 8000
7. **Add SSH Key:** Same as before
8. **Click:** "Deploy"

**✓ Checkpoint:** H100 instance running.

### Step 3.2: Setup H100 Instance

```bash
# SSH to H100 instance
ssh root@<h100-ip> -p <port> -i ~/.ssh/id_ed25519

# Run setup (faster this time since you know what to do)
cd /workspace
git clone <your-repo-url> UltraShape-Training
cd UltraShape-Training
bash scripts/runpod_setup.sh

# When prompted:
# - Build: all
# - WandB: <your-key>
```

**✓ Checkpoint:** Setup complete.

### Step 3.3: Upload Full Dataset

**From macOS (this will take several hours):**

```bash
# Use rsync for resumable upload
rsync -avz --progress \
  -e "ssh -p <h100-ssh-port>" \
  data/input/ \
  root@<h100-ip>:/workspace/UltraShape-Training/data/input/

# Or upload in batches:
# Batch 1: models 1-10000
# Batch 2: models 10001-20000
# Batch 3: models 20001-30000
```

**Alternative:** Upload to cloud storage first, then download on RunPod:

```bash
# On macOS - upload to S3/GCS
aws s3 sync data/input/ s3://your-bucket/ultrashape-models/

# On RunPod - download
aws s3 sync s3://your-bucket/ultrashape-models/ data/input/
```

**✓ Checkpoint:** `ls data/input/*.obj | wc -l` shows ~30,000 files.

### Step 3.4: Process Full Dataset

```bash
# Start processing in tmux
tmux new -s processing

# Process all models (will take 10-20 hours)
docker run --rm \
  -v /workspace/UltraShape-Training/data/input:/input \
  -v /workspace/UltraShape-Training/data/output:/output \
  ultrashape-processing:latest \
  --input-dir /input \
  --output-dir /output \
  --num-workers 32 \
  --num-views 4

# Detach: Ctrl+B, D
```

**Monitor progress:**
```bash
# Check processing logs
tmux attach -t processing

# Or check output periodically
watch -n 60 "ls data/output/watertight/*.glb | wc -l"
```

**✓ Checkpoint:** All 30,000 models processed successfully.

### Step 3.5: Download Pretrained Weights

```bash
docker run --rm \
  -v /workspace/UltraShape-Training/checkpoints:/workspace/checkpoints \
  ultrashape-training:latest \
  python3 scripts/download_pretrained.py \
  --output-dir /workspace/checkpoints \
  --model-type vae
```

**✓ Checkpoint:** Pretrained VAE weights downloaded.

### Step 3.6: Optimize Training Config for H100

```bash
# Edit training config for H100's larger memory
nano configs/train_dit_refine.yaml
```

**Update these settings:**

```yaml
dataset:
  params:
    batch_size: 4  # H100 can handle larger batches (was 1 on A40)

training:
  max_epochs: 100  # Full training
  gradient_accumulation_steps: 2

  # Enable mixed precision
  precision: "bf16"  # H100 supports BF16 for better numerical stability

  # Checkpoint saving
  save_every_n_steps: 1000

  # Validation
  val_check_interval: 0.25  # Check 4 times per epoch
```

**✓ Checkpoint:** Config updated and saved.

### Step 3.7: Start Production Training

```bash
# Start training in tmux
tmux new -s training

# Load WandB key
source ~/.bashrc

# Start production training
docker run --gpus all --rm \
  -v /workspace/UltraShape-Training/data/output:/workspace/data \
  -v /workspace/UltraShape-Training/checkpoints:/workspace/checkpoints \
  -v /workspace/UltraShape-Training/outputs:/workspace/outputs \
  -v /workspace/UltraShape-Training/logs:/workspace/logs \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e WANDB_PROJECT=ultrashape-production \
  -p 6006:6006 \
  --shm-size=32g \
  ultrashape-training:latest

# Detach: Ctrl+B, D
```

**Expected training time:** 50-100 hours (2-4 days) depending on:
- Dataset size
- Number of epochs
- Batch size
- Model complexity

**✓ Checkpoint:** Training starts and runs without errors.

### Step 3.8: Monitor Production Training

**Setup monitoring dashboard:**

```bash
# On RunPod, start monitor
tmux new -s monitor
bash scripts/runpod_monitor.sh
```

**From macOS, setup tunnels:**

```bash
# Terminal 1: TensorBoard tunnel
ssh -L 6006:localhost:6006 root@<h100-ip> -p <port>

# Terminal 2: Keep alive (prevents SSH timeout)
ssh -o ServerAliveInterval=60 root@<h100-ip> -p <port>
```

**Monitor these metrics:**
- **WandB:** Loss curves, learning rate, validation metrics
- **TensorBoard:** Detailed training graphs
- **GPU utilization:** Should be >85% on H100
- **Checkpoints:** New checkpoint every ~1000 steps

**✓ Checkpoint:** All monitoring tools working and showing progress.

### Step 3.9: Training Completion

**When training completes** (you'll see "Training finished" in logs):

```bash
# Find best checkpoint (usually the latest)
ls -lht outputs/dit_ultrashape/exp1_token8192/*.pt | head -n 5

# Download to local machine (optional)
scp -P <port> root@<h100-ip>:/workspace/UltraShape-Training/outputs/dit_ultrashape/exp1_token8192/final.pt ~/Downloads/
```

**✓ Checkpoint:** Training complete with final checkpoint saved.

---

## Phase 4: Inference Deployment

### Step 4.1: Deploy Inference Container

**On your H100 instance (or create new inference-only instance):**

```bash
cd /workspace/UltraShape-Training

# Stop training if still running
tmux attach -t training
# Ctrl+C

# Copy your trained checkpoint to the standard location
cp outputs/dit_ultrashape/exp1_token8192/final.pt checkpoints/ultrashape_v1.pt

# Start inference API
docker run --gpus all -d \
  --name ultrashape-api \
  -v /workspace/UltraShape-Training/checkpoints:/workspace/checkpoints \
  -v /workspace/UltraShape-Training/configs:/workspace/configs \
  -p 8000:8000 \
  -e CHECKPOINT_PATH=/workspace/checkpoints/ultrashape_v1.pt \
  -e CONFIG_PATH=/workspace/configs/infer_dit_refine.yaml \
  ultrashape-inference:latest

# Check it's running
docker ps
docker logs ultrashape-api
```

**✓ Checkpoint:** Container running, logs show "Application startup complete".

### Step 4.2: Test Inference API

**From macOS:**

```bash
# Create SSH tunnel for API
ssh -L 8000:localhost:8000 root@<runpod-ip> -p <port>

# Test health endpoint
curl http://localhost:8000/health
# Should return: {"status":"healthy"}

# Test inference
curl -X POST http://localhost:8000/infer \
  -F "image=@inputs/image/test.png" \
  -F "coarse_mesh=@inputs/coarse_mesh/test.glb" \
  -o refined_output.glb
```

**✓ Checkpoint:** Inference works and returns refined GLB file.

### Step 4.3: Production API Setup (Optional)

For permanent deployment:

1. **Keep RunPod pod running** with inference container
2. **Setup domain/proxy** if you want public access
3. **Add authentication** for security (modify api_server.py)
4. **Setup monitoring** for API usage

Or, for local use only:
- Keep H100 instance stopped when not needed
- Start it and run inference when you need to refine meshes
- Stop it again to save costs

---

## Monitoring & Troubleshooting

### Regular Monitoring Tasks

**Daily checks during training:**

```bash
# Check training is still running
tmux attach -t training

# Check GPU utilization
nvidia-smi

# Check latest checkpoint time
ls -lht outputs/dit_ultrashape/exp1_token8192/*.pt | head -n 1

# Check disk space
df -h /workspace
```

**Weekly checks:**

- Review WandB metrics for training quality
- Check cost in RunPod dashboard
- Backup important checkpoints to local/cloud storage

### Common Issues

#### Training Crashes with OOM

```bash
# Edit config to reduce batch size
nano configs/train_dit_refine.yaml
# Change: batch_size: 1
# Change: gradient_accumulation_steps: 8

# Restart training
```

#### Processing Hangs on Certain Models

```bash
# Check which model is stuck
docker logs <processing-container-id>

# Remove problematic model
rm data/input/problematic_model.obj

# Resume processing
```

#### Can't Access TensorBoard

```bash
# On macOS, check SSH tunnel is active
lsof -i :6006

# If not, recreate tunnel:
ssh -L 6006:localhost:6006 root@<runpod-ip> -p <port>
```

#### WandB Not Logging

```bash
# Check API key is set
echo $WANDB_API_KEY

# If empty, set it:
export WANDB_API_KEY=your_key
source ~/.bashrc
```

#### Inference API Returns Errors

```bash
# Check logs
docker logs ultrashape-api

# Common fixes:
# - Verify checkpoint path exists
# - Check GPU is available: nvidia-smi
# - Restart container: docker restart ultrashape-api
```

---

## Cost Summary

### Total Estimated Costs

**Testing Phase (A40):**
- Instance setup: $2
- Data processing (100 models): $3
- Test training (3 hours): $10
- **Subtotal:** ~$15

**Production Phase (H100):**
- Data processing (30,000 models): $30
- Full training (70 hours): $210
- Inference testing: $5
- **Subtotal:** ~$245

**Grand Total:** ~$260

**Cost Optimization:**
- Use spot instances: Save 50% (total: ~$130)
- Stop instances when not in use
- Process data on CPU-only instances (cheaper)
- Use smaller test dataset first to validate

---

## Next Steps After Training

1. **Evaluate Model Quality**
   - Test on validation set
   - Compare with original UltraShape weights
   - Visualize results in Blender/MeshLab

2. **Fine-tune if Needed**
   - Adjust learning rate
   - Train for more epochs
   - Modify dataset (add more views, better quality models)

3. **Production Deployment**
   - Deploy inference API permanently
   - Create automated mesh refinement pipeline
   - Integrate with your existing workflows

4. **Share Results** (Optional)
   - Upload weights to HuggingFace
   - Share training metrics on WandB
   - Document your process for others

---

## Getting Help

**Documentation:**
- Docker Usage: `docs/README-docker.md`
- RunPod Guide: `docs/RUNPOD-GUIDE.md`
- Design Doc: `docs/plans/2026-01-04-ultrashape-finetuning-design.md`

**Common Resources:**
- RunPod Docs: https://docs.runpod.io
- WandB Docs: https://docs.wandb.ai
- UltraShape Paper: https://arxiv.org/abs/...

**Troubleshooting Checklist:**
- [ ] GPU detected: `nvidia-smi` works
- [ ] Docker running: `docker ps` shows containers
- [ ] Data present: `ls data/input` shows models
- [ ] Checkpoints saved: `ls outputs/**/*.pt` shows files
- [ ] Monitoring active: WandB/TensorBoard accessible
- [ ] No disk space issues: `df -h /workspace` < 80%

---

**You're all set! Start with Phase 1 on your local machine, then move to RunPod for the actual training. Good luck!** 🚀
