# UltraShape Fine-Tuning Pipeline Design

**Date:** 2026-01-04
**Project:** UltraShape 1.0 Fine-Tuning for Hand-Sculpted Collectibles
**Dataset:** 30,000 high-quality ZBrush collectible models

---

## Executive Summary

Design for fine-tuning UltraShape DiT model on custom hand-sculpted collectibles dataset. Two-phase approach: testing on budget GPUs (A40/A5000) with subset data, then production training on H100 with full dataset.

**Goal:** Create specialized 3D refinement model that understands high-quality collectible sculpting style.

**Strategy:** Fine-tune DiT stage only (use pretrained VAE) for faster results and lower computational cost.

---

## System Architecture

### Pipeline Overview

```
Your ZBrush Models (30,000)
         ↓
    [Data Processing Container]
         ↓
    ┌────────────────────────┐
    │ 1. Mesh Export (.obj)  │
    │ 2. Watertight Process  │
    │ 3. Blender Rendering   │
    │ 4. Point Cloud Sampling│
    └────────────────────────┘
         ↓
    Processed Dataset
    (meshes + renders + samples)
         ↓
    [Training Container]
         ↓
    ┌────────────────────────┐
    │ Fine-tune DiT Model    │
    │ (pretrained VAE frozen)│
    └────────────────────────┘
         ↓
    Fine-tuned Model Checkpoint
         ↓
    [Inference Container]
         ↓
    High-Quality 3D Refinement
```

### Three Docker Containers

1. **Data Processing Container** - One-time dataset preparation
2. **Training Container** - GPU-accelerated model fine-tuning
3. **Inference Container** - Production mesh refinement service

---

## Data Preparation Pipeline

### Input Requirements
- **Format:** ZBrush models exported as .obj files
- **Quantity:** 30,000 models
- **Quality:** Hand-sculpted collectibles with fine details

### Processing Steps

#### Step 1: Mesh Watertighting
- **Tool:** PyMeshLab
- **Process:**
  - Fill holes in meshes
  - Fix non-manifold geometry
  - Ensure closed surfaces
  - Verify mesh integrity
- **Output:** Watertight .obj files
- **Time:** ~5-10 seconds per model

#### Step 2: Multi-View Rendering (Blender)
- **Automation:** Headless Blender with Python scripts
- **Camera Setup:** 4-8 views per model
  - Front, back, left, right views
  - Top and angled views
- **Settings:**
  - Resolution: 1024x1024 pixels
  - Lighting: Standard 3-point setup
  - Background: White/transparent
  - Format: PNG
- **Output:** `{model_id}_{view_angle}.png`
- **Time:** ~30 seconds per model (4 views)

#### Step 3: Point Cloud Sampling
- **Script:** UltraShape's `sampling.py`
- **Configuration:**
  - 163,840 points per model
  - Surface points + normals
  - SDF (Signed Distance Field) values
- **Output:** `.npz` files
- **Time:** ~20 seconds per model

### Dataset Organization

```
data/
├── meshes/              # Watertight .obj files
│   ├── model_0001.obj
│   ├── model_0002.obj
│   └── ...
├── renders/             # Multi-view images
│   ├── model_0001/
│   │   ├── view_0.png
│   │   ├── view_1.png
│   │   ├── view_2.png
│   │   └── view_3.png
│   └── ...
├── samples/             # Sampled point clouds (.npz)
│   ├── model_0001.npz
│   ├── model_0002.npz
│   └── ...
├── data_list/
│   ├── train.json      # 90% of models (~27,000)
│   └── val.json        # 10% of models (~3,000)
└── render.json         # Image path mappings
```

### Processing Time Estimates

**Full Dataset (30,000 models) on A40:**
- Watertighting: ~83 hours
- Rendering: ~250 hours
- Sampling: ~167 hours
- **Total: ~500 hours (~21 days)**

**Test Subset (1,000 models):**
- **Total: ~17 hours**

**Optimization:** Can parallelize across multiple CPU cores for rendering and processing.

---

## Training Container (Dockerfile.training)

### Base Configuration

**Base Image:** `nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`

**Core Dependencies:**
- Python 3.10
- PyTorch 2.5.1 with CUDA 12.1
- DeepSpeed (distributed training)
- Flash Attention 2.8.3 (memory efficiency)
- PyTorch3D (mesh operations)
- All requirements from `requirements.txt`

### Training Configuration

#### Hardware Profiles

**Phase 1 - Testing (A40/A5000):**
- GPU: 1x A40 (48GB VRAM) or A5000 (24GB VRAM)
- System RAM: 64GB
- Storage: 500GB-1TB SSD
- Network: High-speed for dataset access

**Phase 2 - Production (H100):**
- GPU: 1x H100 (80GB VRAM)
- System RAM: 128GB
- Storage: 1-2TB NVMe SSD
- Network: High-speed for checkpoint sync

#### Training Hyperparameters

**Model Configuration:**
- Base Model: Pretrained UltraShape DiT
- VAE: Frozen (use pretrained weights)
- DiT: Fine-tune all layers
- Latent Tokens: 8,192
- Hidden Size: 2,048
- Depth: 21 transformer blocks
- Attention Heads: 16

**Optimization:**
- Batch Size: 1 per GPU
- Gradient Accumulation: 4 steps (effective batch size = 4)
- Learning Rate: 1e-5 (base)
- Warmup Steps: 500
- LR Schedule: Cosine with warmup
- Optimizer: AdamW (β1=0.9, β2=0.99, eps=1e-6)
- Weight Decay: 1e-2
- Gradient Clipping: 1.0 (norm)

**Memory Optimization:**
- Mixed Precision: BF16
- Gradient Checkpointing: Enabled
- DeepSpeed ZeRO Stage 2
- Flash Attention: Enabled

**Training Duration:**
- Test Phase: 10,000-20,000 steps
- Full Fine-tune: 50,000-100,000 steps
- Validation: Every 1,000 steps
- Checkpointing: Every 2,500 steps

#### Performance Estimates

**A40 (48GB):**
- Speed: ~3-5 seconds per step
- 50k steps: ~4-8 days
- 100k steps: ~8-15 days

**H100 (80GB):**
- Speed: ~1-2 seconds per step
- 50k steps: ~1.5-3 days
- 100k steps: ~3-6 days

### Container Features

**Persistent Volumes:**
- `/workspace/data` - Dataset mount
- `/workspace/checkpoints` - Model checkpoints
- `/workspace/outputs` - Training outputs
- `/workspace/logs` - TensorBoard/WandB logs

**Monitoring:**
- TensorBoard for local monitoring
- WandB for cloud-based tracking
- Automatic checkpoint resumption
- Health checks and logging

**Scripts:**
- `train.sh` - Main training entry point
- `scripts/train_deepspeed.sh` - DeepSpeed launcher
- Configuration: `configs/train_dit_refine.yaml`

---

## Inference Container (Dockerfile.inference)

### Base Configuration

**Base Image:** `nvidia/cuda:12.1.0-runtime-ubuntu22.04` (lighter than devel)

**Core Dependencies:**
- Python 3.10
- PyTorch 2.5.1 with CUDA 12.1
- Core UltraShape modules only
- FastAPI for REST API
- Minimal inference dependencies (no training libs)

### Inference Pipeline

```
Input: Coarse mesh (.glb/.obj) + Reference image
         ↓
    [Image Preprocessing]
    - Resize to 1022x1022
    - Normalize with mean/std
         ↓
    [Load Fine-tuned DiT Model]
    - VAE: Pretrained (frozen)
    - DiT: Your fine-tuned weights
         ↓
    [VAE Encoding]
    - Encode coarse mesh to latents
         ↓
    [DiT Refinement]
    - 50 denoising steps
    - Conditioned on reference image
         ↓
    [VAE Decoding]
    - Decode to high-res point cloud
    - Query voxel grid (128³ resolution)
         ↓
    [Marching Cubes]
    - Extract mesh surface
    - Resolution: 512³
         ↓
Output: Refined high-quality .glb mesh
```

### REST API Endpoints

**Health & Status:**
- `GET /health` - Container health check
- `GET /status` - Current system status
- `GET /metrics` - Performance metrics

**Inference:**
- `POST /refine` - Submit mesh + image for refinement
  - Body: multipart/form-data
  - Fields: `mesh_file`, `image_file`, `options`
  - Returns: `job_id`
- `GET /jobs/{job_id}` - Check job status
- `GET /jobs/{job_id}/download` - Download refined mesh
- `GET /jobs/{job_id}/preview` - Preview image

**Monitoring:**
- `GET /logs?lines=100` - Recent log entries
- `GET /queue` - Processing queue status

**API Documentation:**
- Interactive docs: `http://{runpod-ip}:8000/docs`
- OpenAPI spec: `http://{runpod-ip}:8000/openapi.json`

### CLI Interface

Alternative command-line interface:
```bash
python infer_dit_refine.py \
    --ckpt /checkpoints/finetuned_model.pt \
    --image /input/reference.png \
    --mesh /input/coarse_mesh.glb \
    --output /output/refined.glb \
    --config configs/infer_dit_refine.yaml
```

### Performance Optimization

**Inference Acceleration:**
- Model compilation: `torch.compile()` (10-30% speedup)
- Half-precision (FP16) inference
- Efficient memory management
- Batch processing support

**Resource Requirements:**
- GPU: 1x RTX 4090 / A5000 (24GB) sufficient
- System RAM: 32GB
- Storage: 50GB for models + temp files

**Inference Speed:**
- A5000: ~30-60 seconds per mesh
- RTX 4090: ~20-40 seconds per mesh
- H100: ~10-20 seconds per mesh

---

## Monitoring & Logging Setup

### Training Monitoring

#### TensorBoard (Local Access)

**What it tracks:**
- Training loss curves
- Validation metrics
- Learning rate schedule
- System metrics (GPU utilization, memory)
- Sample mesh visualizations

**Access from macOS:**
```bash
# SSH tunnel to RunPod
ssh -L 6006:localhost:6006 root@runpod-instance

# Open browser
http://localhost:6006
```

#### WandB (Cloud - Primary Monitoring)

**What it tracks:**
- Real-time training metrics
- System performance (GPU, CPU, memory, network)
- Experiment comparison across runs
- Automatic checkpoint uploads
- Email/Slack alerts on issues

**Setup:**
```bash
# Add to container environment
WANDB_API_KEY=your_api_key_here
WANDB_PROJECT=ultrashape-collectibles
WANDB_ENTITY=your_username
```

**Access:**
- Dashboard: `https://wandb.ai/{username}/{project}`
- Mobile app available for iOS/Android

**Benefits:**
- Access from anywhere (macOS, mobile, etc.)
- Persistent history across runs
- Team collaboration features
- No SSH tunneling needed

### Inference Monitoring

**FastAPI Built-in Logging:**
- Request/response logging
- Error tracking
- Performance metrics

**Custom Logging:**
- Processing queue status
- GPU utilization
- Inference timing statistics
- Error rates and types

**Access Methods:**
- API endpoints: `/logs`, `/status`, `/metrics`
- Direct log files in container
- WandB integration for production metrics

---

## Implementation Phases

### Phase 1: Testing & Validation (A40/A5000)

**Objectives:**
- Validate entire pipeline end-to-end
- Test with subset of data (1,000 models)
- Verify training convergence
- Assess fine-tuning quality

**Steps:**

1. **Data Processing (Day 1)**
   - Export 1,000 models from ZBrush as .obj
   - Run data processing container
   - Generate watertight meshes
   - Render multi-view images
   - Sample point clouds
   - Create train/val split (900/100)

2. **Training Setup (Day 2)**
   - Download pretrained UltraShape weights
   - Configure training parameters for A40/A5000
   - Set up WandB monitoring
   - Launch training container

3. **Fine-Tuning (Days 3-7)**
   - Train for 10,000-20,000 steps
   - Monitor via WandB from macOS
   - Validate checkpoint quality
   - Adjust hyperparameters if needed

4. **Inference Testing (Day 8)**
   - Deploy inference container
   - Test on validation set
   - Compare with pretrained model
   - Assess quality improvement

5. **Validation (Days 9-10)**
   - Visual quality assessment
   - Quantitative metrics analysis
   - Decision: proceed to full training or adjust

**Expected Outcomes:**
- Working pipeline confirmation
- Initial quality assessment
- Identified optimization opportunities
- Go/no-go decision for Phase 2

### Phase 2: Production Training (H100)

**Objectives:**
- Train on full 30,000 model dataset
- Achieve production-quality fine-tuned model
- Deploy inference service

**Steps:**

1. **Data Processing (Weeks 1-3)**
   - Process all 30,000 models
   - Parallelize across multiple machines if needed
   - Quality control checks
   - Create final train/val split (27,000/3,000)

2. **Training Setup (Week 3)**
   - Migrate to H100 instance
   - Adjust batch size and optimization for H100
   - Configure production WandB project

3. **Fine-Tuning (Weeks 4-5)**
   - Train for 50,000-100,000 steps
   - Monitor convergence carefully
   - Regular validation checks
   - Checkpoint management

4. **Model Validation (Week 6)**
   - Comprehensive quality testing
   - Comparison with pretrained baseline
   - Hold-out test set evaluation
   - User acceptance testing

5. **Deployment (Week 7)**
   - Deploy final inference container
   - Production API setup
   - Documentation and handoff
   - Monitoring and maintenance plan

**Expected Outcomes:**
- Production-ready fine-tuned model
- Specialized in hand-sculpted collectibles
- Deployed inference service
- Complete documentation

---

## Docker Container Specifications

### 1. Data Processing Container

**File:** `Dockerfile.processing`

**Purpose:** One-time dataset preparation

**Key Components:**
- Blender (headless mode)
- PyMeshLab
- Python data processing scripts
- Multi-threaded processing support

**Runtime:**
- CPU-focused (GPU optional for rendering acceleration)
- High RAM (64GB+ recommended)
- Large storage for intermediate files

**Execution:**
```bash
docker run -v /path/to/models:/input \
           -v /path/to/output:/output \
           ultrashape-processing:latest \
           --num-workers 16 \
           --render-views 4
```

### 2. Training Container

**File:** `Dockerfile.training`

**Purpose:** GPU-accelerated DiT fine-tuning

**Key Components:**
- NVIDIA CUDA 12.1 + cuDNN 8
- PyTorch 2.5.1 with GPU support
- DeepSpeed for optimization
- Flash Attention 2
- TensorBoard + WandB
- Training scripts and configs

**Runtime:**
- GPU: A40/A5000 (testing) or H100 (production)
- High VRAM (24GB minimum, 48GB+ recommended)
- Fast storage for dataset access
- Network for checkpoint sync

**Execution:**
```bash
docker run --gpus all \
           -v /path/to/data:/workspace/data \
           -v /path/to/checkpoints:/workspace/checkpoints \
           -e WANDB_API_KEY=$WANDB_KEY \
           ultrashape-training:latest \
           bash train.sh
```

### 3. Inference Container

**File:** `Dockerfile.inference`

**Purpose:** Production mesh refinement service

**Key Components:**
- NVIDIA CUDA 12.1 runtime (lighter)
- PyTorch 2.5.1 inference mode
- FastAPI web framework
- Core UltraShape inference modules
- Optimized for low latency

**Runtime:**
- GPU: RTX 4090/A5000 (24GB)
- Moderate RAM (32GB)
- SSD for model weights

**Execution:**
```bash
docker run --gpus all \
           -v /path/to/checkpoints:/checkpoints \
           -p 8000:8000 \
           ultrashape-inference:latest
```

---

## Success Criteria

### Data Processing
- ✓ All 30,000 models successfully converted to watertight meshes
- ✓ Multi-view renders generated with consistent quality
- ✓ Point cloud samples valid and complete
- ✓ Dataset organization matches training requirements

### Training
- ✓ Training loss converges smoothly
- ✓ Validation metrics improve over baseline
- ✓ No overfitting (train/val gap acceptable)
- ✓ Checkpoints save and resume correctly
- ✓ Monitoring dashboards functional

### Inference
- ✓ API endpoints respond correctly
- ✓ Inference speed meets targets (< 60 seconds per mesh)
- ✓ Output mesh quality visually superior to input
- ✓ Fine details preserved from reference images
- ✓ Stable operation over extended periods

### Quality Assessment
- ✓ Fine-tuned model outperforms pretrained on collectibles
- ✓ Sculpting style and detail level preserved
- ✓ Consistent quality across different collectible types
- ✓ User acceptance from sculpting team

---

## Risk Mitigation

### Data Quality Risks
- **Risk:** Poor watertighting results in training issues
- **Mitigation:** Manual review of subset, quality checks, fallback processing methods

### Training Risks
- **Risk:** Training doesn't converge or overfits
- **Mitigation:** Start with small subset, monitor closely, adjust hyperparameters

### Resource Risks
- **Risk:** A40/A5000 insufficient memory for batch size
- **Mitigation:** Gradient accumulation, smaller batch size, gradient checkpointing

### Timeline Risks
- **Risk:** Data processing takes longer than estimated
- **Mitigation:** Parallel processing, cloud compute scaling, pipeline optimization

---

## Next Steps

1. **Review and approve this design**
2. **Create Docker containers:**
   - Dockerfile.processing
   - Dockerfile.training
   - Dockerfile.inference
3. **Create data processing scripts:**
   - Blender rendering automation
   - Watertight mesh processing
   - Dataset organization utilities
4. **Set up RunPod environment:**
   - Instance configuration
   - Volume storage setup
   - WandB account and API key
5. **Begin Phase 1 testing**

---

## References

- UltraShape Repository: https://github.com/PKU-YuanGroup/UltraShape-1.0
- UltraShape Paper: https://arxiv.org/pdf/2512.21185
- Hugging Face Models: https://huggingface.co/infinith/UltraShape
- Hunyuan3D-2.1: https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1
- LATTICE: https://arxiv.org/abs/2512.03052

---

**Document Version:** 1.0
**Last Updated:** 2026-01-04
**Status:** Design Complete - Awaiting Implementation
