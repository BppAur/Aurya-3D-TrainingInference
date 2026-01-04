# UltraShape Implementation Verification Report

**Date:** 2026-01-04
**Status:** ⚠️ CRITICAL ISSUES FOUND - REQUIRES FIXES

## Executive Summary

After detailed verification against the original UltraShape-1.0 repository, **several critical mismatches** were found between our Docker implementation and the expected data format. These MUST be fixed before training.

---

## Critical Issues Found

### 1. ❌ Incorrect Number of Render Views

**Expected:** 16 views per model (indices 000-015)
**Our Implementation:** 4 views per model
**Impact:** Data loader will fail with IndexError when trying to access views 4-15

**Code Evidence:**
```python
# From ultrashape/data/objaverse_dit.py line ~147
sel_idx = random.randint(0, 15)  # Expects 16 views!
img_path = f'{self.image_paths[obj_name]}/{os.path.basename(
    self.image_paths[obj_name])}/rgba/{sel_idx:03d}.png'
```

**Fix Required:**
- Update `process_dataset.py` line 121: `--num-views` default from 4 to 16
- Update Blender rendering to generate 16 views (evenly spaced 360° rotation)

---

### 2. ❌ Incorrect Image Directory Structure

**Expected:** `{base_path}/{basename}/rgba/000.png`
**Our Implementation:** `renders/{model_id}/view_0.png`
**Impact:** Data loader cannot find images

**Code Evidence:**
```python
# Data loader expects this structure:
# image_paths[obj_name] = "/path/to/renders/model123"
# Then constructs: "/path/to/renders/model123/model123/rgba/000.png"
```

**Fix Required:**
- Update Blender script to create `rgba/` subdirectory
- Rename output files from `view_0.png` to `000.png` format
- Create nested directory structure: `{model_id}/{model_id}/rgba/`

---

### 3. ❌ Incorrect render.json Format

**Expected:**
```json
{
  "model_001": "/workspace/data/renders/model_001",
  "model_002": "/workspace/data/renders/model_002"
}
```

**Our Implementation:**
```json
{
  "model_001": ["/workspace/data/renders/model_001/view_0.png", ...],
  "model_002": ["/workspace/data/renders/model_001/view_1.png", ...]
}
```

**Impact:** Data loader gets list instead of path string, crashes

**Fix Required:**
- Update `process_dataset.py` line 109-111 to create correct mapping
- Store base directory path, not list of individual files

---

### 4. ⚠️ Image Format Must Be RGBA PNG

**Expected:** RGBA PNG with 4 channels
**Our Implementation:** Likely RGB (needs verification)
**Impact:** Assertion error if not 4 channels

**Code Evidence:**
```python
# From data loader:
assert image.shape[2] == 4, f"Expected RGBA (4 channels), got {image.shape[2]}"
```

**Fix Required:**
- Update Blender rendering script to ensure alpha channel
- Set output format to RGBA explicitly

---

### 5. ⚠️ Surface Points Must Be Exactly 600,000

**Expected:** 600,000 surface points (300k uniform + 300k curvature)
**Our Implementation:** ✅ CORRECT (300k + 300k)
**Status:** No fix needed

**Verification:**
```python
# From data loader:
assert surface_og_n.shape[0] == 600000, f"assume that suface points = 30w uniform + 30w curvature..."
```

Our `sample_dataset.py` uses correct defaults (300k + 300k).

---

## Verification Against Original Repository

### Data Pipeline Comparison

| Step | Original UltraShape | Our Implementation | Status |
|------|-------------------|-------------------|--------|
| **Watertight Processing** | Custom processing | PyMeshLab | ✅ Compatible |
| **Rendering** | Blender (16 views, RGBA) | Blender (4 views, RGB?) | ❌ MISMATCH |
| **Directory Structure** | `{id}/{id}/rgba/NNN.png` | `{id}/view_N.png` | ❌ MISMATCH |
| **Sampling** | 300k+300k+400k | 300k+300k+400k | ✅ CORRECT |
| **data_list** | JSON array of IDs | JSON array of IDs | ✅ CORRECT |
| **render.json** | Dict of paths | Dict of file lists | ❌ MISMATCH |

### Config Files Comparison

| Config Parameter | Expected | Our Implementation | Status |
|-----------------|----------|-------------------|--------|
| `training_data_list` | `/workspace/data/data_list` | ✅ Fixed | ✅ CORRECT |
| `sample_pcd_dir` | `/workspace/data/sample` | ✅ Fixed | ✅ CORRECT |
| `image_data_json` | `/workspace/data/render.json` | ✅ Fixed | ✅ CORRECT |
| `vae from_pretrained` | Checkpoint path | ✅ Fixed | ✅ CORRECT |

### Dependencies Comparison

| Package | Original | Our Implementation | Status |
|---------|----------|-------------------|--------|
| `pymeshlab` | 2022.2.post3 | ✅ Fixed | ✅ CORRECT |
| `pytorch` | 2.5.1 + CUDA 12.1 | 2.5.1 + CUDA 12.1 | ✅ CORRECT |
| `pytorch3d` | Required | ✅ Included | ✅ CORRECT |
| `cubvh` | Required | ✅ Included | ✅ CORRECT |
| `deepspeed` | Implied | ✅ Included | ✅ CORRECT |

---

## Required Fixes (Priority Order)

### HIGH PRIORITY (Training Will Fail)

1. **Fix Blender Rendering Script**
   - Generate 16 views (not 4)
   - Create proper directory structure: `{model_id}/{model_id}/rgba/`
   - Output RGBA PNG format
   - Name files as `000.png`, `001.png`, ..., `015.png`

2. **Fix render.json Generation**
   - Change from list of files to base directory path
   - Format: `{"model_id": "/workspace/data/renders/model_id"}`

3. **Update process_dataset.py Default Views**
   - Change `--num-views` default from 4 to 16

### MEDIUM PRIORITY (Best Practices)

4. **Verify Image Alpha Channel**
   - Test Blender output has 4 channels (RGBA)
   - Add validation in processing script

5. **Add Data Validation Script**
   - Create script to verify data format before training
   - Check: directory structure, file counts, image channels, npz keys

---

## Recommended Next Steps

**Before proceeding with training:**

1. ✅ Apply HIGH PRIORITY fixes (estimated 1-2 hours)
2. ✅ Test with 10 sample models
3. ✅ Verify data loader can read processed data
4. ✅ Run 100 training steps to validate pipeline
5. ✅ Scale to 1000 models for full validation

**DO NOT** proceed with processing 30,000 models until these fixes are verified with a small subset.

---

## Files That Need Updates

### Must Update:
1. `scripts/blender_render.py` - Fix views count, directory structure, RGBA output
2. `scripts/process_dataset.py` - Fix default views, render.json format

### Should Verify:
3. `configs/train_dit_refine.yaml` - Already fixed, verify paths
4. `scripts/sample_dataset.py` - Already correct, no changes needed

---

## Positive Findings

### What's Already Correct ✅

1. **Sampling parameters** - 300k + 300k + 400k matches exactly
2. **Config file paths** - All updated to absolute `/workspace/` paths
3. **PyMeshLab version** - Fixed to 2022.2.post3
4. **Dependencies** - All required packages included
5. **DeepSpeed config** - Created with ZeRO-2 optimization
6. **Download script** - Handles HuggingFace `ultrashape_v1.pt` correctly
7. **Docker architecture** - Well-designed separation of concerns

---

## Test Plan

### Phase 1: Fix and Validate (10 Models)

```bash
# 1. Apply fixes
git commit -am "fix: correct rendering views and directory structure"

# 2. Process 10 test models
docker run ultrashape-processing:latest \
  --input-dir /input \
  --output-dir /output \
  --limit 10

# 3. Verify output structure
ls /output/renders/model_001/model_001/rgba/*.png | wc -l  # Should be 16
python -c "import json; print(json.load(open('/output/render.json')))"  # Should be dict of paths

# 4. Sample point clouds
docker run --gpus all ultrashape-training:latest \
  python scripts/sample_dataset.py \
  --input-dir /output/watertight \
  --output-dir /output/sample

# 5. Test data loading
python -c "
from ultrashape.data.objaverse_dit import ObjaverseDataModule
dm = ObjaverseDataModule(
    batch_size=1,
    training_data_list='/output/data_list',
    sample_pcd_dir='/output/sample',
    image_data_json='/output/render.json'
)
dm.setup()
batch = next(iter(dm.train_dataloader()))
print('Success! Data loaded correctly')
print(f'Batch keys: {batch.keys()}')
print(f'Image shape: {batch[\"image\"].shape}')  # Should be [B, 4, H, W]
"
```

### Phase 2: Scale Test (1000 Models)

Only proceed after Phase 1 passes completely.

### Phase 3: Production (30,000 Models)

Only proceed after Phase 2 validates quality.

---

## Conclusion

**Current Status:** Implementation has correct architecture but critical data format mismatches.

**Time to Fix:** ~2-3 hours of focused work

**Risk Level:** ⚠️ HIGH - Training will fail without fixes

**Confidence After Fixes:** 🟢 HIGH - All other components are correct

---

**Recommendation:** Apply fixes immediately, test with 10 models, then proceed confidently with full dataset.
