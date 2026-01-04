#!/usr/bin/env python3
"""
FastAPI server for UltraShape inference.
Provides REST API for mesh refinement given image + coarse mesh.
"""
import logging
from pathlib import Path
from typing import Optional
import tempfile
import os
import subprocess
import asyncio

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="UltraShape Inference API", version="1.0.0")

# Configuration from environment
TEMP_DIR = Path(os.getenv("TEMP_DIR", "/tmp/ultrashape_inference"))
TEMP_DIR.mkdir(parents=True, exist_ok=True)
INFERENCE_TIMEOUT = int(os.getenv("INFERENCE_TIMEOUT", "300"))
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
PORT = int(os.getenv("PORT", "8000"))

# Concurrency control - only 1 inference at a time to avoid GPU OOM
inference_semaphore = asyncio.Semaphore(1)

# Get script directory for absolute paths
SCRIPT_DIR = Path(__file__).parent.resolve()
WORKSPACE_DIR = SCRIPT_DIR.parent

def validate_path_in_temp_dir(file_path: Path) -> Path:
    """
    Validate that a path is within TEMP_DIR to prevent path traversal.

    Args:
        file_path: Path to validate

    Returns:
        Resolved absolute path

    Raises:
        ValueError: If path is outside TEMP_DIR
    """
    resolved = file_path.resolve()
    temp_resolved = TEMP_DIR.resolve()

    if not str(resolved).startswith(str(temp_resolved)):
        raise ValueError(f"Path traversal detected: {file_path}")

    return resolved

def run_inference(image_path: str, mesh_path: str, output_path: str, config_path: str, checkpoint_path: str) -> str:
    """
    Run UltraShape inference using the existing inference script.

    Args:
        image_path: Path to input image
        mesh_path: Path to coarse mesh
        output_path: Path for output refined mesh (target location)
        config_path: Path to config YAML
        checkpoint_path: Path to model checkpoint

    Returns:
        Path to generated refined mesh

    Raises:
        RuntimeError: If inference fails
    """
    # Validate all paths are absolute and exist (except output)
    image_path = str(Path(image_path).resolve())
    mesh_path = str(Path(mesh_path).resolve())
    config_path = str(Path(config_path).resolve())
    checkpoint_path = str(Path(checkpoint_path).resolve())

    if not Path(image_path).exists():
        raise RuntimeError("Input image file not found")
    if not Path(mesh_path).exists():
        raise RuntimeError("Input mesh file not found")
    if not Path(config_path).exists():
        raise RuntimeError("Config file not found")
    if not Path(checkpoint_path).exists():
        raise RuntimeError("Checkpoint file not found")

    # Use absolute path for inference script
    infer_script = SCRIPT_DIR / "infer_dit_refine.py"
    if not infer_script.exists():
        raise RuntimeError(f"Inference script not found: {infer_script}")

    # The actual script outputs to: output_dir/{image_basename}_refined.glb
    # We need to use a temporary output directory, then move the file
    temp_output_dir = Path(output_path).parent

    cmd = [
        "python3", str(infer_script),
        "--ckpt", checkpoint_path,
        "--image", image_path,
        "--mesh", mesh_path,
        "--config", config_path,
        "--output_dir", str(temp_output_dir)
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=INFERENCE_TIMEOUT,
            cwd=str(WORKSPACE_DIR)  # Run from workspace root
        )
        logger.info("Inference completed successfully")

        # The actual script outputs to: outputs/{image_basename}_refined.glb
        # Extract image basename to find actual output
        image_basename = Path(image_path).stem
        actual_output = temp_output_dir / f"{image_basename}_refined.glb"

        if not actual_output.exists():
            raise RuntimeError(f"Output file not created at expected location: {actual_output}")

        # Move to requested output location if different
        if str(actual_output) != str(output_path):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            actual_output.rename(output_path)

        return output_path

    except subprocess.TimeoutExpired:
        logger.error(f"Inference timed out after {INFERENCE_TIMEOUT}s")
        raise RuntimeError("Inference timeout - request took too long")
    except subprocess.CalledProcessError as e:
        # Sanitize error message - don't expose internal paths or raw stderr
        logger.error(f"Inference failed: {e.stderr}")
        raise RuntimeError("Inference failed - please check input files are valid")
    except Exception as e:
        logger.error(f"Unexpected error during inference: {e}")
        raise RuntimeError("Internal server error")

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/infer")
async def infer(
    image: UploadFile = File(..., description="Input RGB image"),
    coarse_mesh: UploadFile = File(..., description="Coarse mesh (GLB format)")
) -> FileResponse:
    """
    Perform mesh refinement inference.

    Args:
        image: Input image file (max 100MB)
        coarse_mesh: Coarse mesh file in GLB format (max 100MB)

    Returns:
        Refined mesh in GLB format
    """
    # Acquire semaphore to limit concurrent requests
    async with inference_semaphore:
        # Validate file types by extension
        if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Image must be PNG or JPEG")
        if not coarse_mesh.filename.lower().endswith('.glb'):
            raise HTTPException(status_code=400, detail="Mesh must be GLB format")

        # Create temporary files with validated paths
        temp_image = TEMP_DIR / f"input_{os.urandom(8).hex()}.png"
        temp_mesh = TEMP_DIR / f"coarse_{os.urandom(8).hex()}.glb"
        temp_output = TEMP_DIR / f"output_{os.urandom(8).hex()}.glb"

        # Validate paths are within TEMP_DIR (prevent path traversal)
        try:
            temp_image = validate_path_in_temp_dir(temp_image)
            temp_mesh = validate_path_in_temp_dir(temp_mesh)
            temp_output = validate_path_in_temp_dir(temp_output)
        except ValueError as e:
            logger.error(f"Path validation failed: {e}")
            raise HTTPException(status_code=400, detail="Invalid file path")

        try:
            # Save uploaded files with size validation
            with open(temp_image, "wb") as f:
                content = await image.read()
                if len(content) > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail=f"Image file too large (max {MAX_FILE_SIZE // (1024*1024)}MB)")
                f.write(content)

            with open(temp_mesh, "wb") as f:
                content = await coarse_mesh.read()
                if len(content) > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail=f"Mesh file too large (max {MAX_FILE_SIZE // (1024*1024)}MB)")
                f.write(content)

            # Get config and checkpoint paths (use absolute paths)
            config_path = os.getenv("CONFIG_PATH", str(WORKSPACE_DIR / "configs/infer_dit_refine.yaml"))
            checkpoint_path = os.getenv("CHECKPOINT_PATH", str(WORKSPACE_DIR / "checkpoints/ultrashape_v1.pt"))

            if not Path(checkpoint_path).exists():
                raise HTTPException(status_code=503, detail="Model checkpoint not available")
            if not Path(config_path).exists():
                raise HTTPException(status_code=503, detail="Configuration not available")

            # Run inference
            output_path = run_inference(
                str(temp_image),
                str(temp_mesh),
                str(temp_output),
                config_path,
                checkpoint_path
            )

            # Read the output file content before cleanup
            output_content = Path(output_path).read_bytes()

            # Return the refined mesh as a response
            from fastapi.responses import Response
            return Response(
                content=output_content,
                media_type="model/gltf-binary",
                headers={"Content-Disposition": "attachment; filename=refined_mesh.glb"}
            )

        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except RuntimeError as e:
            logger.error(f"Inference error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")
        finally:
            # Cleanup temporary files
            for temp_file in [temp_image, temp_mesh, temp_output]:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to cleanup {temp_file}: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
