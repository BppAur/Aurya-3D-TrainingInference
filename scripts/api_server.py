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

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="UltraShape Inference API", version="1.0.0")

# Create temporary directory for inference operations
TEMP_DIR = Path("/tmp/ultrashape_inference")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

def run_inference(image_path: str, mesh_path: str, output_path: str, config_path: str, checkpoint_path: str) -> str:
    """
    Run UltraShape inference using the existing inference script.

    Args:
        image_path: Path to input image
        mesh_path: Path to coarse mesh
        output_path: Path for output refined mesh
        config_path: Path to config YAML
        checkpoint_path: Path to model checkpoint

    Returns:
        Path to generated refined mesh

    Raises:
        RuntimeError: If inference fails
    """
    cmd = [
        "python3", "scripts/infer_dit_refine.py",
        "--ckpt", checkpoint_path,
        "--image", image_path,
        "--mesh", mesh_path,
        "--config", config_path,
        "--output_dir", str(Path(output_path).parent)
    ]

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        logger.info(f"Inference completed: {result.stdout}")

        if not Path(output_path).exists():
            raise RuntimeError(f"Output file not created: {output_path}")

        return output_path

    except subprocess.TimeoutExpired as e:
        logger.error(f"Inference timed out after 300s")
        raise RuntimeError("Inference timeout") from e
    except subprocess.CalledProcessError as e:
        logger.error(f"Inference failed: {e.stderr}")
        raise RuntimeError(f"Inference failed: {e.stderr}") from e
    except Exception as e:
        logger.error(f"Unexpected error during inference: {e}")
        raise RuntimeError(f"Inference error: {e}") from e

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
        image: Input image file
        coarse_mesh: Coarse mesh file in GLB format

    Returns:
        Refined mesh in GLB format
    """
    # Validate file types
    if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        raise HTTPException(status_code=400, detail="Image must be PNG or JPEG")
    if not coarse_mesh.filename.lower().endswith('.glb'):
        raise HTTPException(status_code=400, detail="Mesh must be GLB format")

    # Create temporary files for processing
    temp_image = TEMP_DIR / f"input_{os.urandom(8).hex()}.png"
    temp_mesh = TEMP_DIR / f"coarse_{os.urandom(8).hex()}.glb"
    temp_output = TEMP_DIR / f"output_{os.urandom(8).hex()}.glb"

    try:
        # Save uploaded files
        with open(temp_image, "wb") as f:
            content = await image.read()
            f.write(content)

        with open(temp_mesh, "wb") as f:
            content = await coarse_mesh.read()
            f.write(content)

        # Run inference
        config_path = os.getenv("CONFIG_PATH", "configs/infer_dit_refine.yaml")
        checkpoint_path = os.getenv("CHECKPOINT_PATH", "checkpoints/ultrashape_v1.pt")

        if not Path(checkpoint_path).exists():
            raise HTTPException(status_code=503, detail=f"Model checkpoint not found: {checkpoint_path}")
        if not Path(config_path).exists():
            raise HTTPException(status_code=503, detail=f"Config file not found: {config_path}")

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
    uvicorn.run(app, host="0.0.0.0", port=8000)
