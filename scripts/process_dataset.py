#!/usr/bin/env python3
"""
Main dataset processing orchestration script.
Processes meshes through watertighting, rendering, and sampling pipeline.
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import subprocess
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get script directory for relative path resolution
SCRIPT_DIR = Path(__file__).parent.resolve()


def process_single_model(args_tuple):
    """Process a single model through the pipeline."""
    model_path, output_dir, num_views = args_tuple
    model_id = model_path.stem

    try:
        # Determine script paths (works in Docker and locally)
        # In Docker: /workspace/scripts/
        # Locally: relative to this script
        watertight_script = SCRIPT_DIR / "watertight_mesh.py"
        blender_script = SCRIPT_DIR / "blender_render.py"

        # Fallback to Docker paths if local scripts don't exist
        if not watertight_script.exists():
            watertight_script = Path("/workspace/scripts/watertight_mesh.py")
        if not blender_script.exists():
            blender_script = Path("/workspace/scripts/blender_render.py")

        # Step 1: Watertight processing
        watertight_dir = output_dir / "meshes"
        watertight_dir.mkdir(parents=True, exist_ok=True)
        watertight_path = watertight_dir / f"{model_id}.obj"

        cmd = [
            "python3", str(watertight_script),
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
            str(blender_script), "--",
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


def create_dataset_splits(metadata: List[Dict], output_dir: Path, train_ratio=0.9, seed=42):
    """Create train/val splits and save JSON files."""
    random.seed(seed)
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
