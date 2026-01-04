#!/usr/bin/env python3
"""
Point cloud sampling script for UltraShape training.
This is a GPU-based operation that should run AFTER watertighting and rendering.

Usage:
  # Single GPU
  python scripts/sample_dataset.py --input-dir data/meshes --output-dir data/sample --num-gpus 1

  # Multi-GPU (distributed)
  python scripts/sample_dataset.py --input-dir data/meshes --output-dir data/sample --num-gpus 4
"""
import argparse
import json
import logging
from pathlib import Path
import sys

# Add parent directory to path to import sampling module
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.sampling import process_mesh_directory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mesh_json(input_dir: Path, output_path: Path):
    """
    Create a JSON file listing all mesh files for sampling.

    Args:
        input_dir: Directory containing watertight .obj files
        output_path: Path to save the JSON file
    """
    mesh_files = list(input_dir.glob("*.obj"))
    mesh_paths = [str(f.absolute()) for f in mesh_files]

    logger.info(f"Found {len(mesh_paths)} mesh files in {input_dir}")

    with open(output_path, "w") as f:
        json.dump(mesh_paths, f, indent=2)

    logger.info(f"Saved mesh list to {output_path}")
    return mesh_paths


def main():
    parser = argparse.ArgumentParser(
        description="Sample point clouds from watertight meshes (GPU-based)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script performs GPU-based point cloud sampling using PyTorch3D and cuBVH.
It should be run AFTER watertighting and rendering, but BEFORE training.

Workflow:
  1. Process Dataset (CPU): watertighting + rendering
  2. Sample Dataset (GPU): point cloud sampling  <- This script
  3. Train (GPU): model training

Examples:
  # Sample with single GPU
  python scripts/sample_dataset.py \\
    --input-dir /workspace/data/meshes \\
    --output-dir /workspace/data/sample \\
    --num-gpus 1

  # Sample with multiple GPUs (faster)
  python scripts/sample_dataset.py \\
    --input-dir /workspace/data/meshes \\
    --output-dir /workspace/data/sample \\
    --num-gpus 4 \\
    --batch-size 1 \\
    --num-workers 8

  # Custom sampling parameters
  python scripts/sample_dataset.py \\
    --input-dir /workspace/data/meshes \\
    --output-dir /workspace/data/sample \\
    --surface-uniform-samples 300000 \\
    --surface-curvature-samples 300000 \\
    --space-samples 400000
        """
    )

    # Required arguments
    parser.add_argument("--input-dir", required=True, help="Directory with watertight .obj files")
    parser.add_argument("--output-dir", required=True, help="Output directory for sampled point clouds")

    # Sampling parameters
    parser.add_argument("--surface-uniform-samples", type=int, default=300000,
                       help="Number of uniform samples on surface (default: 300000)")
    parser.add_argument("--surface-curvature-samples", type=int, default=300000,
                       help="Number of curvature-based samples on surface (default: 300000)")
    parser.add_argument("--space-samples", type=int, default=400000,
                       help="Number of samples in 3D space (default: 400000)")
    parser.add_argument("--noise-sigma", type=float, default=0.01,
                       help="Gaussian noise sigma (default: 0.01)")

    # GPU and parallelization
    parser.add_argument("--num-gpus", type=int, default=1,
                       help="Number of GPUs to use (-1 for all available, default: 1)")
    parser.add_argument("--batch-size", type=int, default=1,
                       help="Batch size per GPU (default: 1)")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of data loading workers (default: 4)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Validate input directory
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create temporary mesh JSON file
    mesh_json_path = output_dir / "_mesh_list.json"
    mesh_paths = create_mesh_json(input_dir, mesh_json_path)

    if len(mesh_paths) == 0:
        logger.error("No .obj files found in input directory")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Point Cloud Sampling Configuration:")
    logger.info(f"  Input directory: {input_dir}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Number of meshes: {len(mesh_paths)}")
    logger.info(f"  Surface uniform samples: {args.surface_uniform_samples}")
    logger.info(f"  Surface curvature samples: {args.surface_curvature_samples}")
    logger.info(f"  Space samples: {args.space_samples}")
    logger.info(f"  Noise sigma: {args.noise_sigma}")
    logger.info(f"  Number of GPUs: {args.num_gpus}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Workers: {args.num_workers}")
    logger.info("=" * 60)

    # Run sampling
    try:
        logger.info("Starting point cloud sampling (this may take a while)...")
        process_mesh_directory(
            mesh_json=str(mesh_json_path),
            output_dir=str(output_dir),
            data_type="mesh",
            surface_uniform_samples=args.surface_uniform_samples,
            surface_curvature_samples=args.surface_curvature_samples,
            space_samples=args.space_samples,
            noise_sigma=args.noise_sigma,
            num_gpus=args.num_gpus,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )

        logger.info("=" * 60)
        logger.info(f"Point cloud sampling complete!")
        logger.info(f"Output saved to: {output_dir}")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Sampling failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        # Clean up temporary mesh JSON
        if mesh_json_path.exists():
            mesh_json_path.unlink()


if __name__ == "__main__":
    main()
