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

    except FileNotFoundError as e:
        logger.error(f"Input file not found {input_path}: {e}")
        return False
    except PermissionError as e:
        logger.error(f"Permission denied accessing {input_path} or {output_path}: {e}")
        return False
    except (ValueError, RuntimeError) as e:
        logger.error(f"Mesh processing error for {input_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing {input_path}: {e}")
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
