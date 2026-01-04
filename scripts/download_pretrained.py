#!/usr/bin/env python3
"""
Download pretrained UltraShape weights from Hugging Face.
"""
import argparse
from pathlib import Path
from huggingface_hub import hf_hub_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_pretrained_weights(output_dir: str, model_type: str = "dit"):
    """
    Download pretrained weights from Hugging Face.

    Args:
        output_dir: Directory to save weights
        model_type: 'vae' or 'dit'
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    repo_id = "infinith/UltraShape"

    if model_type == "dit":
        filename = "ultrashape_v1.pt"
    elif model_type == "vae":
        filename = "vae_weights.pt"  # Update with actual filename
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Downloading {filename} from {repo_id}...")

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        logger.info(f"Downloaded to: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"Failed to download {filename} from {repo_id}")
        logger.error(f"Error: {str(e)}")
        logger.error("Please check:")
        logger.error("  1. Your internet connection")
        logger.error("  2. The repository exists and is accessible")
        logger.error("  3. The filename is correct")
        logger.error("  4. You have sufficient disk space")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download pretrained UltraShape weights")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    parser.add_argument("--model-type", choices=["vae", "dit"], default="dit", help="Model type")
    args = parser.parse_args()

    download_pretrained_weights(args.output_dir, args.model_type)


if __name__ == "__main__":
    main()
