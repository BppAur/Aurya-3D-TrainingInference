#!/usr/bin/env python3
"""
Download pretrained UltraShape weights from Hugging Face.
"""
import argparse
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_pretrained_weights(output_dir: str, model_type: str = "dit"):
    """
    Download pretrained weights from Hugging Face.

    The HuggingFace repository contains ultrashape_v1.pt which includes the full model.
    For compatibility with the training config, we:
    - For 'dit': Download as ultrashape_v1.pt
    - For 'vae': Download the same file as ultrashape_v1_vae.pt
    - For 'both': Download both versions (same file, different names)

    Args:
        output_dir: Directory to save weights
        model_type: 'vae', 'dit', or 'both'
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    repo_id = "infinith/UltraShape"
    source_filename = "ultrashape_v1.pt"

    logger.info(f"Downloading {source_filename} from {repo_id}...")

    try:
        # Download the main model file
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=source_filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        logger.info(f"Downloaded to: {local_path}")

        # Create appropriate copies/symlinks based on model_type
        dit_path = output_path / "ultrashape_v1.pt"
        vae_path = output_path / "ultrashape_v1_vae.pt"

        if model_type == "dit":
            if not dit_path.exists():
                shutil.copy2(local_path, dit_path)
                logger.info(f"Created DiT weights: {dit_path}")
        elif model_type == "vae":
            if not vae_path.exists():
                shutil.copy2(local_path, vae_path)
                logger.info(f"Created VAE weights: {vae_path}")
        elif model_type == "both":
            if not dit_path.exists():
                shutil.copy2(local_path, dit_path)
                logger.info(f"Created DiT weights: {dit_path}")
            if not vae_path.exists():
                shutil.copy2(local_path, vae_path)
                logger.info(f"Created VAE weights: {vae_path}")
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        return local_path
    except Exception as e:
        logger.error(f"Failed to download {source_filename} from {repo_id}")
        logger.error(f"Error: {str(e)}")
        logger.error("Please check:")
        logger.error("  1. Your internet connection")
        logger.error("  2. The repository exists and is accessible")
        logger.error("  3. The filename is correct")
        logger.error("  4. You have sufficient disk space")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Download pretrained UltraShape weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download for DiT training (creates ultrashape_v1.pt)
  python download_pretrained.py --model-type dit

  # Download for VAE (creates ultrashape_v1_vae.pt)
  python download_pretrained.py --model-type vae

  # Download both versions (recommended)
  python download_pretrained.py --model-type both
        """
    )
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    parser.add_argument("--model-type", choices=["vae", "dit", "both"], default="both",
                       help="Model type to download")
    args = parser.parse_args()

    download_pretrained_weights(args.output_dir, args.model_type)


if __name__ == "__main__":
    main()
