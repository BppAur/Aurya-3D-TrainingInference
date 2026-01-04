#!/usr/bin/env python3
"""Validate processed data structure matches UltraShape expectations."""
import json
import sys
from pathlib import Path
from PIL import Image

def validate_structure(output_dir):
    """
    Validate that the processed data matches UltraShape's expected format.

    Args:
        output_dir: Path to the output directory from processing

    Returns:
        List of error messages (empty if all valid)
    """
    output_dir = Path(output_dir)
    errors = []
    warnings = []

    print(f"Validating data in: {output_dir}")
    print("=" * 60)

    # Check render.json exists
    render_json = output_dir / "render.json"
    if not render_json.exists():
        errors.append("❌ render.json not found")
        return errors  # Can't continue without this

    try:
        render_map = json.load(open(render_json))
    except json.JSONDecodeError as e:
        errors.append(f"❌ render.json is not valid JSON: {e}")
        return errors

    if not isinstance(render_map, dict):
        errors.append(f"❌ render.json must be a dictionary, got {type(render_map)}")
        return errors

    print(f"✓ render.json found with {len(render_map)} models\n")

    # Check each model
    for idx, (model_id, render_base) in enumerate(render_map.items(), 1):
        print(f"[{idx}/{len(render_map)}] Checking {model_id}...")

        # Check watertight mesh
        watertight = output_dir / "watertight" / f"{model_id}.obj"
        if not watertight.exists():
            errors.append(f"❌ Watertight mesh missing: {watertight}")
        else:
            print(f"  ✓ Watertight mesh exists")

        # Check render base is a string (not list)
        if not isinstance(render_base, str):
            errors.append(f"❌ render.json value must be string path, got {type(render_base)}: {model_id}")
            continue

        # Check renders directory structure
        # Expected: {render_base}/{model_id}/rgba/
        rgba_dir = output_dir / render_base / model_id / "rgba"
        if not rgba_dir.exists():
            errors.append(f"❌ RGBA directory missing: {rgba_dir}")
            continue
        else:
            print(f"  ✓ RGBA directory exists: {rgba_dir}")

        # Check 16 views
        view_count = 0
        for i in range(16):
            img_path = rgba_dir / f"{i:03d}.png"
            if not img_path.exists():
                errors.append(f"❌ Missing view {i:03d}.png for {model_id}")
            else:
                view_count += 1
                # Check RGBA format
                try:
                    img = Image.open(img_path)
                    if img.mode != "RGBA":
                        errors.append(f"❌ Wrong format {img.mode} (expected RGBA): {img_path.name}")
                    if len(img.getbands()) != 4:
                        errors.append(f"❌ Wrong channels {len(img.getbands())} (expected 4): {img_path.name}")

                    # Check resolution (should be 1024x1024)
                    if img.size != (1024, 1024):
                        warnings.append(f"⚠️  Unexpected size {img.size} (expected 1024x1024): {img_path.name}")

                except Exception as e:
                    errors.append(f"❌ Error reading {img_path.name}: {e}")

        if view_count == 16:
            print(f"  ✓ All 16 views present and valid RGBA\n")
        else:
            print(f"  ✗ Only {view_count}/16 views found\n")

    # Check data_list
    print("Checking data_list...")
    data_list_dir = output_dir / "data_list"
    if not (data_list_dir / "train.json").exists():
        errors.append("❌ train.json not found")
    else:
        try:
            train_data = json.load(open(data_list_dir / "train.json"))
            if not isinstance(train_data, list):
                errors.append(f"❌ train.json must be a list, got {type(train_data)}")
            else:
                print(f"  ✓ train.json exists with {len(train_data)} models")
        except json.JSONDecodeError as e:
            errors.append(f"❌ train.json is not valid JSON: {e}")

    if not (data_list_dir / "val.json").exists():
        errors.append("❌ val.json not found")
    else:
        try:
            val_data = json.load(open(data_list_dir / "val.json"))
            if not isinstance(val_data, list):
                errors.append(f"❌ val.json must be a list, got {type(val_data)}")
            else:
                print(f"  ✓ val.json exists with {len(val_data)} models")
        except json.JSONDecodeError as e:
            errors.append(f"❌ val.json is not valid JSON: {e}")

    # Check samples directory exists (will be populated by sampling step)
    sample_dir = output_dir / "sample"
    if not sample_dir.exists():
        warnings.append("⚠️  sample/ directory not found (will be created during sampling step)")
    else:
        print(f"\n  ✓ sample/ directory exists")

    return errors, warnings


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Validate processed data structure")
    parser.add_argument("--output-dir", default="data/output", help="Output directory to validate")
    args = parser.parse_args()

    errors, warnings = validate_structure(args.output_dir)

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)

    if warnings:
        print(f"\n⚠️  WARNINGS ({len(warnings)}):")
        for warning in warnings:
            print(f"  {warning}")

    if errors:
        print(f"\n❌ ERRORS FOUND ({len(errors)}):")
        for error in errors:
            print(f"  {error}")
        print("\n❌ VALIDATION FAILED!")
        print("Please fix the errors above before proceeding to RunPod.")
        sys.exit(1)
    else:
        print("\n✅ ALL VALIDATION PASSED!")
        print("\nYour data structure is correct and matches UltraShape expectations!")
        print("\nNext steps:")
        print("1. Commit and push your code to Git")
        print("2. Deploy to RunPod following docs/STEP-BY-STEP-GUIDE.md")
        print("3. Run sampling step (GPU required)")
        print("4. Start training!")
        sys.exit(0)


if __name__ == "__main__":
    main()
