import os
from pathlib import Path
from tqdm import tqdm

meshes_dir = Path("data/output/meshes")
samples_dir = Path("data/output/samples")
samples_dir.mkdir(parents=True, exist_ok=True)

obj_files = list(meshes_dir.glob("*.obj"))
print(f"Found {len(obj_files)} meshes to sample")

for obj_file in tqdm(obj_files, desc="Sampling point clouds"):
    model_id = obj_file.stem
    output_path = samples_dir / f"{model_id}.npz"

    # Skip if already exists
    if output_path.exists():
        print(f"Skipping {model_id} (already exists)")
        continue

    cmd = f'python3 scripts/sampling.py --mesh_path "{obj_file}" --output_path "{output_path}" --num_surface 600000 --num_space 400000'
    print(f"\nProcessing: {model_id}")
    os.system(cmd)

print("\nDone! Sampled point clouds saved to data/output/samples/")
