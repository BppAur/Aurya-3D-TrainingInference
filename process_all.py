import os
from pathlib import Path
from tqdm import tqdm

input_dir = Path("data/input")
output_dir = Path("data/output")
output_dir.mkdir(parents=True, exist_ok=True)
meshes_dir = output_dir / "meshes"
meshes_dir.mkdir(parents=True, exist_ok=True)

models = list(input_dir.glob("*.stl")) + list(input_dir.glob("*.obj"))
print(f"Found {len(models)} models")

for model_path in tqdm(models, desc="Processing"):
    model_id = model_path.stem
    output_path = meshes_dir / f"{model_id}.obj"
    cmd = f'python3 scripts/watertight_simple.py "{model_path}" "{output_path}"'
    os.system(cmd)

print(f"Done! Processed {len(models)} models")
print(f"Output: {meshes_dir}")
