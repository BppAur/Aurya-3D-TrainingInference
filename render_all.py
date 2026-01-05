import os
from pathlib import Path
from tqdm import tqdm

meshes_dir = Path("data/output/meshes")
renders_dir = Path("data/output/renders")

obj_files = list(meshes_dir.glob("*.obj"))
print(f"Found {len(obj_files)} OBJ files to render")

for obj_file in tqdm(obj_files, desc="Rendering"):
    model_id = obj_file.stem
    render_output = renders_dir / model_id / model_id
    render_output.mkdir(parents=True, exist_ok=True)
    
    cmd = f'blender --background --python scripts/blender_render.py -- --mesh "{obj_file}" --output "{render_output}" --views 16'
    print(f"\nRendering {model_id}...")
    os.system(cmd)

print(f"\n✅ Done! Rendered {len(obj_files)} models")
print(f"Output: {renders_dir}")
