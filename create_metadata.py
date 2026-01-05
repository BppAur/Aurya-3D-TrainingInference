import json
from pathlib import Path

meshes_dir = Path("data/output/meshes")
data_list_dir = Path("data/output/data_list")
data_list_dir.mkdir(parents=True, exist_ok=True)

obj_files = list(meshes_dir.glob("*.obj"))
model_ids = [f.stem for f in obj_files]

print(f"Found {len(model_ids)} models")

train_split = int(len(model_ids) * 0.9)
train_ids = model_ids[:train_split]
val_ids = model_ids[train_split:]

with open(data_list_dir / "train.json", "w") as f:
    json.dump(train_ids, f, indent=2)

with open(data_list_dir / "val.json", "w") as f:
    json.dump(val_ids, f, indent=2)

print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
print("✅ Metadata created!")
