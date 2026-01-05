import json
from pathlib import Path

samples_dir = Path("data/output/samples")
data_list_dir = Path("data/output/data_list")
data_list_dir.mkdir(parents=True, exist_ok=True)

  # Get model IDs from .npz files
npz_files = list(samples_dir.glob("*.npz"))
model_ids = [f.stem for f in npz_files]

print(f"Found {len(model_ids)} sampled models:")
for mid in model_ids:
    print(f"  - {mid}")

  # 90/10 train/val split
train_split = int(len(model_ids) * 0.9)
train_ids = model_ids[:train_split]
val_ids = model_ids[train_split:]

print(f"\nTrain: {len(train_ids)} models")
print(f"Val: {len(val_ids)} models")

  # Save metadata
with open(data_list_dir / "train.json", "w") as f:
    json.dump(train_ids, f, indent=2)

with open(data_list_dir / "val.json", "w") as f:
    json.dump(val_ids, f, indent=2)

print("\nMetadata updated!")

