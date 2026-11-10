"""
Create splits.json for Kaggle dataset from kaggle_manifest.csv
"""
import json
import pandas as pd
from pathlib import Path

# Load kaggle manifest
kaggle_dir = Path("data/kaggle")
manifest = pd.read_csv(kaggle_dir / "kaggle_manifest.csv")

print(f"Total cells in kaggle manifest: {len(manifest)}")
print(f"\nSplit distribution:")
print(manifest['split'].value_counts())

# Get unique patient IDs per split
splits = {
    "train": sorted(manifest[manifest['split'] == 'train']['patient_id'].unique().tolist()),
    "val": sorted(manifest[manifest['split'] == 'val']['patient_id'].unique().tolist()),
    "test": sorted(manifest[manifest['split'] == 'test']['patient_id'].unique().tolist())
}

print(f"\nPatient counts:")
print(f"  Train: {len(splits['train'])} patients")
print(f"  Val:   {len(splits['val'])} patients")
print(f"  Test:  {len(splits['test'])} patients")

# Save splits.json
output_path = kaggle_dir / "splits.json"
with open(output_path, 'w') as f:
    json.dump(splits, f, indent=2)

print(f"\n✓ Created: {output_path}")
