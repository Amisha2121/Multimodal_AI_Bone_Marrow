# Multi-Dataset Training Guide

## Overview

The `train_multi_dataset.py` script allows you to train the bone marrow cell classifier on **multiple datasets simultaneously**. This is useful for:

- **Increasing training data** when a single dataset is too small
- **Improving generalization** by training on diverse patient populations
- **Transfer learning** from related datasets (e.g., adult + pediatric leukemia)
- **Cross-institutional validation** by combining data from multiple hospitals

---

## Quick Start

### Single Dataset (Current Behavior)

```bash
python vision/train_multi_dataset.py --config configs/default.yaml
```

This works exactly like the original `train.py` - uses only the primary BoneMarrowWSI-PediatricLeukemia dataset.

### Two Datasets

```bash
python vision/train_multi_dataset.py --config configs/default.yaml \
    --additional-metadata data/external/dataset2_metadata.csv \
    --additional-splits data/external/dataset2_splits.json \
    --additional-crops data/external/dataset2_crops
```

### Three or More Datasets

```bash
python vision/train_multi_dataset.py --config configs/default.yaml \
    --additional-metadata dataset2.csv dataset3.csv dataset4.csv \
    --additional-splits dataset2_splits.json dataset3_splits.json dataset4_splits.json \
    --additional-crops dataset2_crops dataset3_crops dataset4_crops
```

---

## Dataset Requirements

Each additional dataset must have the **same structure** as the primary dataset:

### 1. **Metadata CSV** (`*_metadata.csv`)

Required columns:
- `cell_id` — Unique identifier for each cell
- `patient_id` — Patient/case identifier (for train/val/test splits)
- `mapped_cell_category` — Cell type from taxonomy (see below)
- `extraction_status` — Must be "valid" for training
- `crop_path` — Relative or absolute path to cell crop PNG

**Example:**
```csv
cell_id,patient_id,mapped_cell_category,extraction_status,crop_path,original_cell_label
cell_001,patient_A,myeloblast,valid,crops/cell_001.png,Myeloblast
cell_002,patient_A,lymphocyte,valid,crops/cell_002.png,Mature Lymphocyte
cell_003,patient_B,blast,valid,crops/cell_003.png,Blast
```

### 2. **Splits JSON** (`*_splits.json`)

Patient-level splits to prevent data leakage:

```json
{
  "train": ["patient_A", "patient_C", "patient_D"],
  "val":   ["patient_B", "patient_E"],
  "test":  ["patient_F", "patient_G"]
}
```

**Important:** 
- Each patient must appear in **exactly one split**
- Splits are patient-level, not cell-level
- Train/val/test ratio typically 70/15/15 or 80/10/10

### 3. **Cell Crops Directory**

PNG images of individual cells (224×224 recommended):

```
dataset2_crops/
├── cell_001.png
├── cell_002.png
├── cell_003.png
└── ...
```

---

## Cell Type Taxonomy

All datasets must use the **same 8-class taxonomy**:

| Index | Cell Category              | Clinical Context                  |
|-------|----------------------------|-----------------------------------|
| 0     | blast                      | Immature, undifferentiated        |
| 1     | mature_neutrophil          | Mature granulocyte (seg/band)     |
| 2     | immature_granulocyte       | Myelocyte, metamyelocyte, promyelo|
| 3     | lymphocyte                 | Mature lymphoid lineage           |
| 4     | erythroblast               | Red cell precursor                |
| 5     | monocyte                   | Mature monocytic lineage          |
| 6     | eosinophil                 | Mature eosinophilic granulocyte   |
| 7     | basophil                   | Mature basophilic granulocyte     |

**Mapping Your Labels:**

If your dataset uses different terminology, create a mapping:

```python
# Example mapping for external dataset
label_mapping = {
    "Myeloblast": "blast",
    "Blast": "blast",
    "Segmented Neutrophil": "mature_neutrophil",
    "Band": "mature_neutrophil",
    "Myelocyte": "immature_granulocyte",
    "Promyelocyte": "immature_granulocyte",
    "Metamyelocyte": "immature_granulocyte",
    "Lymphocyte": "lymphocyte",
    "Plasma Cell": "lymphocyte",  # Map rare types to closest match
    "Erythroblast": "erythroblast",
    "Orthochromatic Erythroblast": "erythroblast",
    "Monocyte": "monocyte",
    "Eosinophil": "eosinophil",
    "Basophil": "basophil",
}
```

Apply this mapping to the `mapped_cell_category` column in your metadata CSV.

---

## Preparing an External Dataset

### Step 1: Convert Images to Cell Crops

If you have whole slide images or multi-cell images:

1. Use the detector to identify individual cells
2. Extract 224×224 crops centered on each cell
3. Save as PNG files

```python
from detector.segment import detect_and_segment
from PIL import Image

def extract_cells(image_path, output_dir):
    img = Image.open(image_path)
    cells = detect_and_segment(img)
    
    for i, cell_bbox in enumerate(cells):
        x, y, w, h = cell_bbox
        crop = img.crop((x, y, x+w, y+h))
        crop = crop.resize((224, 224))
        crop.save(f"{output_dir}/cell_{patient_id}_{i:04d}.png")
```

### Step 2: Create Metadata CSV

```python
import pandas as pd

# Your data
cells_data = [
    {"cell_id": "cell_001", "patient_id": "A", "label": "Myeloblast", "path": "crops/cell_001.png"},
    {"cell_id": "cell_002", "patient_id": "A", "label": "Lymphocyte", "path": "crops/cell_002.png"},
    # ... more cells
]

# Map to taxonomy
label_mapping = { ... }  # See above
for cell in cells_data:
    cell["mapped_cell_category"] = label_mapping[cell["label"]]
    cell["extraction_status"] = "valid"
    cell["original_cell_label"] = cell["label"]

# Save CSV
df = pd.DataFrame(cells_data)
df.to_csv("dataset2_metadata.csv", index=False)
```

### Step 3: Create Patient Splits

```python
import json
from sklearn.model_selection import train_test_split

# Get unique patients
patients = df["patient_id"].unique().tolist()

# Split: 70% train, 15% val, 15% test
train_patients, test_patients = train_test_split(patients, test_size=0.15, random_state=42)
train_patients, val_patients = train_test_split(train_patients, test_size=0.176, random_state=42)  # 0.176 of 0.85 ≈ 0.15

splits = {
    "train": train_patients,
    "val": val_patients,
    "test": test_patients
}

with open("dataset2_splits.json", "w") as f:
    json.dump(splits, f, indent=2)
```

### Step 4: Verify Dataset

```bash
python vision/train_multi_dataset.py --config configs/default.yaml \
    --additional-metadata dataset2_metadata.csv \
    --additional-splits dataset2_splits.json \
    --additional-crops dataset2_crops \
    --skip-calibration  # Quick dry-run
```

Look for:
- ✅ All samples loaded successfully
- ✅ No missing crop files
- ✅ Reasonable class distribution
- ✅ Train/val/test splits have correct patient counts

---

## Training Output

### Console Output

```
======================================================================
Loading PRIMARY dataset for split 'train'
======================================================================
[train] Loaded 3847 valid cell samples from 75 patients.
✓ Primary dataset: 3847 samples

======================================================================
Loading 2 ADDITIONAL dataset(s)
======================================================================

[dataset_1] Loading from:
  Metadata: data/external/dataset2_metadata.csv
  Splits:   data/external/dataset2_splits.json
  Crops:    data/external/dataset2_crops
✓ [dataset_1] Added 1523 samples

[dataset_2] Loading from:
  Metadata: data/external/dataset3_metadata.csv
  Splits:   data/external/dataset3_splits.json
  Crops:    data/external/dataset3_crops
✓ [dataset_2] Added 892 samples

======================================================================
MERGED DATASET SUMMARY (train)
======================================================================
  primary............   3847 samples ( 59.7%)
  dataset_1..........   1523 samples ( 23.6%)
  dataset_2..........    892 samples ( 13.8%)
  TOTAL..............   6262 samples
======================================================================
```

### Saved Files

1. **`models/checkpoints/best_vit_lora_morphology_multi_dataset.pt`**  
   - Best model checkpoint
   - Includes metadata about all datasets used

2. **`models/checkpoints/taxonomy_multi_dataset.json`**  
   - Per-class cell counts across all datasets
   - Dataset source breakdown
   - Training configuration

3. **`models/checkpoints/calibration_multi_dataset/`**  
   - Temperature scaling calibration results
   - Reliability diagrams
   - Calibration metrics

---

## Best Practices

### 1. **Balance Dataset Sizes**

If one dataset is much larger than others, consider:

**Option A: Downsample the large dataset**
```python
# In your metadata CSV, randomly sample from large dataset
large_df = pd.read_csv("large_dataset.csv")
sampled_df = large_df.sample(n=5000, random_state=42)
sampled_df.to_csv("large_dataset_sampled.csv", index=False)
```

**Option B: Use weighted sampling** (already enabled in training)
```yaml
# configs/default.yaml
training:
  use_weighted_sampler: true  # Balances class frequencies, not dataset sources
```

### 2. **Verify Label Consistency**

Before training, manually review a sample from each dataset:

```python
import random
from PIL import Image

for dataset_name in ["primary", "dataset_1", "dataset_2"]:
    # Load 10 random blast cells from this dataset
    samples = [s for s in all_samples if s["class_code"] == "blast" and s["dataset_source"] == dataset_name]
    for sample in random.sample(samples, 10):
        img = Image.open(sample["image_path"])
        img.show()
        print(f"{dataset_name}: {sample['original_label']}")
```

Look for:
- ✅ Labels are consistent across datasets
- ❌ One dataset's "blast" looks like another's "lymphocyte"

### 3. **Monitor Per-Dataset Performance**

After training, evaluate on each dataset separately:

```python
from vision.evaluate import evaluate_predictions

# Separate predictions by dataset source
primary_preds = [pred for pred, sample in zip(all_preds, test_samples) if sample["dataset_source"] == "primary"]
primary_true = [true for true, sample in zip(all_true, test_samples) if sample["dataset_source"] == "primary"]

metrics_primary = evaluate_predictions(primary_true, primary_preds, MORPH_CLASSES)
print(f"Primary dataset F1: {metrics_primary['macro_f1']:.3f}")

# Repeat for each dataset
```

### 4. **Domain Adaptation (Advanced)**

If datasets have domain shift (different staining protocols, imaging equipment):

**Option A: Fine-tune in stages**
```bash
# Stage 1: Train on primary dataset
python vision/train.py --config configs/default.yaml

# Stage 2: Fine-tune with additional datasets
python vision/train_multi_dataset.py --config configs/finetune.yaml \
    --additional-metadata dataset2.csv \
    --additional-splits dataset2_splits.json \
    --additional-crops dataset2_crops
```

**Option B: Use domain-adversarial training** (future work)

---

## Troubleshooting

### Error: "Number of additional metadata files must match..."

**Cause:** Mismatch in number of arguments

**Fix:** Ensure equal counts:
```bash
# ❌ Wrong - 2 metadata but 1 splits
--additional-metadata data1.csv data2.csv \
--additional-splits splits1.json

# ✅ Correct - 2 of each
--additional-metadata data1.csv data2.csv \
--additional-splits splits1.json splits2.json \
--additional-crops crops1 crops2
```

### Error: "FileNotFoundError: Additional metadata not found"

**Cause:** Invalid path

**Fix:** Use absolute paths or paths relative to repo root:
```bash
# Absolute
--additional-metadata /full/path/to/data.csv

# Relative to repo root
--additional-metadata data/external/dataset2.csv
```

### Warning: "X cell crop files not found on disk"

**Cause:** `crop_path` in CSV doesn't match actual files

**Fix:** 
1. Check `crop_path` format (relative vs absolute)
2. Verify files exist: `ls data/external/dataset2_crops/*.png | wc -l`
3. Update CSV if paths changed

### Low accuracy on additional dataset

**Possible causes:**
1. **Label inconsistency** — Check manual review (see Best Practices #2)
2. **Domain shift** — Try fine-tuning approach (see Best Practices #4)
3. **Data quality** — Verify image quality, focus, staining

---

## Example: Adding MATEK-19 Dataset

The [MATEK-19 dataset](https://wiki.cancerimagingarchive.net/x/fZB7Aw) contains bone marrow cells:

```bash
# 1. Download and extract
wget https://...matek19.zip
unzip matek19.zip -d data/external/matek19

# 2. Convert their labels to our taxonomy
python scripts/convert_matek19.py --input data/external/matek19 --output data/external/matek19_processed

# 3. Train with both datasets
python vision/train_multi_dataset.py --config configs/default.yaml \
    --additional-metadata data/external/matek19_processed/metadata.csv \
    --additional-splits data/external/matek19_processed/splits.json \
    --additional-crops data/external/matek19_processed/crops
```

---

## Performance Expectations

| Scenario | Expected Improvement |
|----------|---------------------|
| **Adding 50% more data** (same quality) | +3-5% macro F1 |
| **Adding diverse patient populations** | +5-10% macro F1, better generalization |
| **Adding different staining/imaging** | +2-4% macro F1, more robust |
| **Adding low-quality data** | -2% to +1% macro F1, may hurt |

**Rule of thumb:** More data helps if:
- ✅ Similar quality and labeling standards
- ✅ Diverse patient demographics (age, ethnicity, disease subtypes)
- ✅ Consistent preprocessing (staining, imaging resolution)

More data can hurt if:
- ❌ Label inconsistency (different annotators, unclear definitions)
- ❌ Severe domain shift (H&E vs Wright-Giemsa staining)
- ❌ Low image quality (blur, artifacts, poor focus)

---

## Next Steps

1. **Prepare your external dataset** using the guide above
2. **Verify dataset format** with a dry-run
3. **Train multi-dataset model** with full epochs
4. **Evaluate per-dataset performance** to check for issues
5. **Deploy the best model** for inference

For questions or issues, check the troubleshooting section or open a GitHub issue.

---

## Additional Resources

- **Primary dataset:** [BoneMarrowWSI-PediatricLeukemia](https://www.cancerimagingarchive.net/)
- **Alternative datasets:**
  - [MATEK-19 Blood Cells](https://wiki.cancerimagingarchive.net/x/fZB7Aw)
  - [ALL Challenge 2019](https://homes.di.unimi.it/scotti/all/)
  - [ASH Image Bank](https://imagebank.hematology.org/)
- **Taxonomy standards:**
  - WHO Classification of Tumours of Haematopoietic and Lymphoid Tissues (2022)
  - ICC Consensus Guidelines (2022)

Happy training! 🚀
