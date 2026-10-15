# 🎓 ML Model Training Code Guide

## 📋 Overview

This document identifies all files related to ML model training in the project, organized by training pipeline and purpose.

---

## 🎯 Main Training Files

### 1. **Primary Training Module** ⭐⭐⭐

**File**: `vision/train.py`  
**Purpose**: Main training script for ViT-Small/16 + LoRA cell classifier  
**Lines of Code**: ~400  
**Key Functions**:
- `train_epoch()` - Single epoch training loop
- `validate()` - Validation evaluation
- `main()` - Entry point with full training pipeline

**What It Trains**:
- Model: ViT-Small/16 backbone
- Adaptation: LoRA (Low-Rank Adaptation)
- Task: 8-class cell morphology classification
- Dataset: 20,852 cell crops from 113 patients
- Splits: 79 train / 17 val / 17 test patients

**Key Components**:
```python
# Model initialization
model = BoneMarrowViT(
    num_classes=8,
    backbone="vit_small_patch16_224",
    lora_enabled=True,
    lora_r=8,
    lora_alpha=16
)

# Optimizer
optimizer = torch.optim.AdamW(
    trainable_params,
    lr=3e-4,
    weight_decay=1e-2
)

# Loss function
criterion = nn.CrossEntropyLoss(
    label_smoothing=0.1
)

# Training loop
for epoch in range(30):
    train_loss = train_epoch(model, train_loader, ...)
    val_loss, val_acc = validate(model, val_loader, ...)
    # Save best model
    if val_acc > best_acc:
        torch.save(model.state_dict(), "best_model.pt")
```

**How to Run**:
```bash
python vision/train.py
```

**Output**:
- Checkpoint: `models/checkpoints/best_vit_lora_morphology.pt`
- Logs: Console output with epoch metrics
- Optional: Weights & Biases logging

---

### 2. **Training Script (Step 4 in Pipeline)** ⭐⭐

**File**: `scripts/step4_train_cell_classifier.py`  
**Purpose**: Wrapper script for training as part of full pipeline  
**Lines of Code**: ~350  

**Relationship to `vision/train.py`**:
- This is an **alternative/wrapper** to `vision/train.py`
- Part of the 6-step pipeline (steps 1-6)
- Has similar functionality but integrated with pipeline

**Key Differences**:
- Includes more logging
- Part of `scripts/run_all.py` workflow
- Used for automated training runs

**How to Run**:
```bash
python scripts/step4_train_cell_classifier.py
```

Or as part of full pipeline:
```bash
python scripts/run_all.py --start-step 4 --stop-step 4
```

---

## 📁 Supporting Training Files

### 3. **Dataset Loader**

**File**: `vision/dataset.py`  
**Purpose**: PyTorch Dataset classes for loading cell crops  
**Key Classes**:
- `CellCropDataset` - Loads 224×224 cell crops
- `PatientAwareDataset` - Patient-level split handling

**Key Features**:
```python
class CellCropDataset(Dataset):
    def __init__(self, metadata_df, image_dir, transform):
        # Load cell metadata
        # Apply patient-level splits
        # Ensure zero overlap
    
    def __getitem__(self, idx):
        # Load 224×224 crop
        # Apply augmentation
        # Return (image, label)
```

**Data Augmentation** (during training):
- Random horizontal/vertical flips
- Rotation (±15°)
- Color jitter
- Normalization (ImageNet stats)

---

### 4. **Model Architecture**

**File**: `vision/model.py`  
**Purpose**: ViT-Small + LoRA model definition  
**Key Classes**:
- `BoneMarrowViT` - Main model class
- LoRA integration with PEFT library

**Architecture**:
```python
class BoneMarrowViT(nn.Module):
    def __init__(self, num_classes=8, lora_enabled=True):
        # Load pretrained ViT-Small/16
        self.backbone = timm.create_model(
            "vit_small_patch16_224",
            pretrained=True
        )
        
        # Apply LoRA
        if lora_enabled:
            lora_config = LoraConfig(
                r=8,                    # Rank
                lora_alpha=16,          # Scaling
                target_modules=["qkv"], # Q/K/V projections
                lora_dropout=0.05
            )
            self.backbone = get_peft_model(self.backbone, lora_config)
        
        # Classification head
        self.classifier = nn.Linear(384, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        return logits
```

**Trainable Parameters**:
- Frozen: ViT backbone (~22M params)
- Trainable: LoRA adapters (~0.3M params) + classifier head
- Total trainable: ~0.5M params (2% of full model)

---

### 5. **Taxonomy Definition**

**File**: `vision/taxonomy.py`  
**Purpose**: Cell morphology class definitions  
**Key Constants**:
```python
MORPH_CLASSES = [
    "Blast",
    "Lymphoid Series",
    "Granulocyte (Mature/Band)",
    "Granulocyte (Immature)",
    "Erythroid Precursor",
    "Monocyte",
    "Megakaryocyte / Platelet",
    "Other/Artifact"
]

IDX_TO_MORPH = {i: cls for i, cls in enumerate(MORPH_CLASSES)}
MORPH_TO_IDX = {cls: i for i, cls in enumerate(MORPH_CLASSES)}
```

---

### 6. **Evaluation Module**

**File**: `vision/evaluate.py`  
**Purpose**: Model evaluation metrics and analysis  
**Key Functions**:
- `evaluate_model()` - Compute metrics on test set
- `generate_confusion_matrix()` - Class-wise performance
- `compute_patient_level_metrics()` - Aggregate performance

**Metrics Computed**:
- Overall accuracy
- Per-class precision/recall/F1
- Confusion matrix
- Patient-level aggregation accuracy

---

### 7. **Evaluation Script (Step 5)**

**File**: `scripts/step5_evaluate_and_xai.py`  
**Purpose**: Evaluate trained model and generate XAI  
**Features**:
- Load best checkpoint
- Evaluate on test set
- Generate attention maps
- Save results

**How to Run**:
```bash
python scripts/step5_evaluate_and_xai.py
```

---

## 🔄 Training Pipeline (6 Steps)

The complete training pipeline is orchestrated by:

**File**: `scripts/run_all.py`  
**Purpose**: Run entire pipeline from data prep to evaluation

### Pipeline Steps:

1. **Step 1**: Delete incomplete crops  
   File: `scripts/step1_delete_incomplete_crops.py`

2. **Step 2**: Extract all cell crops  
   File: `scripts/step2_extract_all_crops.py`

3. **Step 3**: Verify crop integrity  
   File: `scripts/step3_verify_crops.py`

4. **Step 4**: Train cell classifier ⭐  
   File: `scripts/step4_train_cell_classifier.py`

5. **Step 5**: Evaluate and XAI  
   File: `scripts/step5_evaluate_and_xai.py`

6. **Step 6**: Patient-level baseline  
   File: `scripts/step6_patient_baseline.py`

**Run Full Pipeline**:
```bash
python scripts/run_all.py
```

**Run Specific Steps**:
```bash
# Train only
python scripts/run_all.py --start-step 4 --stop-step 4

# Evaluate only
python scripts/run_all.py --start-step 5 --stop-step 5
```

---

## 🎯 Quick Reference: Which File to Use?

### For Training:

| Scenario | File to Use |
|----------|-------------|
| **Train from scratch** | `python vision/train.py` |
| **Part of pipeline** | `python scripts/step4_train_cell_classifier.py` |
| **Full pipeline** | `python scripts/run_all.py` |
| **Evaluate model** | `python scripts/step5_evaluate_and_xai.py` |
| **Patient metrics** | `python scripts/step6_patient_baseline.py` |

### For Inference (Using Trained Model):

| Scenario | File to Use |
|----------|-------------|
| **Single image analysis** | `python pipeline/new_patient_engine.py` |
| **Web UI** | `streamlit run app/streamlit_app.py` |
| **Batch inference** | Use `vision/model.py` + custom loop |

---

## 📊 Training Configuration

**Configuration File**: `configs/default.yaml`

**Key Training Parameters**:
```yaml
training:
  epochs: 30
  batch_size: 64
  lr: 3.0e-4                    # Learning rate
  weight_decay: 1.0e-2          # AdamW weight decay
  label_smoothing: 0.1          # Cross-entropy smoothing
  precision: bf16               # Mixed precision (bf16/fp16/fp32)
  gradient_checkpointing: true  # Memory optimization
  use_weighted_sampler: true    # Handle class imbalance
  early_stopping_patience: 5    # Stop if no improvement

vision:
  backbone: vit_small_patch16_224
  pretrained: true
  lora:
    enabled: true
    r: 8                        # LoRA rank
    alpha: 16                   # LoRA alpha
    dropout: 0.05
    target_modules: ["qkv"]     # Apply LoRA to Q/K/V

dataset:
  image_size: 224
  augmentation: medium          # light/medium/heavy
  split:
    train: 0.70
    val: 0.15
    test: 0.15
```

**Modify Training**:
Edit `configs/default.yaml` to change hyperparameters.

---

## 🔬 Training Details

### Dataset:
- **Total cells**: 20,852 valid crops
- **Patients**: 113 (with consensus annotations)
- **Patient splits**: 79 train / 17 val / 17 test
- **Zero overlap**: Strict patient-level separation
- **Classes**: 8 morphology categories
- **Image size**: 224×224 RGB

### Model:
- **Architecture**: ViT-Small/16 (Vision Transformer)
- **Backbone params**: 22M (frozen)
- **LoRA params**: 0.3M (trainable)
- **Classifier head**: 0.2M (trainable)
- **Total trainable**: ~0.5M params (2.3%)

### Training:
- **Optimizer**: AdamW (lr=3e-4, weight_decay=1e-2)
- **Loss**: CrossEntropyLoss with label smoothing (0.1)
- **Batch size**: 64
- **Epochs**: 30 (with early stopping)
- **Mixed precision**: BF16 (on supported GPUs)
- **Augmentation**: Rotation, flip, color jitter
- **Class balancing**: Weighted sampler

### Hardware:
- **Minimum**: CPU (slow but works)
- **Recommended**: NVIDIA GPU with 8GB+ VRAM
- **Training time**: 
  - GPU: ~2-3 hours
  - CPU: ~10-15 hours

---

## 📈 Training Monitoring

### Console Output:
```
Epoch 1/30
Train Loss: 1.234
Val Loss: 1.456
Val Acc: 45.2%
Best model saved!

Epoch 2/30
Train Loss: 0.987
Val Loss: 1.123
Val Acc: 52.3%
Best model saved!
...
```

### Weights & Biases (Optional):
Set in `configs/default.yaml`:
```yaml
training:
  wandb_enabled: true
  wandb_project: bonemarrow-vision
```

Then add to `.env`:
```
WANDB_API_KEY=your_key_here
```

---

## 🎓 Additional Training Resources

### Cell Extraction (Pre-training):
- `vision/extract_cells.py` - Extract crops from DICOM
- `vision/run_extraction.py` - Batch extraction script
- Used to create 224×224 crops from original images

### Aggregation (Post-training):
- `vision/aggregate.py` - Patient-level aggregation
- Combines cell predictions into patient diagnosis

### Explainability:
- `vision/attention_rollout.py` - Attention map generation
- Visualizes what model looks at

---

## 🔧 Troubleshooting Training

### GPU Out of Memory:
```yaml
# Reduce batch size in configs/default.yaml
training:
  batch_size: 32  # or 16
  gradient_checkpointing: true
```

### Training Too Slow:
```yaml
# Use mixed precision
training:
  precision: bf16  # or fp16
```

### Overfitting:
```yaml
# Increase regularization
training:
  weight_decay: 5.0e-2
  label_smoothing: 0.15
  
dataset:
  augmentation: heavy
```

### Underfitting:
```yaml
# Train longer or adjust learning rate
training:
  epochs: 50
  lr: 5.0e-4
```

---

## 📝 Summary Table

| File | Purpose | Type | Priority |
|------|---------|------|----------|
| **vision/train.py** | Main training loop | Primary | ⭐⭐⭐ |
| **vision/model.py** | Model architecture | Core | ⭐⭐⭐ |
| **vision/dataset.py** | Data loading | Core | ⭐⭐⭐ |
| **vision/taxonomy.py** | Class definitions | Core | ⭐⭐ |
| **vision/evaluate.py** | Evaluation metrics | Core | ⭐⭐ |
| **scripts/step4_train_cell_classifier.py** | Pipeline training | Alternative | ⭐⭐ |
| **scripts/step5_evaluate_and_xai.py** | Evaluation script | Pipeline | ⭐⭐ |
| **scripts/run_all.py** | Full pipeline | Orchestration | ⭐ |
| **configs/default.yaml** | Hyperparameters | Config | ⭐⭐⭐ |

---

## 🚀 Quick Start Training

```bash
# 1. Ensure data is ready
python scripts/step3_verify_crops.py

# 2. Train model
python vision/train.py

# 3. Evaluate
python scripts/step5_evaluate_and_xai.py

# 4. Use trained model
streamlit run app/streamlit_app.py
```

---

## 📚 Code Example: Minimal Training Script

```python
# minimal_train.py
import torch
from torch.utils.data import DataLoader
from vision.model import BoneMarrowViT
from vision.dataset import CellCropDataset
from configs.config_loader import load_config

# Load config
cfg = load_config()

# Create model
model = BoneMarrowViT(
    num_classes=8,
    backbone="vit_small_patch16_224",
    lora_enabled=True
)
model = model.to("cuda")

# Load dataset
train_dataset = CellCropDataset(
    metadata_csv="data/processed/cell_metadata.csv",
    split="train"
)
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

# Training loop
for epoch in range(30):
    model.train()
    for images, labels in train_loader:
        images = images.to("cuda")
        labels = labels.to("cuda")
        
        optimizer.zero_grad()
        logits = model(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}: Loss {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "model.pt")
```

---

**This guide provides complete reference for all ML training code in the project!**
