"""
vision/dataset.py
─────────────────────────────────────────────────────────────────────────────
Bone Marrow Cell Dataset for BoneMarrowWSI-PediatricLeukemia.

Key design principles:
  1. Patient-level split isolation is ALWAYS enforced via splits.json.
     No cell-level random splits are ever created.
  2. Cells are loaded from pre-extracted PNG crops saved under
     data/processed/cell_crops/{split}/{patient_id}/.
  3. Only "valid" cells (extraction_status == "valid") from the consensus
     cohort are used for training and evaluation.
  4. The 8-class taxonomy from vision.taxonomy is the sole source of truth
     for class indices.
  5. Class imbalance is handled via WeightedRandomSampler (or class-weighted
     loss — controlled by the training script).
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import json
import logging
import os

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import torch
    from torch.utils.data import Dataset, WeightedRandomSampler
    from torchvision import transforms
    from PIL import Image
except ImportError:
    torch = None
    Dataset = object
    WeightedRandomSampler = None
    transforms = None
    Image = None

from vision.taxonomy import (
    MORPH_CLASSES,
    MORPH_TO_IDX,
    IDX_TO_MORPH,
    EXCLUDED_CATEGORIES,
)

# Re-export for backward compatibility
CELL_CLASSES = MORPH_CLASSES
CLASS_TO_IDX = MORPH_TO_IDX
IDX_TO_CLASS = IDX_TO_MORPH

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Split / sample loading helpers ────────────────────────────────────────────

def load_splits(splits_path: Union[str, Path]) -> Dict[str, List[str]]:
    """
    Loads the patient-level splits JSON and returns a dict with keys
    'train', 'val', 'test', each mapping to a list of patient IDs.
    """
    splits_path = Path(splits_path)
    if not splits_path.exists():
        raise FileNotFoundError(
            f"splits.json not found at '{splits_path}'. "
            "Run the split generation script before training."
        )
    with open(splits_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Support both flat format {"train": [...]} and nested {"splits": {"train": [...]}}
    if "splits" in data:
        return data["splits"]
    return data


def load_cell_samples(
    cell_metadata_path: Union[str, Path],
    splits_path: Union[str, Path],
    split: str,
    crop_base_dir: Optional[Union[str, Path]] = None,
) -> List[Dict[str, Any]]:
    """
    Loads the list of valid cell samples for a given split from cell_metadata.csv.

    Patient-level split membership is enforced using splits.json.
    Only cells with extraction_status == 'valid' and whose mapped_cell_category
    is not in EXCLUDED_CATEGORIES are returned.

    Args:
        cell_metadata_path: Path to data/processed/cell_metadata.csv.
        splits_path: Path to data/processed/splits.json.
        split: One of 'train', 'val', 'test'.
        crop_base_dir: Override for the base directory of crop PNG files.
            Defaults to REPO_ROOT / 'data' / 'processed' / 'cell_crops'.

    Returns:
        List of sample dicts with keys:
            'image_path', 'class_code', 'patient_id', 'original_label', 'cell_id'
    """
    if pd is None:
        raise ImportError("pandas is required. Install with: pip install pandas")

    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be one of 'train', 'val', 'test', got: {split!r}")

    # Load patient IDs for this split
    split_data = load_splits(splits_path)
    patient_ids_in_split = set(split_data.get(split, []))
    if not patient_ids_in_split:
        raise ValueError(f"No patients found for split '{split}' in {splits_path}")

    # Load cell metadata
    df = pd.read_csv(cell_metadata_path)

    # Filter: valid cells only
    df = df[df["extraction_status"] == "valid"].copy()

    # Filter: only patients in this split (patient-level isolation guarantee)
    df = df[df["patient_id"].isin(patient_ids_in_split)].copy()

    # Filter: exclude non-cell / ambiguous categories
    df = df[~df["mapped_cell_category"].isin(EXCLUDED_CATEGORIES)].copy()

    # Resolve crop paths
    if crop_base_dir is None:
        crop_base_dir = REPO_ROOT / "data" / "processed" / "cell_crops"
    crop_base_dir = Path(crop_base_dir)

    samples = []
    missing_count = 0
    for _, row in df.iterrows():
        # Resolve the crop path — support both absolute and relative paths
        raw_path = str(row["crop_path"])
        if os.path.isabs(raw_path):
            crop_path = Path(raw_path)
        else:
            crop_path = REPO_ROOT / raw_path

        if not crop_path.exists():
            missing_count += 1
            continue

        category = row["mapped_cell_category"]
        if category not in MORPH_TO_IDX:
            logger.warning(
                f"Unknown category '{category}' for cell {row['cell_id']}; skipping."
            )
            continue

        samples.append({
            "image_path": str(crop_path),
            "class_code": category,
            "patient_id": str(row["patient_id"]),
            "original_label": str(row["original_cell_label"]),
            "cell_id": str(row["cell_id"]),
        })

    if missing_count > 0:
        logger.warning(
            f"[{split}] {missing_count} cell crop files not found on disk and were skipped. "
            "Run vision/run_extraction.py to extract all cell crops before training."
        )

    logger.info(f"[{split}] Loaded {len(samples)} valid cell samples from {len(patient_ids_in_split)} patients.")
    return samples


# ── Image transforms ──────────────────────────────────────────────────────────

def get_default_transforms(image_size: int = 224, is_train: bool = True) -> Any:
    """
    Standard image transforms for Wright-Giemsa bone marrow cell microscopy.

    Training augmentations include spatial and color jitter appropriate for
    cytology images.  Validation/test uses deterministic centre-crop + resize.
    """
    if transforms is None:
        return None

    # ImageNet normalization (ViT pre-training distribution)
    norm_mean = [0.485, 0.456, 0.406]
    norm_std  = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=180),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class BoneMarrowCellDataset(Dataset):
    """
    PyTorch Dataset for pre-extracted BoneMarrowWSI-PediatricLeukemia cell crops.

    Each sample dict must contain:
        'image_path' : str   – absolute path to a 224×224 PNG crop
        'class_code' : str   – one of the 8 MORPH_CLASSES categories
        'patient_id' : str   – patient ID (for patient-level metrics)
        'original_label' : str  – the fine-grained consensus annotation label
        'cell_id'     : str  – unique cell identifier

    Missing image files raise FileNotFoundError immediately (no synthetic fallback).
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        transform: Optional[Callable] = None,
        image_size: int = 224,
    ):
        """
        Args:
            samples: Output of load_cell_samples() for a given split.
            transform: Optional torchvision transform. Defaults to val transforms.
            image_size: Image resize target (default 224 for ViT-Small/16).
        """
        self.samples = samples
        self.transform = transform or get_default_transforms(image_size=image_size, is_train=False)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Any, int, str, str]:
        """
        Returns:
            (image_tensor, label_idx, class_code, patient_id)
        """
        item = self.samples[idx]
        image_path = item["image_path"]
        class_code = item["class_code"]
        patient_id = item["patient_id"]

        label_idx = MORPH_TO_IDX.get(class_code, len(MORPH_CLASSES) - 1)

        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"Cell crop not found at '{image_path}'. "
                "Synthetic fallbacks are strictly prohibited in the research pipeline. "
                "Run vision/run_extraction.py to pre-extract all cell crops."
            )

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise IOError(f"Failed to read crop '{image_path}': {e}") from e

        if self.transform is not None:
            img = self.transform(img)

        return img, label_idx, class_code, patient_id


# ── Class-imbalance helpers ───────────────────────────────────────────────────

def compute_class_weights(dataset: "BoneMarrowCellDataset") -> "torch.Tensor":
    """
    Computes per-class inverse-frequency weights for CrossEntropyLoss.

    Returns:
        1-D FloatTensor of shape (num_classes,) for use as `weight` in
        nn.CrossEntropyLoss(weight=...).
    """
    if torch is None:
        raise ImportError("PyTorch is required.")

    num_classes = len(MORPH_CLASSES)
    class_counts = [0] * num_classes

    for s in dataset.samples:
        lbl = MORPH_TO_IDX.get(s["class_code"], num_classes - 1)
        class_counts[lbl] += 1

    total = sum(class_counts)
    weights = []
    for cnt in class_counts:
        if cnt > 0:
            weights.append(total / (num_classes * cnt))
        else:
            weights.append(1.0)  # class not present — weight won't matter

    return torch.tensor(weights, dtype=torch.float)


def compute_weighted_sampler(dataset: "BoneMarrowCellDataset") -> Any:
    """
    Computes a WeightedRandomSampler that samples inversely proportional to
    class frequency.  Ensures rare categories (Monocyte Series, Lymphoid, etc.)
    are seen frequently during training.

    Returns:
        torch.utils.data.WeightedRandomSampler
    """
    if torch is None:
        raise ImportError("PyTorch is required.")

    num_classes = len(MORPH_CLASSES)
    class_counts = [0] * num_classes
    for s in dataset.samples:
        lbl = MORPH_TO_IDX.get(s["class_code"], num_classes - 1)
        class_counts[lbl] += 1

    total = len(dataset)
    class_weights = [
        total / (num_classes * cnt) if cnt > 0 else 1.0
        for cnt in class_counts
    ]

    sample_weights = [
        class_weights[MORPH_TO_IDX.get(s["class_code"], num_classes - 1)]
        for s in dataset.samples
    ]

    sampler_weights = torch.as_tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(
        weights=sampler_weights,
        num_samples=len(sampler_weights),
        replacement=True,
    )


# ── Legacy stub: create_patient_level_splits ──────────────────────────────────
# The real split is generated by scripts/generate_splits.py and stored in
# data/processed/splits.json.  This stub is kept for backward-compatibility
# with any existing tests that import it.

def create_patient_level_splits(
    patient_records: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    output_path: Optional[str] = "data/processed/splits.json",
) -> Dict[str, List[str]]:
    """
    DEPRECATED: Use scripts/generate_splits.py to regenerate splits.
    This function is kept only for backward compatibility with tests.
    """
    import warnings
    warnings.warn(
        "create_patient_level_splits() is deprecated.  "
        "The authoritative splits are stored in data/processed/splits.json. "
        "Use scripts/generate_splits.py to regenerate.",
        DeprecationWarning,
        stacklevel=2,
    )
    import random
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5
    patient_ids = sorted({p["patient_id"] for p in patient_records})
    rng = random.Random(seed)
    rng.shuffle(patient_ids)
    n = len(patient_ids)
    n_train = int(round(n * train_ratio))
    n_val   = int(round(n * val_ratio))
    return {
        "train": patient_ids[:n_train],
        "val":   patient_ids[n_train:n_train + n_val],
        "test":  patient_ids[n_train + n_val:],
    }
