"""
synthetic/sampler.py
─────────────────────────────────────────────────────────────────────────────
Samples cell crops to match a target cell-differential vector.
Supports both real images from downloaded TCIA raw data and mock image
descriptors for development/evaluation before full dataset download.
"""

from pathlib import Path
from typing import Dict, List, Optional
import random
import numpy as np

from synthetic.templates import CELL_CLASSES


def compute_discrete_counts(differential: Dict[str, float], total_crops: int) -> Dict[str, int]:
    """
    Convert fractional differential into discrete integer cell counts summing to total_crops.
    Uses the largest remainder method (Hare-Niemeyer) for exact summation.
    """
    fractions = [max(0.0, differential.get(c, 0.0)) for c in CELL_CLASSES]
    total_frac = sum(fractions)
    if total_frac == 0:
        fractions = [1.0 / len(CELL_CLASSES)] * len(CELL_CLASSES)
        total_frac = 1.0
    normalized = [f / total_frac for f in fractions]

    raw_counts = [f * total_crops for f in normalized]
    floored_counts = [int(np.floor(c)) for c in raw_counts]
    remainder = total_crops - sum(floored_counts)

    # Distribute remaining crops according to fractional parts
    fractional_parts = [raw_counts[i] - floored_counts[i] for i in range(len(CELL_CLASSES))]
    sorted_indices = np.argsort(fractional_parts)[::-1]

    counts = floored_counts.copy()
    for i in range(remainder):
        counts[sorted_indices[i]] += 1

    return {CELL_CLASSES[i]: counts[i] for i in range(len(CELL_CLASSES))}


def sample_crops_for_differential(
    differential: Dict[str, float],
    total_crops: int = 100,
    raw_dir: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, str]]:
    """
    Samples cell crops matching the specified differential.

    Returns:
        List of crop dicts with keys:
            - 'crop_id': unique identifier
            - 'class_code': cell class (e.g. 'MYO')
            - 'image_path': relative or absolute path (or simulated path)
            - 'is_mock': boolean indicating if real file exists
    """
    if rng is None:
        rng = random.Random()

    counts = compute_discrete_counts(differential, total_crops)
    selected_crops: List[Dict[str, str]] = []

    raw_path = Path(raw_dir) if raw_dir and Path(raw_dir).exists() else None

    # Cache existing images per class if directory exists
    class_to_files: Dict[str, List[Path]] = {}
    if raw_path:
        for c in CELL_CLASSES:
            # Check for folder by class name or code
            candidates = list(raw_path.glob(f"**/{c}/*.png")) + list(raw_path.glob(f"**/{c}/*.jpg"))
            if candidates:
                class_to_files[c] = candidates

    crop_idx = 0
    for class_code, count in counts.items():
        if count <= 0:
            continue

        available = class_to_files.get(class_code, [])
        if available:
            # Sample with replacement if count > available
            chosen = rng.choices(available, k=count) if count > len(available) else rng.sample(available, count)
            for path in chosen:
                crop_idx += 1
                selected_crops.append({
                    "crop_id": f"crop_{crop_idx:04d}",
                    "class_code": class_code,
                    "image_path": str(path),
                    "is_mock": False,
                })
        else:
            # Mock crop descriptor
            for i in range(count):
                crop_idx += 1
                selected_crops.append({
                    "crop_id": f"mock_{class_code}_{crop_idx:04d}",
                    "class_code": class_code,
                    "image_path": f"data/raw/{class_code}/mock_cell_{i+1:03d}.png",
                    "is_mock": True,
                })

    rng.shuffle(selected_crops)
    return selected_crops
