"""
vision/xai_metrics.py
─────────────────────────────────────────────────────────────────────────────
Quantitative XAI Evaluation Metrics for Vision Transformers.

Literature Foundation:
    - Adebayo et al. (2018): "Sanity Checks for Saliency Maps"
      NIPS 2018. https://arxiv.org/abs/1810.03292
      
    - Petsiuk et al. (2018): "RISE: Randomized Input Sampling for Explanation"
      BMVC 2018. https://arxiv.org/abs/1806.07421
      
    - Samek et al. (2017): "Evaluating the Visualization of What a Deep Neural
      Network Has Learned" IEEE TNNLS. https://arxiv.org/abs/1509.06321

Implemented Metrics:
    1. Pointing Game: Does peak attention overlap with annotated diagnostic regions?
    2. Deletion AUC: Remove high-attention regions, measure confidence drop
    3. Insertion AUC: Progressively add high-attention regions from baseline
    4. Energy-Based Pointing Game: Uses attention mass instead of peak
    5. Average Drop: Mean confidence decrease after perturbation
    6. Increase in Confidence: Confidence gain from attention-guided insertion

Clinical Rationale:
    Real pathologists focus on specific morphological features:
    - Nucleus (chromatin pattern, nucleoli)
    - Cytoplasm (granules, Auer rods)
    - Nuclear-to-cytoplasmic (N:C) ratio
    
    These metrics objectively validate that the model's attention aligns with
    clinically relevant regions, not artifacts or background.

Author: AI Assistant
Date: November 8, 2026
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logger = logging.getLogger("xai_metrics")


@dataclass
class AnnotatedRegion:
    """
    Pathologist-annotated diagnostically relevant region.
    
    Attributes:
        cell_id: Identifier for the cell crop
        bbox: Bounding box [x, y, width, height] in pixel coordinates
        region_type: Type of region ("nucleus", "nucleolus", "cytoplasm", "auer_rod", "granules")
        annotator_id: Optional pathologist identifier
        confidence: Annotator confidence (0-1), optional
    """
    cell_id: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    region_type: str
    annotator_id: Optional[str] = None
    confidence: float = 1.0
    
    def to_mask(self, height: int = 224, width: int = 224) -> np.ndarray:
        """Convert bounding box to binary mask."""
        mask = np.zeros((height, width), dtype=np.float32)
        x, y, w, h = self.bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        mask[y1:y2, x1:x2] = 1.0
        return mask


@dataclass
class XAIMetricsResult:
    """
    Results from quantitative XAI evaluation.
    
    All metrics are computed per cell and aggregated.
    """
    cell_id: str
    predicted_class: str
    confidence: float
    
    # Pointing Game metrics
    pointing_game_hit: bool = False  # Did peak attention hit annotated region?
    pointing_game_score: float = 0.0  # 1.0 if hit, 0.0 if miss
    energy_pointing_game_score: float = 0.0  # Attention mass in annotated region
    
    # Perturbation metrics
    deletion_auc: float = 0.0  # Area under curve when deleting high-attention regions
    insertion_auc: float = 0.0  # Area under curve when inserting high-attention regions
    average_drop: float = 0.0  # Mean confidence drop after perturbation
    increase_in_confidence: float = 0.0  # Confidence gain from attention-guided insertion
    
    # Attention statistics
    attention_max: float = 0.0  # Peak attention value
    attention_mean: float = 0.0  # Mean attention in annotated regions
    attention_std: float = 0.0  # Std of attention in annotated regions
    
    # Metadata
    has_annotation: bool = False
    num_annotated_regions: int = 0


@dataclass
class AggregatedXAIMetrics:
    """
    Aggregated XAI metrics across multiple cells.
    
    Used for reporting overall explainability quality.
    """
    num_cells: int = 0
    num_annotated_cells: int = 0
    
    # Pointing Game (higher = better)
    pointing_game_accuracy: float = 0.0  # % of cells with peak hit
    energy_pointing_game_mean: float = 0.0  # Mean attention mass in annotated regions
    
    # Perturbation metrics (higher = better)
    deletion_auc_mean: float = 0.0  # Higher = model relies on attention regions
    insertion_auc_mean: float = 0.0  # Higher = attention regions are informative
    average_drop_mean: float = 0.0  # Higher = attention is faithful
    increase_in_confidence_mean: float = 0.0  # Higher = attention improves prediction
    
    # Per-class breakdown
    per_class_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Clinical interpretation
    clinical_interpretation: str = ""


def pointing_game(
    attention_map: np.ndarray,
    annotated_regions: List[AnnotatedRegion],
    threshold: float = 0.5,
) -> Tuple[bool, float]:
    """
    Pointing Game: Does the peak attention fall within any annotated region?
    
    Literature: Zhang et al. (2018) "Top-down Neural Attention by Excitation Backprop"
    
    Args:
        attention_map: 2D attention heatmap (H, W), normalized to [0, 1]
        annotated_regions: List of pathologist-annotated diagnostic regions
        threshold: Tolerance around peak (proportion of max attention)
    
    Returns:
        (hit: bool, score: float)
        - hit: True if peak attention overlaps with any annotated region
        - score: 1.0 if hit, 0.0 if miss
    """
    if len(annotated_regions) == 0:
        return False, 0.0
    
    # Find peak attention location
    peak_value = attention_map.max()
    peak_y, peak_x = np.unravel_index(attention_map.argmax(), attention_map.shape)
    
    # Create combined mask from all annotated regions
    h, w = attention_map.shape
    combined_mask = np.zeros((h, w), dtype=np.float32)
    for region in annotated_regions:
        combined_mask = np.maximum(combined_mask, region.to_mask(h, w))
    
    # Check if peak falls within any annotated region
    hit = combined_mask[peak_y, peak_x] > 0
    
    return hit, 1.0 if hit else 0.0


def energy_pointing_game(
    attention_map: np.ndarray,
    annotated_regions: List[AnnotatedRegion],
) -> float:
    """
    Energy-Based Pointing Game: Total attention mass within annotated regions.
    
    More robust than standard pointing game (not sensitive to single peak location).
    
    Args:
        attention_map: 2D attention heatmap (H, W), normalized to [0, 1]
        annotated_regions: List of pathologist-annotated diagnostic regions
    
    Returns:
        score: Proportion of total attention mass inside annotated regions [0, 1]
    """
    if len(annotated_regions) == 0:
        return 0.0
    
    # Create combined mask
    h, w = attention_map.shape
    combined_mask = np.zeros((h, w), dtype=np.float32)
    for region in annotated_regions:
        combined_mask = np.maximum(combined_mask, region.to_mask(h, w))
    
    # Compute attention mass inside vs outside
    attention_inside = (attention_map * combined_mask).sum()
    total_attention = attention_map.sum()
    
    if total_attention < 1e-8:
        return 0.0
    
    return float(attention_inside / total_attention)


def deletion_metric(
    model: nn.Module,
    input_tensor: torch.Tensor,
    attention_map: np.ndarray,
    predicted_class_idx: int,
    num_steps: int = 50,
    deletion_mode: str = "blur",
    device: str = "cpu",
) -> Tuple[float, np.ndarray]:
    """
    Deletion AUC: Remove high-attention regions progressively, measure confidence drop.
    
    Literature: Petsiuk et al. (2018) "RISE: Randomized Input Sampling"
    
    Higher AUC = model relies more on attention regions (faithful explanation).
    
    Args:
        model: PyTorch model
        input_tensor: Input image tensor (1, 3, H, W)
        attention_map: 2D attention heatmap (H, W)
        predicted_class_idx: Index of predicted class
        num_steps: Number of deletion steps
        deletion_mode: "blur" (Gaussian blur) or "zero" (set to 0)
        device: torch device
    
    Returns:
        (auc: float, curve: np.ndarray of confidences at each step)
    """
    model.eval()
    h, w = attention_map.shape
    
    # Sort pixels by attention (high to low)
    flat_attention = attention_map.flatten()
    sorted_indices = np.argsort(-flat_attention)  # Descending order
    
    confidences = []
    pixels_per_step = len(sorted_indices) // num_steps
    
    # Create perturbed versions
    with torch.no_grad():
        for step in range(num_steps + 1):
            # Number of pixels to delete
            num_deleted = step * pixels_per_step
            
            # Create mask (0 = delete, 1 = keep)
            mask = np.ones(len(sorted_indices), dtype=np.float32)
            if num_deleted > 0:
                mask[sorted_indices[:num_deleted]] = 0.0
            mask_2d = mask.reshape(h, w)
            
            # Apply perturbation
            perturbed = input_tensor.clone()
            
            if deletion_mode == "blur":
                # Gaussian blur the deleted regions
                blurred = F.interpolate(
                    F.avg_pool2d(perturbed, kernel_size=11, stride=1, padding=5),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False
                )
                mask_3d = torch.from_numpy(mask_2d).unsqueeze(0).unsqueeze(0).float().to(device)
                perturbed = mask_3d * perturbed + (1 - mask_3d) * blurred
            else:  # zero
                mask_3d = torch.from_numpy(mask_2d).unsqueeze(0).unsqueeze(0).float().to(device)
                perturbed = mask_3d * perturbed
            
            # Forward pass
            logits = model(perturbed.to(device))
            probs = torch.softmax(logits, dim=1)
            conf = probs[0, predicted_class_idx].item()
            confidences.append(conf)
    
    # Compute AUC (normalize by number of steps)
    confidences = np.array(confidences)
    # Use trapezoid (NumPy 2.0+) or fall back to trapz for older versions
    try:
        auc = np.trapezoid(confidences, dx=1.0 / num_steps)
    except AttributeError:
        auc = np.trapz(confidences, dx=1.0 / num_steps)  # type: ignore
    
    return auc, confidences


def insertion_metric(
    model: nn.Module,
    input_tensor: torch.Tensor,
    attention_map: np.ndarray,
    predicted_class_idx: int,
    num_steps: int = 50,
    baseline_mode: str = "blur",
    device: str = "cpu",
) -> Tuple[float, np.ndarray]:
    """
    Insertion AUC: Progressively reveal high-attention regions from baseline, measure confidence gain.
    
    Literature: Petsiuk et al. (2018) "RISE: Randomized Input Sampling"
    
    Higher AUC = attention regions are more informative (faithful explanation).
    
    Args:
        model: PyTorch model
        input_tensor: Input image tensor (1, 3, H, W)
        attention_map: 2D attention heatmap (H, W)
        predicted_class_idx: Index of predicted class
        num_steps: Number of insertion steps
        baseline_mode: "blur" (start from blur) or "zero" (start from black)
        device: torch device
    
    Returns:
        (auc: float, curve: np.ndarray of confidences at each step)
    """
    model.eval()
    h, w = attention_map.shape
    
    # Sort pixels by attention (high to low)
    flat_attention = attention_map.flatten()
    sorted_indices = np.argsort(-flat_attention)  # Descending order
    
    confidences = []
    pixels_per_step = len(sorted_indices) // num_steps
    
    # Create baseline (fully perturbed)
    if baseline_mode == "blur":
        baseline = F.interpolate(
            F.avg_pool2d(input_tensor, kernel_size=11, stride=1, padding=5),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )
    else:  # zero
        baseline = torch.zeros_like(input_tensor)
    
    # Progressively reveal high-attention regions
    with torch.no_grad():
        for step in range(num_steps + 1):
            # Number of pixels to reveal
            num_revealed = step * pixels_per_step
            
            # Create mask (1 = reveal, 0 = keep baseline)
            mask = np.zeros(len(sorted_indices), dtype=np.float32)
            if num_revealed > 0:
                mask[sorted_indices[:num_revealed]] = 1.0
            mask_2d = mask.reshape(h, w)
            
            # Apply insertion
            mask_3d = torch.from_numpy(mask_2d).unsqueeze(0).unsqueeze(0).float().to(device)
            inserted = mask_3d * input_tensor + (1 - mask_3d) * baseline
            
            # Forward pass
            logits = model(inserted.to(device))
            probs = torch.softmax(logits, dim=1)
            conf = probs[0, predicted_class_idx].item()
            confidences.append(conf)
    
    # Compute AUC
    confidences = np.array(confidences)
    # Use trapezoid (NumPy 2.0+) or fall back to trapz for older versions
    try:
        auc = np.trapezoid(confidences, dx=1.0 / num_steps)
    except AttributeError:
        auc = np.trapz(confidences, dx=1.0 / num_steps)  # type: ignore
    
    return auc, confidences


def average_drop(
    model: nn.Module,
    input_tensor: torch.Tensor,
    attention_map: np.ndarray,
    predicted_class_idx: int,
    deletion_ratio: float = 0.3,
    deletion_mode: str = "blur",
    device: str = "cpu",
) -> float:
    """
    Average Drop: Mean confidence decrease after removing top attention regions.
    
    Simplified version of deletion metric (single-point measurement).
    
    Args:
        model: PyTorch model
        input_tensor: Input image tensor (1, 3, H, W)
        attention_map: 2D attention heatmap (H, W)
        predicted_class_idx: Index of predicted class
        deletion_ratio: Proportion of pixels to delete (default: 30%)
        deletion_mode: "blur" or "zero"
        device: torch device
    
    Returns:
        drop: Confidence decrease (original_conf - perturbed_conf)
    """
    model.eval()
    h, w = attention_map.shape
    
    # Original confidence
    with torch.no_grad():
        logits = model(input_tensor.to(device))
        probs = torch.softmax(logits, dim=1)
        original_conf = probs[0, predicted_class_idx].item()
    
    # Delete top attention pixels
    flat_attention = attention_map.flatten()
    sorted_indices = np.argsort(-flat_attention)
    num_deleted = int(len(sorted_indices) * deletion_ratio)
    
    mask = np.ones(len(sorted_indices), dtype=np.float32)
    mask[sorted_indices[:num_deleted]] = 0.0
    mask_2d = mask.reshape(h, w)
    
    # Apply perturbation
    perturbed = input_tensor.clone()
    
    if deletion_mode == "blur":
        blurred = F.interpolate(
            F.avg_pool2d(perturbed, kernel_size=11, stride=1, padding=5),
            size=(h, w),
            mode="bilinear",
            align_corners=False
        )
        mask_3d = torch.from_numpy(mask_2d).unsqueeze(0).unsqueeze(0).float().to(device)
        perturbed = mask_3d * perturbed + (1 - mask_3d) * blurred
    else:
        mask_3d = torch.from_numpy(mask_2d).unsqueeze(0).unsqueeze(0).float().to(device)
        perturbed = mask_3d * perturbed
    
    # Perturbed confidence
    with torch.no_grad():
        logits_perturbed = model(perturbed.to(device))
        probs_perturbed = torch.softmax(logits_perturbed, dim=1)
        perturbed_conf = probs_perturbed[0, predicted_class_idx].item()
    
    drop = original_conf - perturbed_conf
    return max(0.0, drop)  # Clip to non-negative


def compute_xai_metrics(
    model: nn.Module,
    input_tensor: torch.Tensor,
    attention_map: np.ndarray,
    predicted_class_idx: int,
    cell_id: str,
    predicted_class: str,
    confidence: float,
    annotated_regions: Optional[List[AnnotatedRegion]] = None,
    device: str = "cpu",
    num_steps: int = 50,
) -> XAIMetricsResult:
    """
    Compute all quantitative XAI metrics for a single cell.
    
    Args:
        model: PyTorch model
        input_tensor: Input image tensor (1, 3, H, W)
        attention_map: 2D attention heatmap (H, W), normalized [0, 1]
        predicted_class_idx: Index of predicted class
        cell_id: Cell identifier
        predicted_class: Predicted class name
        confidence: Model confidence
        annotated_regions: Optional list of pathologist annotations
        device: torch device
        num_steps: Number of steps for deletion/insertion curves
    
    Returns:
        XAIMetricsResult object with all computed metrics
    """
    result = XAIMetricsResult(
        cell_id=cell_id,
        predicted_class=predicted_class,
        confidence=confidence,
    )
    
    # Annotation-dependent metrics
    if annotated_regions and len(annotated_regions) > 0:
        result.has_annotation = True
        result.num_annotated_regions = len(annotated_regions)
        
        # Pointing Game
        pg_hit, pg_score = pointing_game(attention_map, annotated_regions)
        result.pointing_game_hit = pg_hit
        result.pointing_game_score = pg_score
        
        # Energy Pointing Game
        result.energy_pointing_game_score = energy_pointing_game(attention_map, annotated_regions)
        
        # Attention statistics in annotated regions
        h, w = attention_map.shape
        combined_mask = np.zeros((h, w), dtype=np.float32)
        for region in annotated_regions:
            combined_mask = np.maximum(combined_mask, region.to_mask(h, w))
        
        attention_in_region = attention_map[combined_mask > 0]
        if len(attention_in_region) > 0:
            result.attention_mean = float(np.mean(attention_in_region))
            result.attention_std = float(np.std(attention_in_region))
        result.attention_max = float(attention_map.max())
    
    # Perturbation metrics (always computed)
    try:
        del_auc, _ = deletion_metric(
            model, input_tensor, attention_map, predicted_class_idx,
            num_steps=num_steps, deletion_mode="blur", device=device
        )
        result.deletion_auc = del_auc
        
        ins_auc, _ = insertion_metric(
            model, input_tensor, attention_map, predicted_class_idx,
            num_steps=num_steps, baseline_mode="blur", device=device
        )
        result.insertion_auc = ins_auc
        
        avg_drop = average_drop(
            model, input_tensor, attention_map, predicted_class_idx,
            deletion_ratio=0.3, deletion_mode="blur", device=device
        )
        result.average_drop = avg_drop
        
        # Increase in confidence (difference between final insertion and initial baseline)
        _, ins_curve = insertion_metric(
            model, input_tensor, attention_map, predicted_class_idx,
            num_steps=num_steps, baseline_mode="blur", device=device
        )
        result.increase_in_confidence = ins_curve[-1] - ins_curve[0]
        
    except Exception as e:
        logger.warning(f"Perturbation metrics failed for {cell_id}: {e}")
    
    return result


def aggregate_xai_metrics(
    results: List[XAIMetricsResult],
) -> AggregatedXAIMetrics:
    """
    Aggregate XAI metrics across multiple cells for summary statistics.
    
    Args:
        results: List of per-cell XAIMetricsResult objects
    
    Returns:
        AggregatedXAIMetrics with mean values and per-class breakdown
    """
    if len(results) == 0:
        return AggregatedXAIMetrics()
    
    agg = AggregatedXAIMetrics(num_cells=len(results))
    
    # Filter to annotated cells only
    annotated = [r for r in results if r.has_annotation]
    agg.num_annotated_cells = len(annotated)
    
    if len(annotated) > 0:
        # Pointing Game
        agg.pointing_game_accuracy = np.mean([r.pointing_game_score for r in annotated])
        agg.energy_pointing_game_mean = np.mean([r.energy_pointing_game_score for r in annotated])
    
    # Perturbation metrics (all cells)
    agg.deletion_auc_mean = np.mean([r.deletion_auc for r in results])
    agg.insertion_auc_mean = np.mean([r.insertion_auc for r in results])
    agg.average_drop_mean = np.mean([r.average_drop for r in results])
    agg.increase_in_confidence_mean = np.mean([r.increase_in_confidence for r in results])
    
    # Per-class breakdown
    classes = set(r.predicted_class for r in results)
    for cls in classes:
        cls_results = [r for r in results if r.predicted_class == cls]
        agg.per_class_metrics[cls] = {
            "count": len(cls_results),
            "deletion_auc": np.mean([r.deletion_auc for r in cls_results]),
            "insertion_auc": np.mean([r.insertion_auc for r in cls_results]),
            "average_drop": np.mean([r.average_drop for r in cls_results]),
        }
        
        # Add pointing game if annotated cells exist for this class
        cls_annotated = [r for r in cls_results if r.has_annotation]
        if len(cls_annotated) > 0:
            agg.per_class_metrics[cls]["pointing_game_accuracy"] = np.mean(
                [r.pointing_game_score for r in cls_annotated]
            )
            agg.per_class_metrics[cls]["energy_pointing_game"] = np.mean(
                [r.energy_pointing_game_score for r in cls_annotated]
            )
    
    # Clinical interpretation
    agg.clinical_interpretation = _generate_clinical_interpretation(agg)
    
    return agg


def _generate_clinical_interpretation(agg: AggregatedXAIMetrics) -> str:
    """Generate human-readable interpretation of XAI metrics."""
    lines = []
    
    # Pointing Game interpretation
    if agg.num_annotated_cells > 0:
        pg_acc = agg.pointing_game_accuracy
        if pg_acc >= 0.80:
            lines.append(f"✅ HIGH FAITHFULNESS: Attention peaks align with pathologist annotations ({pg_acc:.1%} accuracy)")
        elif pg_acc >= 0.60:
            lines.append(f"⚠️  MODERATE FAITHFULNESS: Attention partially aligns with annotations ({pg_acc:.1%} accuracy)")
        else:
            lines.append(f"❌ LOW FAITHFULNESS: Attention does not align with annotations ({pg_acc:.1%} accuracy)")
    
    # Deletion AUC interpretation
    del_auc = agg.deletion_auc_mean
    if del_auc >= 0.70:
        lines.append(f"✅ HIGH RELIANCE: Model strongly depends on attention regions (Deletion AUC: {del_auc:.3f})")
    elif del_auc >= 0.50:
        lines.append(f"⚠️  MODERATE RELIANCE: Model moderately depends on attention regions (Deletion AUC: {del_auc:.3f})")
    else:
        lines.append(f"❌ LOW RELIANCE: Model weakly depends on attention regions (Deletion AUC: {del_auc:.3f})")
    
    # Insertion AUC interpretation
    ins_auc = agg.insertion_auc_mean
    if ins_auc >= 0.70:
        lines.append(f"✅ HIGH INFORMATIVENESS: Attention regions are highly predictive (Insertion AUC: {ins_auc:.3f})")
    elif ins_auc >= 0.50:
        lines.append(f"⚠️  MODERATE INFORMATIVENESS: Attention regions are moderately predictive (Insertion AUC: {ins_auc:.3f})")
    else:
        lines.append(f"❌ LOW INFORMATIVENESS: Attention regions are weakly predictive (Insertion AUC: {ins_auc:.3f})")
    
    return "\n".join(lines)


def visualize_xai_metrics(
    image: np.ndarray,
    attention_map: np.ndarray,
    annotated_regions: List[AnnotatedRegion],
    result: XAIMetricsResult,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Visualize XAI metrics with annotations overlaid on attention heatmap.
    
    Args:
        image: Original image (H, W, 3), RGB, [0, 255]
        attention_map: Attention heatmap (H, W), [0, 1]
        annotated_regions: List of pathologist annotations
        result: XAI metrics result
        save_path: Optional path to save figure
    
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image with annotations
    axes[0].imshow(image.astype(np.uint8))
    axes[0].set_title("Original + Annotations", fontsize=12)
    axes[0].axis("off")
    
    for region in annotated_regions:
        x, y, w, h = region.bbox
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none', label=region.region_type
        )
        axes[0].add_patch(rect)
    
    # Attention heatmap
    axes[1].imshow(attention_map, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Attention Heatmap", fontsize=12)
    axes[1].axis("off")
    
    # Mark peak attention
    peak_y, peak_x = np.unravel_index(attention_map.argmax(), attention_map.shape)
    axes[1].scatter(peak_x, peak_y, c='white', s=100, marker='x', linewidths=3)
    
    # Overlay with annotations
    axes[2].imshow(image.astype(np.uint8))
    axes[2].imshow(attention_map, cmap="jet", alpha=0.5, vmin=0, vmax=1)
    axes[2].set_title("Overlay + Metrics", fontsize=12)
    axes[2].axis("off")
    
    for region in annotated_regions:
        x, y, w, h = region.bbox
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none'
        )
        axes[2].add_patch(rect)
    
    # Add metrics text
    metrics_text = f"""
Pointing Game: {'✓' if result.pointing_game_hit else '✗'} ({result.pointing_game_score:.2f})
Energy PG: {result.energy_pointing_game_score:.3f}
Deletion AUC: {result.deletion_auc:.3f}
Insertion AUC: {result.insertion_auc:.3f}
Avg Drop: {result.average_drop:.3f}
""".strip()
    
    fig.text(0.02, 0.02, metrics_text, fontsize=9, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"XAI metrics visualization saved to {save_path}")
    
    return fig


def format_xai_metrics_report(agg: AggregatedXAIMetrics) -> str:
    """
    Generate human-readable markdown report for XAI metrics.
    
    Args:
        agg: Aggregated XAI metrics
    
    Returns:
        Formatted markdown report string
    """
    report = f"""
# Quantitative XAI Evaluation Report

**Total Cells Evaluated**: {agg.num_cells}  
**Cells with Annotations**: {agg.num_annotated_cells}

---

## Summary Metrics

### Pointing Game (Annotation Overlap)
- **Accuracy**: {agg.pointing_game_accuracy:.1%} ({agg.num_annotated_cells} annotated cells)
- **Energy-Based Score**: {agg.energy_pointing_game_mean:.3f}

### Perturbation Metrics
- **Deletion AUC**: {agg.deletion_auc_mean:.3f} (higher = model relies on attention regions)
- **Insertion AUC**: {agg.insertion_auc_mean:.3f} (higher = attention regions are informative)
- **Average Drop**: {agg.average_drop_mean:.3f} (confidence decrease when removing attention)
- **Increase in Confidence**: {agg.increase_in_confidence_mean:.3f} (confidence gain from attention)

---

## Clinical Interpretation

{agg.clinical_interpretation}

---

## Per-Class Breakdown

"""
    
    if agg.per_class_metrics:
        report += "| Class | Count | Deletion AUC | Insertion AUC | Avg Drop | PG Accuracy |\n"
        report += "|-------|-------|--------------|---------------|----------|-------------|\n"
        
        for cls, metrics in sorted(agg.per_class_metrics.items()):
            pg_acc = metrics.get("pointing_game_accuracy", 0.0)
            pg_str = f"{pg_acc:.2f}" if pg_acc > 0 else "N/A"
            
            report += f"| {cls[:20]} | {metrics['count']} | {metrics['deletion_auc']:.3f} | "
            report += f"{metrics['insertion_auc']:.3f} | {metrics['average_drop']:.3f} | {pg_str} |\n"
    
    report += "\n---\n\n"
    report += "**Literature**: Adebayo et al. (2018), Petsiuk et al. (2018), Samek et al. (2017)\n"
    
    return report
