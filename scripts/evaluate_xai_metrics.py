"""
scripts/evaluate_xai_metrics.py
─────────────────────────────────────────────────────────────────────────────
Evaluate Quantitative XAI Metrics for Bone Marrow Cytomorphology Classifier.

Usage:
    python scripts/evaluate_xai_metrics.py --checkpoint models/checkpoints/best_vit_lora_morphology.pt

This script:
1. Loads test set cell crops
2. Generates attention heatmaps (ViTAttentionRollout)
3. Computes quantitative metrics (Pointing Game, Deletion/Insertion AUC)
4. Generates visualizations and reports
5. Exports results to JSON/CSV for further analysis

Author: AI Assistant
Date: November 8, 2026
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vision.model import BoneMarrowViT
from vision.attention_rollout import ViTAttentionRollout
from vision.taxonomy import MORPH_CLASSES, MORPH_TO_IDX
from vision.xai_metrics import (
    AnnotatedRegion,
    compute_xai_metrics,
    aggregate_xai_metrics,
    visualize_xai_metrics,
    format_xai_metrics_report,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("evaluate_xai")

# Directories
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"
OUTPUT_DIR = REPO_ROOT / "results" / "xai_evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_model(checkpoint_path: Path, device: torch.device) -> BoneMarrowViT:
    """Load trained BoneMarrowViT model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model = BoneMarrowViT(
        backbone_name=ckpt.get("backbone_name", "vit_small_patch16_224"),
        num_classes=ckpt.get("num_classes", len(MORPH_CLASSES)),
        pretrained=False,
        use_lora=ckpt.get("use_lora", False),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    
    # Disable fused attention (required for attention rollout)
    for m in model.modules():
        if hasattr(m, "fused_attn"):
            m.fused_attn = False
    
    logger.info("Model loaded successfully")
    return model


def load_test_samples(
    num_samples: int = 50,
    data_dir: Path = REPO_ROOT / "data" / "processed" / "224x224_patches"
) -> List[Dict[str, Any]]:
    """
    Load test samples for XAI evaluation.
    
    Returns list of dicts with keys: 'image_path', 'label', 'class_name'
    """
    logger.info(f"Loading test samples from {data_dir}")
    
    samples = []
    test_dir = data_dir / "test"
    
    if not test_dir.exists():
        logger.warning(f"Test directory not found: {test_dir}")
        return samples
    
    # Collect samples from each class
    for class_dir in test_dir.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        if class_name not in MORPH_TO_IDX:
            continue
        
        image_files = list(class_dir.glob("*.png")) + list(class_dir.glob("*.jpg"))
        
        # Sample uniformly from each class
        sample_size = min(num_samples // len(MORPH_CLASSES), len(image_files))
        sampled_files = np.random.choice(image_files, size=sample_size, replace=False)
        
        for img_path in sampled_files:
            samples.append({
                "image_path": str(img_path),
                "label": MORPH_TO_IDX[class_name],
                "class_name": class_name,
            })
    
    logger.info(f"Loaded {len(samples)} test samples across {len(set(s['class_name'] for s in samples))} classes")
    return samples


def load_annotations(annotation_file: Path) -> Dict[str, List[AnnotatedRegion]]:
    """
    Load pathologist annotations from JSON file.
    
    Expected format:
    {
        "cell_id_001": [
            {"bbox": [x, y, w, h], "region_type": "nucleus", "confidence": 1.0},
            {"bbox": [x, y, w, h], "region_type": "nucleolus", "confidence": 0.9}
        ],
        ...
    }
    
    Returns:
        Dict mapping cell_id -> List[AnnotatedRegion]
    """
    if not annotation_file.exists():
        logger.warning(f"Annotation file not found: {annotation_file}. Proceeding without annotations.")
        return {}
    
    logger.info(f"Loading annotations from {annotation_file}")
    with open(annotation_file, 'r') as f:
        data = json.load(f)
    
    annotations = {}
    for cell_id, regions in data.items():
        annotations[cell_id] = [
            AnnotatedRegion(
                cell_id=cell_id,
                bbox=tuple(r["bbox"]),
                region_type=r["region_type"],
                confidence=r.get("confidence", 1.0)
            )
            for r in regions
        ]
    
    logger.info(f"Loaded annotations for {len(annotations)} cells")
    return annotations


def main():
    parser = argparse.ArgumentParser(description="Evaluate Quantitative XAI Metrics")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_DIR / "best_vit_lora_morphology.pt"),
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of test samples to evaluate (default: 50)"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="",
        help="Path to pathologist annotations JSON file (optional)"
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=50,
        help="Number of steps for deletion/insertion curves (default: 50)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate per-cell visualizations"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    model = load_model(checkpoint_path, device)
    
    # Initialize attention rollout
    rollout = ViTAttentionRollout(model.backbone, discard_ratio=0.1, head_fusion="mean")
    
    # Load test samples
    test_samples = load_test_samples(num_samples=args.num_samples)
    if len(test_samples) == 0:
        logger.error("No test samples found. Exiting.")
        return 1
    
    # Load annotations (if provided)
    annotations = {}
    if args.annotations:
        annotations = load_annotations(Path(args.annotations))
    
    # Image preprocessing
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Evaluate each sample
    results = []
    logger.info(f"Evaluating {len(test_samples)} samples...")
    
    for sample in tqdm(test_samples, desc="Computing XAI metrics"):
        try:
            # Load image
            img_path = Path(sample["image_path"])
            cell_id = img_path.stem
            pil_img = Image.open(img_path).convert("RGB")
            img_np = np.array(pil_img)
            
            # Prepare tensor
            tensor = transform(pil_img).unsqueeze(0)
            
            # Forward pass to get prediction
            with torch.no_grad():
                logits = model(tensor.to(device))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                pred_class = list(MORPH_TO_IDX.keys())[pred_idx]
                confidence = float(probs[pred_idx])
            
            # Generate attention heatmap
            attention_map = rollout(tensor)
            
            # Get annotations for this cell (if available)
            cell_annotations = annotations.get(cell_id, [])
            
            # Compute XAI metrics
            result = compute_xai_metrics(
                model=model,
                input_tensor=tensor,
                attention_map=attention_map,
                predicted_class_idx=pred_idx,
                cell_id=cell_id,
                predicted_class=pred_class,
                confidence=confidence,
                annotated_regions=cell_annotations,
                device=str(device),
                num_steps=args.num_steps,
            )
            
            results.append(result)
            
            # Visualize (if requested)
            if args.visualize and len(cell_annotations) > 0:
                viz_dir = output_dir / "visualizations"
                viz_dir.mkdir(exist_ok=True)
                viz_path = viz_dir / f"{cell_id}_xai_metrics.png"
                
                visualize_xai_metrics(
                    image=img_np,
                    attention_map=attention_map,
                    annotated_regions=cell_annotations,
                    result=result,
                    save_path=str(viz_path)
                )
        
        except Exception as e:
            logger.warning(f"Failed to evaluate {sample['image_path']}: {e}")
            continue
    
    logger.info(f"Successfully evaluated {len(results)}/{len(test_samples)} samples")
    
    # Aggregate metrics
    agg_metrics = aggregate_xai_metrics(results)
    
    # Generate report
    report = format_xai_metrics_report(agg_metrics)
    report_path = output_dir / "xai_metrics_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_path}")
    
    # Export raw results to JSON
    results_json = [
        {
            "cell_id": r.cell_id,
            "predicted_class": r.predicted_class,
            "confidence": r.confidence,
            "pointing_game_hit": r.pointing_game_hit,
            "pointing_game_score": r.pointing_game_score,
            "energy_pointing_game_score": r.energy_pointing_game_score,
            "deletion_auc": r.deletion_auc,
            "insertion_auc": r.insertion_auc,
            "average_drop": r.average_drop,
            "increase_in_confidence": r.increase_in_confidence,
            "has_annotation": r.has_annotation,
            "num_annotated_regions": r.num_annotated_regions,
        }
        for r in results
    ]
    
    json_path = output_dir / "xai_metrics_results.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Raw results exported to {json_path}")
    
    # Print summary to console
    print("\n" + "="*70)
    print("QUANTITATIVE XAI EVALUATION SUMMARY")
    print("="*70)
    print(report)
    
    logger.info("XAI evaluation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
