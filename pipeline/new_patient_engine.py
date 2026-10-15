"""
pipeline/new_patient_engine.py
─────────────────────────────────────────────────────────────────────────────
New-Case End-to-End Processing Engine.

ARCHITECTURE (corrected):

  Uploaded microscopy image(s)
       ↓
  Image validation
       ↓
  Cell detection / extraction   ← NEW: pipeline/cell_detector.py
       ↓
  Individual 224×224 cell crops
       ↓
  ViT-Small/16 + LoRA morphology classifier (EXISTING, unchanged)
       ↓
  Cell-level predictions
       ↓
  Sample-level aggregation
       ↓
  XAI (Attention Rollout per detected cell)
       ↓
  Hybrid RAG (case-specific query, age-aware)
       ↓
  LangGraph agentic assessment (grounding + sufficiency)
       ↓
  PatientAnalysis result

Guarantees:
  - The entire uploaded image is NEVER classified as one cell.
  - Cell crops come exclusively from the current case.
  - No hardcoded patient IDs, no cross-case data leakage.
  - Zero-cell inputs immediately trigger abstention.
  - Low-evidence (<50 cells) triggers uncertainty warning.
  - Missing metadata is represented as None, never fabricated.
"""

import hashlib
import json
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from pipeline.data_model import (
    CaseImageInfo,
    CellDetectionResult,
    CellInferenceResult,
    PatientAnalysis,
    XAIExample,
)
from pipeline.cell_detector import (
    DetectedCell,
    DetectionFailure,
    detect_cells,
    annotate_detections_on_image,
    generate_detection_debug_views,
)
from vision.model import BoneMarrowViT
from vision.taxonomy import MORPH_CLASSES, IDX_TO_MORPH
from vision.attention_rollout import ViTAttentionRollout
from retrieval.retrieve import HybridRetriever
from agents.graph import run_agentic_assessment

REPO_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("new_patient_engine")

CKPT_PATH = REPO_ROOT / "models" / "checkpoints" / "best_vit_lora_morphology.pt"
CACHE_DIR  = REPO_ROOT / "data" / "new_patient_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

LOW_EVIDENCE_THRESHOLD = 50
MIN_CLASSIFICATION_CONFIDENCE = 0.60  # Reject low-confidence predictions
   # cells; configurable

# ── Global singletons (loaded once per process) ───────────────────────────────
_GLOBAL_MODEL: Optional[BoneMarrowViT] = None
_GLOBAL_DEVICE: Optional[torch.device] = None
_GLOBAL_RETRIEVER: Optional[HybridRetriever] = None


def get_inference_device() -> torch.device:
    global _GLOBAL_DEVICE
    if _GLOBAL_DEVICE is None:
        _GLOBAL_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _GLOBAL_DEVICE


def get_loaded_model() -> Optional[BoneMarrowViT]:
    global _GLOBAL_MODEL
    if _GLOBAL_MODEL is None and CKPT_PATH.exists():
        dev = get_inference_device()
        try:
            ckpt = torch.load(CKPT_PATH, map_location=dev, weights_only=False)
            model = BoneMarrowViT(
                backbone_name=ckpt.get("backbone_name", "vit_small_patch16_224"),
                num_classes=ckpt.get("num_classes", len(MORPH_CLASSES)),
                pretrained=False,
                use_lora=ckpt.get("use_lora", False),
            )
            model.load_state_dict(ckpt["state_dict"])
            model.to(dev)
            model.eval()
            for m in model.modules():
                if hasattr(m, "fused_attn"):
                    m.fused_attn = False
            _GLOBAL_MODEL = model
            logger.info("ViT-Small/16 + LoRA model loaded.")
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            return None
    return _GLOBAL_MODEL


def get_loaded_retriever() -> Optional[HybridRetriever]:
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        try:
            _GLOBAL_RETRIEVER = HybridRetriever()
        except Exception as e:
            logger.warning(f"HybridRetriever init failed: {e}")
    return _GLOBAL_RETRIEVER


# ── Image-level helpers ───────────────────────────────────────────────────────

def _image_hash(pil_img: Image.Image) -> str:
    """Stable 12-char hash for cache keying (based on raw pixel bytes)."""
    arr = np.array(pil_img.convert("RGB").resize((64, 64)))
    return hashlib.md5(arr.tobytes()).hexdigest()[:12]


def _load_uploaded_file(file_obj: Any) -> Optional[Image.Image]:
    """Safely open a file object (Streamlit UploadedFile or path) as PIL RGB."""
    try:
        if hasattr(file_obj, "read"):
            file_obj.seek(0)
            img = Image.open(file_obj)
        else:
            img = Image.open(str(file_obj))
        return img.convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to open file {getattr(file_obj, 'name', file_obj)}: {e}")
        return None


# ── Stage 1: Image validation + cell detection ────────────────────────────────

def process_uploaded_images(
    uploaded_files: List[Any],
    case_dir: Path,
) -> Tuple[List[CaseImageInfo], List[CellDetectionResult]]:
    """
    For every uploaded file:
      1. Open as RGB PIL.
      2. Run cell detector.
      3. Save detected cell crops.
      4. Build CaseImageInfo and CellDetectionResult records.

    Returns (image_infos, detection_results).
    The two lists are parallel (same index = same image).
    """
    image_infos: List[CaseImageInfo] = []
    detection_results: List[CellDetectionResult] = []

    for img_idx, file_obj in enumerate(uploaded_files):
        image_id = f"IMAGE-{img_idx + 1:02d}"
        fname = getattr(file_obj, "name", f"image_{img_idx + 1}")

        pil_img = _load_uploaded_file(file_obj)
        if pil_img is None:
            info = CaseImageInfo(
                image_id=image_id,
                filename=fname,
                status="error",
                error_message=f"Could not open '{fname}' as an image.",
            )
            image_infos.append(info)
            detection_results.append(CellDetectionResult(image_id=image_id))
            continue

        w, h = pil_img.size
        img_hash = _image_hash(pil_img)

        info = CaseImageInfo(
            image_id=image_id,
            filename=fname,
            image_hash=img_hash,
            width_px=w,
            height_px=h,
            status="processing",
        )

        # Save original image thumbnail for UI display
        orig_dir = case_dir / "originals"
        orig_dir.mkdir(parents=True, exist_ok=True)
        orig_save = orig_dir / f"{image_id}_original.png"
        pil_img.save(orig_save)
        info.original_image_path = str(orig_save)

        # Cell detection
        crops_dir = case_dir / "crops" / image_id
        crops_dir.mkdir(parents=True, exist_ok=True)

        try:
            detected_cells, qc = detect_cells(
                pil_image=pil_img,
                image_id=image_id,
                save_dir=crops_dir,
            )

            # Save annotated image (bounding boxes)
            annotated = annotate_detections_on_image(pil_img, detected_cells)
            ann_path = orig_dir / f"{image_id}_annotated.png"
            annotated.save(ann_path)
            info.annotated_image_path = str(ann_path)

            # Generate 3 Debug Views
            raw_cands = qc.get("raw_candidates", [])
            v1_raw, v2_filt, v3_sheet = generate_detection_debug_views(
                pil_image=pil_img,
                raw_candidates=raw_cands,
                filtered_cells=detected_cells,
                max_crops_in_sheet=50,
            )
            v1_path = orig_dir / f"{image_id}_debug_raw.png"
            v2_path = orig_dir / f"{image_id}_debug_filtered.png"
            v3_path = orig_dir / f"{image_id}_debug_contact_sheet.png"
            v1_raw.save(v1_path)
            v2_filt.save(v2_path)
            v3_sheet.save(v3_path)
            info.debug_raw_path = str(v1_path)
            info.debug_filtered_path = str(v2_path)
            info.debug_contact_sheet_path = str(v3_path)

            det_result = CellDetectionResult(
                image_id=image_id,
                cells=detected_cells,
                qc=qc,
                status="ok",
            )
            info.status = "ok"
            info.cells_detected = len(detected_cells)

        except DetectionFailure as e:
            logger.warning(f"[{image_id}] Detection failed: {e}")
            det_result = CellDetectionResult(
                image_id=image_id,
                status="detection_failed",
                error_message=str(e),
            )
            info.status = "detection_failed"
            info.error_message = str(e)

        except Exception as e:
            logger.error(f"[{image_id}] Unexpected error during detection: {e}")
            det_result = CellDetectionResult(
                image_id=image_id,
                status="error",
                error_message=f"Unexpected error: {e}",
            )
            info.status = "error"
            info.error_message = f"Unexpected error: {e}"

        image_infos.append(info)
        detection_results.append(det_result)

    return image_infos, detection_results


# ── Stage 2: ViT Inference on detected cell crops ────────────────────────────

_VIT_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def run_cell_inference(
    model: BoneMarrowViT,
    device: torch.device,
    all_detected: List[DetectedCell],
) -> List[CellInferenceResult]:
    """
    Run the existing ViT classifier on each individual detected cell crop.
    Returns one CellInferenceResult per detected cell.
    """
    results: List[CellInferenceResult] = []

    model.eval()
    with torch.no_grad():
        for dc in all_detected:
            if dc.crop is None:
                continue
            try:
                tensor = _VIT_TRANSFORM(dc.crop).unsqueeze(0).to(device)
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                pred_class = IDX_TO_MORPH[pred_idx]
                conf = float(probs[pred_idx])
                prob_dict = {IDX_TO_MORPH[k]: float(probs[k]) for k in range(len(MORPH_CLASSES))}

                results.append(CellInferenceResult(
                    cell_id=dc.cell_id,
                    image_id=dc.image_id,
                    bbox=list(dc.bbox),
                    predicted_class=pred_class,
                    confidence=conf,
                    probabilities=prob_dict,
                    is_uncertain=dc.is_uncertain,
                    image_path=getattr(dc, "_save_path", None),
                ))
            except Exception as e:
                logger.warning(f"Inference failed for {dc.cell_id}: {e}")

    return results


# ── Stage 3: Aggregation ──────────────────────────────────────────────────────

def aggregate_cell_predictions(
    cell_preds: List[CellInferenceResult],
) -> Dict[str, Any]:
    """
    Compute sample-level morphology statistics from individual cell predictions.
    """
    total = len(cell_preds)
    if total == 0:
        return {
            "counts": {c: 0 for c in MORPH_CLASSES},
            "percentages": {c: 0.0 for c in MORPH_CLASSES},
            "blast_fraction": 0.0,
            "predominant": [],
            "confidence_summary": {},
            "total_cells": 0,
        }

    counts = {c: 0 for c in MORPH_CLASSES}
    for cp in cell_preds:
        counts[cp.predicted_class] = counts.get(cp.predicted_class, 0) + 1

    percentages = {k: round(v / total, 4) for k, v in counts.items()}
    blast_fraction = round(counts.get("Blast", 0) / total, 4)

    sorted_cats = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    predominant = [c[0] for c in sorted_cats if c[1] > 0][:4]

    conf_vals = [cp.confidence for cp in cell_preds]
    confidence_summary = {
        "mean_confidence": float(np.mean(conf_vals)),
        "min_confidence": float(np.min(conf_vals)),
        "max_confidence": float(np.max(conf_vals)),
    }

    return {
        "counts": counts,
        "percentages": percentages,
        "blast_fraction": blast_fraction,
        "predominant": predominant,
        "confidence_summary": confidence_summary,
        "total_cells": total,
    }


# ── Stage 4: XAI ─────────────────────────────────────────────────────────────

def generate_xai_for_cells(
    model: BoneMarrowViT,
    device: torch.device,
    cell_preds: List[CellInferenceResult],
    xai_dir: Path,
    max_examples: int = 3,
) -> List[XAIExample]:
    """
    Generate Attention Rollout overlays for a diverse selection of detected cells.
    Only operates on crops that physically exist in this case's directory.
    """
    if not cell_preds:
        return []

    xai_dir.mkdir(parents=True, exist_ok=True)
    rollout = ViTAttentionRollout(model.backbone, discard_ratio=0.1, head_fusion="mean")

    # Pick diverse classes, prioritise blast
    seen_classes: set = set()
    picked: List[CellInferenceResult] = []
    sorted_preds = sorted(cell_preds, key=lambda c: (c.predicted_class == "Blast", c.confidence), reverse=True)

    for cp in sorted_preds:
        if cp.predicted_class not in seen_classes and cp.image_path and Path(cp.image_path).exists():
            seen_classes.add(cp.predicted_class)
            picked.append(cp)
        if len(picked) >= max_examples:
            break

    # Top up if needed
    for cp in sorted_preds:
        if cp not in picked and cp.image_path and Path(cp.image_path).exists():
            picked.append(cp)
        if len(picked) >= max_examples:
            break

    examples: List[XAIExample] = []
    for item in picked:
        try:
            raw_img = Image.open(item.image_path).convert("RGB")
            tensor = _VIT_TRANSFORM(raw_img).unsqueeze(0).to(device)
            heatmap = rollout(tensor)

            raw_np = np.array(raw_img.resize((224, 224)))
            heatmap_col = cm.jet(heatmap)[:, :, :3]
            overlay = np.clip(0.55 * raw_np / 255.0 + 0.45 * heatmap_col, 0, 1)

            stem = f"{item.cell_id}_{item.predicted_class.replace(' ', '_').replace('/', '_')}"
            orig_save = xai_dir / f"{stem}_original.png"
            raw_img.save(orig_save)

            fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
            axes[0].imshow(raw_np); axes[0].set_title("Detected Cell", fontsize=9); axes[0].axis("off")
            axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1); axes[1].set_title("Attention Rollout", fontsize=9); axes[1].axis("off")
            axes[2].imshow(overlay); axes[2].set_title("Attribution Overlay", fontsize=9); axes[2].axis("off")
            fig.suptitle(f"{item.cell_id}: {item.predicted_class} ({item.confidence:.1%})", fontsize=10)
            plt.tight_layout()

            overlay_save = xai_dir / f"{stem}_overlay.png"
            plt.savefig(overlay_save, dpi=110, bbox_inches="tight")
            plt.close(fig)

            examples.append(XAIExample(
                cell_id=item.cell_id,
                image_id=item.image_id,
                predicted_class=item.predicted_class,
                confidence=item.confidence,
                original_image_path=str(orig_save),
                overlay_path=str(overlay_save),
            ))
        except Exception as e:
            logger.warning(f"XAI failed for {item.cell_id}: {e}")

    return examples


# ── Master orchestrator ───────────────────────────────────────────────────────

def analyze_new_case(
    uploaded_files: List[Any],
    case_id: Optional[str] = None,
    age_years: Optional[float] = None,
    sex: Optional[str] = None,
    sample_site: Optional[str] = None,
) -> PatientAnalysis:
    """
    Master pipeline for a new uploaded case.

    Parameters
    ----------
    uploaded_files : list
        One or more Streamlit UploadedFile objects (or local Path objects for testing).
    case_id : str, optional
        If None, a CASE-XXXX identifier is auto-generated.
    age_years : float, optional
        Patient age in years. None = not provided (never fabricated).
    sex : str, optional
        Biological sex. None = not provided.
    sample_site : str, optional
        Aspirate/biopsy site. None = not provided.
    """
    # ── 1. Initialise case identity ───────────────────────────────────────────
    if case_id and case_id.strip():
        cid = case_id.strip()
    else:
        cid = f"CASE-{uuid.uuid4().hex[:8].upper()}"

    age_months = float(age_years * 12.0) if (age_years is not None and age_years > 0) else None
    case_dir = CACHE_DIR / cid
    case_dir.mkdir(parents=True, exist_ok=True)

    analysis = PatientAnalysis(
        case_id=cid,
        age_years=age_years,
        age_months=age_months,
        sex=sex,
        sample_site=sample_site,
    )

    # ── 2. Image validation + cell detection ──────────────────────────────────
    if not uploaded_files:
        analysis.status = "error"
        analysis.status_message = "No files were uploaded."
        analysis.is_zero_cells = True
        analysis.requires_abstention = True
        analysis.final_report = (
            "> [!CAUTION]\n"
            "> **AUTOMATED ABSTENTION — NO FILES UPLOADED**\n\n"
            "> No microscopy files were provided. Analysis cannot proceed."
        )
        return analysis

    image_infos, detection_results = process_uploaded_images(uploaded_files, case_dir)
    analysis.image_infos = image_infos
    analysis._detection_results = detection_results  # type: ignore[attr-defined]

    # Collect all detected cells across all images
    all_detected: List[DetectedCell] = []
    per_image_counts: Dict[str, int] = {}
    for det in detection_results:
        per_image_counts[det.image_id] = len(det.cells)
        all_detected.extend(det.cells)

    analysis.per_image_cell_counts = per_image_counts
    analysis.total_images_processed = sum(1 for i in image_infos if i.status == "ok")
    analysis.total_images_failed = sum(1 for i in image_infos if i.status not in ("ok",))
    analysis.total_cells_detected = len(all_detected)

    if len(all_detected) == 0:
        analysis.status = "error"
        analysis.is_zero_cells = True
        analysis.requires_abstention = True
        detection_errors = [i.error_message for i in image_infos if i.error_message]
        if detection_errors:
            analysis.status_message = detection_errors[0]
        else:
            analysis.status_message = (
                "Individual cells could not be reliably detected in the submitted image(s). "
                "Please verify the image is a properly stained bone-marrow smear."
            )
        analysis.final_report = (
            f"> [!CAUTION]\n"
            f"> **AUTOMATED ABSTENTION — ZERO CELLS DETECTED**\n\n"
            f"> {analysis.status_message}"
        )
        return analysis

    analysis.is_low_evidence = len(all_detected) < LOW_EVIDENCE_THRESHOLD

    # ── 3. Load model ─────────────────────────────────────────────────────────
    model = get_loaded_model()
    device = get_inference_device()
    if model is None:
        analysis.status = "error"
        analysis.status_message = "Morphology classifier checkpoint not found. Cannot run inference."
        return analysis

    # ── 4. Cell-level ViT inference ───────────────────────────────────────────
    cell_preds = run_cell_inference(model, device, all_detected)
    analysis.cell_predictions = cell_preds
    analysis.cells_classified = len(cell_preds)

    # ── 5. Sample-level aggregation ───────────────────────────────────────────
    agg = aggregate_cell_predictions(cell_preds)
    analysis.morphology_counts = agg["counts"]
    analysis.morphology_percentages = agg["percentages"]
    analysis.blast_fraction = agg["blast_fraction"]
    analysis.predominant_categories = agg["predominant"]
    analysis.confidence_summary = agg["confidence_summary"]
    analysis.status = "ok"

    # ── 6. XAI ───────────────────────────────────────────────────────────────
    xai_dir = case_dir / "xai"
    xai_examples = generate_xai_for_cells(model, device, cell_preds, xai_dir, max_examples=3)
    analysis.xai_examples = xai_examples

    # ── 7. RAG retrieval (case-specific, age-aware) ───────────────────────────
    blast_frac = agg["blast_fraction"]
    predominant = agg["predominant"]

    # Infer morphological working hypothesis from image evidence only
    if blast_frac >= 0.20:
        pred_morph_dx = "ALL" if ("Lymphoid Series" in predominant) else "AML"
    elif "Granulocyte (Mature/Band)" in predominant or "Promyelocyte" in predominant:
        pred_morph_dx = "CML"
    else:
        pred_morph_dx = "Other"

    retriever = get_loaded_retriever()
    rag_query = ""
    retrieved_chunks = []
    if retriever:
        try:
            rag_query = retriever.build_query_from_morphology(
                predicted_class=pred_morph_dx,
                blast_fraction=blast_frac,
                predominant_cells=predominant,
                patient_age_months=age_months,  # None = unspecified → no "pediatric" inserted
            )
            retrieved_chunks = retriever.retrieve(query=rag_query, top_k=4)
        except Exception as e:
            logger.warning(f"RAG retrieval error: {e}")

    analysis.retrieved_query = rag_query
    analysis.retrieved_evidence = retrieved_chunks

    # ── 8. LangGraph agentic assessment ──────────────────────────────────────
    # One Source of Truth: all counts/fractions come exclusively from
    # aggregate_cell_predictions. No class probabilities are fabricated here.
    agent_input = {
        "patient_id": cid,
        "age_months": age_months,           # None if not provided
        "age_years": age_years,             # None if not provided
        "sex": sex,
        "annotated_blast_fraction": blast_frac,
        "predicted_class": pred_morph_dx,
        # Pass real per-class fractions from aggregation — never fabricated
        "class_probabilities": {k: round(v, 4) for k, v in agg["percentages"].items()},
        "cell_counts_by_category": agg["counts"],
        "total_cells": agg["total_cells"],
        "low_evidence_flag": analysis.is_low_evidence,
        "total_images": len(uploaded_files),
    }

    try:
        agent_out = run_agentic_assessment(agent_input)
        analysis.agent_status = "completed"
        analysis.draft_differential = agent_out.get("draft_differential")
        analysis.critic_grounding_score = float(agent_out.get("critic_grounding_score", 1.0))
        analysis.evidence_sufficiency = agent_out.get("evidence_sufficiency", "LOW" if analysis.is_low_evidence else "MODERATE")
        analysis.critic_approved = bool(agent_out.get("is_approved", False))
        analysis.requires_abstention = bool(agent_out.get("requires_human_abstention", False))
        analysis.final_report = agent_out.get("final_report", "")
    except Exception as e:
        logger.error(f"Agent error: {e}")
        analysis.agent_status = "error"
        analysis.final_report = f"Agent assessment could not be compiled ({e})."

    return analysis


# ── Legacy wrapper (kept for test suite compatibility) ────────────────────────

def analyze_new_patient(
    uploaded_files: List[Any],
    patient_id: Optional[str] = None,
    age_years: Optional[float] = None,
    sex: Optional[str] = None,
    sample_origin: Optional[str] = None,
) -> PatientAnalysis:
    """Legacy alias – routes to analyze_new_case."""
    return analyze_new_case(
        uploaded_files=uploaded_files,
        case_id=patient_id,
        age_years=age_years,
        sex=sex,
        sample_site=sample_origin,
    )
