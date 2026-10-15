"""
pipeline/cell_detector.py
─────────────────────────────────────────────────────────────────────────────
Bone-Marrow Microscopy Cell Detector

Detects and extracts individual cell candidates from large uploaded
bone-marrow microscopy images across the entire field of view.

Core principle:
  ONE FINAL DETECTION = ONE INDIVIDUAL CELL.
  A detection must surround the complete visible cell body / nucleus region,
  NOT a small texture fragment, stain artifact, edge, or sub-region.
  Tiny fragments are suppressed by strict minimum nuclear mass and area bounds.
  Large multi-cell regions are split via scale-adaptive nuclear peak seeds;
  if an overlapping cluster cannot be cleanly resolved, it is flagged/rejected.

Pipeline Stages:
  1. Optical-density Romanowsky / Giemsa nuclear chromatin absorption contrast.
  2. Multi-scale nuclear distance transform peak extraction.
  3. Marker-controlled watershed expansion to establish cell boundaries.
  4. Candidate filtering:
     - Minimum nucleus content (rejects unnucleated stain fragments/RBC edges)
     - Minimum & maximum physical dimensions
     - Aspect ratio & circularity checks
     - Non-Maximum Suppression (NMS) to eliminate duplicate/fragment boxes.
  5. Generating THREE Debug Views:
     - View 1: Raw candidate detections
     - View 2: Filtered accepted detections
     - View 3: Contact sheet of detection crops labeled with Cell ID, bbox, and dims.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("cell_detector")

# ── Tunable parameters (in original image coordinates) ────────────────────────
MIN_CELL_AREA_PX = 350         # Minimum individual cell body area in original pixels
MAX_CELL_AREA_PX = 22_000      # Maximum individual cell area (reject giant tissue chunks)
MIN_DIM_PX = 20                # Minimum width/height (reject fragments < 20px)
MAX_DIM_PX = 130               # Maximum width/height for an individual cell
MIN_NUCLEUS_PIXELS = 100       # Minimum Azure B stained nuclear pixels inside cell
MAX_COVERAGE_PCT = 3.5         # Maximum percentage of total image area (reject multi-cell clumps)
NMS_IOU_THRESHOLD = 0.35       # IoU threshold for suppressing overlapping duplicate boxes
MARGIN_RATIO = 0.15            # Contextual padding around cell bbox
CROP_SIZE = 224                # ViT input size
MAX_CELLS_PER_IMAGE = 600      # Guard against runaway processing


class DetectionFailure(Exception):
    """Raised when the detector cannot reliably identify any individual cells."""
    pass


@dataclass
class DetectedCell:
    """Metadata + PIL crop for a single detected cell candidate."""
    image_id: str           # Source image ID
    cell_id: str            # e.g. "IMAGE-01_CELL-007"
    bbox: Tuple[int, int, int, int]   # (x, y, w, h) in source-image pixel coords
    center: Tuple[int, int]           # (cx, cy) in source-image pixels
    area_px: int            # Contour area in source pixels
    is_uncertain: bool = False        # True if borderline size, circularity, or cluster
    crop: Optional[Image.Image] = field(default=None, repr=False)  # 224x224 PIL.Image
    img_coverage_pct: float = 0.0     # Percentage of total image area covered
    confidence: float = 1.0           # Detection confidence (geometric/color completeness)
    nuc_pixel_count: int = 0          # Number of nuclear pixels inside bbox
    # ── Per-cell diagnostic quality metrics ───────────────────────────────────
    aspect_ratio: float = 1.0         # max(w,h) / min(w,h)  — ideal ~1.0 for round cells
    foreground_pct: float = 0.0       # % of bbox pixels that are non-background (cell body)
    is_centered: bool = True          # True when cell is not clipped by image border
    neighbour_overlap_count: int = 0  # How many other accepted cells this bbox overlaps (IoU>0.05)


# ── Public API ────────────────────────────────────────────────────────────────

def detect_cells(
    pil_image: Image.Image,
    image_id: str = "IMAGE-01",
    save_dir: Optional[Path] = None,
    min_area: int = MIN_CELL_AREA_PX,
    max_area: int = MAX_CELL_AREA_PX,
    max_cells: int = MAX_CELLS_PER_IMAGE,
) -> Tuple[List[DetectedCell], dict]:
    """
    Detect and extract individual cell candidates from a bone-marrow
    microscopy image. Guarantees 1 box = 1 individual cell.
    """
    img_rgb = _to_rgb_array(pil_image)
    h, w = img_rgb.shape[:2]
    total_img_pixels = float(h * w)

    if h < 32 or w < 32:
        raise DetectionFailure("Image is too small for cell detection (< 32 × 32 px).")

    # ── Step 1: Chromatin & Stain Discrimination ──────────────────────────────
    # Stained leukocyte/precursor nuclei absorb Green strongly and exhibit
    # characteristic purple/blue hue (Azure B / Giemsa).
    # Background illumination is dynamically estimated to set a robust upper V limit.
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    bg_v = float(np.percentile(hsv[:, :, 2], 90))
    v_thresh = max(160.0, min(195.0, bg_v - 45.0))
    is_nuc = (hsv[:, :, 0] >= 105) & (hsv[:, :, 0] <= 165) & (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] <= v_thresh)
    
    # Morphological cleanup on nucleus mask: eliminate sub-resolution speckles (< 5px)
    kernel_speckle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    nuc_clean = cv2.morphologyEx(is_nuc.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel_speckle)

    # ── Step 2: Distance Transform Peak Seeding ───────────────────────────────
    dist = cv2.distanceTransform(nuc_clean, cv2.DIST_L2, 5)

    # Scale-adaptive peak kernel: cell nuclei in typical 1000-3000px smears have radius 10-25px
    diag = np.sqrt(h * w)
    ksize = max(15, min(27, int(diag * 0.018)))
    if ksize % 2 == 0:
        ksize += 1

    peak_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    dist_dilated = cv2.dilate(dist, peak_kernel)

    min_dist_val = max(6.0, dist.max() * 0.10)
    peaks = (dist == dist_dilated) & (dist >= min_dist_val)

    num_markers, markers = cv2.connectedComponents(peaks.astype(np.uint8))

    # ── Step 3: Marker-Controlled Watershed on Cell Body ──────────────────────
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    body_v_thresh = min(220.0, bg_v - 15.0)
    is_cell_body = (gray < body_v_thresh) | is_nuc
    sure_bg = cv2.dilate(is_cell_body.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=2)
    unknown = cv2.subtract(sure_bg, peaks.astype(np.uint8) * 255)

    raw_candidates: List[Tuple[int, int, int, int, int, int]] = []  # (x, y, w, h, area, nuc_cnt)
    
    if num_markers > 1:
        markers_ws = markers.copy() + 1
        markers_ws[sure_bg == 0] = 1        # Background is marker 1
        markers_ws[unknown == 255] = 0      # Watershed boundary search

        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        markers_res = cv2.watershed(img_bgr, markers_ws)

        unique_labels = np.unique(markers_res)
        unique_labels = unique_labels[unique_labels > 1]

        for lbl in unique_labels:
            mask = (markers_res == lbl).astype(np.uint8)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            x, y, bw, bh = cv2.boundingRect(cnt)
            nuc_in_box = int(np.count_nonzero(is_nuc[y:y+bh, x:x+bw]))
            raw_candidates.append((x, y, bw, bh, int(area), nuc_in_box))

    raw_candidates_count = len(raw_candidates)

    # ── Step 4: Strict Filtering & Cluster Rejection ──────────────────────────
    rejected_reasons: Dict[str, int] = {
        "too_small_area": 0,
        "too_large_cluster": 0,
        "dim_too_small": 0,
        "dim_too_large": 0,
        "insufficient_nuclear_mass": 0,
        "excessive_coverage": 0,
        "suppressed_by_nms": 0,
    }

    filtered_candidates: List[Tuple[int, int, int, int, int, int, float]] = []

    for (x, y, bw, bh, area, nuc_cnt) in raw_candidates:
        coverage_pct = (bw * bh) / total_img_pixels * 100.0

        if area < min_area:
            rejected_reasons["too_small_area"] += 1
            continue
        if area > max_area:
            rejected_reasons["too_large_cluster"] += 1
            continue
        if bw < MIN_DIM_PX or bh < MIN_DIM_PX:
            rejected_reasons["dim_too_small"] += 1
            continue
        if bw > MAX_DIM_PX or bh > MAX_DIM_PX:
            rejected_reasons["dim_too_large"] += 1
            continue
        if nuc_cnt < MIN_NUCLEUS_PIXELS:
            rejected_reasons["insufficient_nuclear_mass"] += 1
            continue
        if coverage_pct > MAX_COVERAGE_PCT:
            rejected_reasons["excessive_coverage"] += 1
            continue

        # Confidence based on nuclear density and compactness
        det_conf = min(1.0, max(0.5, (nuc_cnt / max(1.0, bw * bh * 0.35))))
        filtered_candidates.append((x, y, bw, bh, area, nuc_cnt, det_conf))

    # ── Step 5: Non-Maximum Suppression (NMS) to eliminate duplicate boxes ────
    # Sort candidates by nuclear density / area
    filtered_candidates.sort(key=lambda c: (c[5], c[4]), reverse=True)

    def _compute_iou(b1, b2):
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[0] + b1[2], b2[0] + b2[2])
        yB = min(b1[1] + b1[3], b2[1] + b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        area1 = b1[2] * b1[3]
        area2 = b2[2] * b2[3]
        return inter / float(area1 + area2 - inter + 1e-6)

    kept_candidates = []
    for cand in filtered_candidates:
        if any(_compute_iou(cand, kept) > NMS_IOU_THRESHOLD for kept in kept_candidates):
            rejected_reasons["suppressed_by_nms"] += 1
            continue
        kept_candidates.append(cand)

    # ── Step 6: Extract Padded Crops & Build DetectedCell objects ─────────────
    detected: List[DetectedCell] = []
    cell_counter = 0

    for (x, y, bw, bh, area, nuc_cnt, conf) in kept_candidates:
        if cell_counter >= max_cells:
            break

        coverage_pct = (bw * bh) / total_img_pixels * 100.0
        aspect_ratio = max(bw, bh) / (min(bw, bh) + 1e-6)
        is_uncertain = (aspect_ratio > 2.5) or (nuc_cnt < 140)

        # Foreground %: pixels in bbox that are part of the cell body (non-white)
        patch_gray = cv2.cvtColor(img_rgb[y:y+bh, x:x+bw], cv2.COLOR_RGB2GRAY)
        fg_pixels = int(np.count_nonzero(patch_gray < int(min(220.0, bg_v - 5.0))))
        foreground_pct = round((fg_pixels / max(1, bw * bh)) * 100.0, 2)

        # Centring: True if bbox does not clip image borders
        border_margin = max(5, int(min(h, w) * 0.02))
        is_centered = (
            x > border_margin and y > border_margin
            and (x + bw) < (w - border_margin)
            and (y + bh) < (h - border_margin)
        )

        cx, cy = x + bw // 2, y + bh // 2

        cell_crop = _extract_padded_crop(img_rgb, x, y, bw, bh, MARGIN_RATIO, CROP_SIZE)
        if cell_crop is None:
            continue

        cell_counter += 1
        cid = f"{image_id}_CELL-{cell_counter:03d}"

        dc = DetectedCell(
            image_id=image_id,
            cell_id=cid,
            bbox=(x, y, bw, bh),
            center=(cx, cy),
            area_px=area,
            is_uncertain=is_uncertain,
            crop=cell_crop,
            img_coverage_pct=round(coverage_pct, 4),
            confidence=round(conf, 3),
            nuc_pixel_count=nuc_cnt,
            aspect_ratio=round(aspect_ratio, 2),
            foreground_pct=foreground_pct,
            is_centered=is_centered,
            # neighbour_overlap_count computed after all cells are collected
        )

        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"{cid}.png"
            cell_crop.save(save_path)
            dc._save_path = str(save_path)  # type: ignore[attr-defined]

        detected.append(dc)

    # ── Post-loop: compute neighbour_overlap_count for each accepted cell ──────
    for i, dc_i in enumerate(detected):
        overlap_cnt = 0
        for j, dc_j in enumerate(detected):
            if i == j:
                continue
            iou = _compute_iou(dc_i.bbox, dc_j.bbox)
            if iou > 0.05:
                overlap_cnt += 1
        dc_i.neighbour_overlap_count = overlap_cnt

    if not detected:
        raise DetectionFailure(
            "Individual cells could not be reliably detected in this image. "
            "Verify the image is a bone-marrow smear (Romanowsky/Giemsa/Wright stain)."
        )

    # ── Step 7: QC Statistics & Debug Summaries ───────────────────────────────
    widths     = [c.bbox[2] for c in detected]
    heights    = [c.bbox[3] for c in detected]
    coverages  = [c.img_coverage_pct for c in detected]
    fg_pcts    = [c.foreground_pct for c in detected]
    asp_ratios = [c.aspect_ratio for c in detected]

    # Per-cell quality summary list for external consumers (e.g. UI diagnostics)
    per_cell_quality = [
        {
            "cell_id":           dc.cell_id,
            "bbox":              list(dc.bbox),
            "area_px":           dc.area_px,
            "aspect_ratio":      dc.aspect_ratio,
            "foreground_pct":    dc.foreground_pct,
            "nuc_pixel_count":   dc.nuc_pixel_count,
            "is_centered":       dc.is_centered,
            "neighbour_overlaps": dc.neighbour_overlap_count,
            "is_uncertain":      dc.is_uncertain,
            "confidence":        dc.confidence,
        }
        for dc in detected
    ]

    qc = {
        "image_id": image_id,
        "image_dims": (w, h),
        "raw_candidates_count": raw_candidates_count,
        "accepted_cells": len(detected),
        "rejected_count": sum(rejected_reasons.values()),
        "rejected_reasons": rejected_reasons,
        "uncertain_cells": sum(1 for c in detected if c.is_uncertain),
        "median_bbox_dims": (float(np.median(widths)), float(np.median(heights))),
        "min_bbox_dims": (int(min(widths)), int(min(heights))),
        "max_bbox_dims": (int(max(widths)), int(max(heights))),
        "median_coverage_pct": float(np.median(coverages)),
        "max_coverage_pct": float(max(coverages)),
        "median_foreground_pct": float(np.median(fg_pcts)),
        "median_aspect_ratio": float(np.median(asp_ratios)),
        "border_clipped_cells": sum(1 for c in detected if not c.is_centered),
        "cells_with_overlaps": sum(1 for c in detected if c.neighbour_overlap_count > 0),
        "per_cell_quality": per_cell_quality,
        "raw_candidates": raw_candidates,
        "detection_method": "nucleus_anchored_watershed_with_nms",
    }

    logger.info(
        f"[{image_id}] Detection complete: {len(detected)} accepted genuine cells "
        f"(from {raw_candidates_count} raw candidates, {qc['rejected_count']} rejected). "
        f"Median dims: {qc['median_bbox_dims']} px."
    )
    return detected, qc


# ── Debug View Generators ─────────────────────────────────────────────────────

def generate_detection_debug_views(
    pil_image: Image.Image,
    raw_candidates: List[Tuple[int, int, int, int, int, int]],
    filtered_cells: List[DetectedCell],
    max_crops_in_sheet: int = 50,
) -> Tuple[Image.Image, Image.Image, Image.Image]:
    """
    Generates THREE debug inspection views:
      View 1: Original image with all raw candidate detections.
      View 2: Original image with filtered accepted detections.
      View 3: Contact sheet of up to 50 final detection crops with labels.
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    h, w = img_rgb.shape[:2]

    # ── View 1: Raw candidate detections (Gray/Magenta) ──────────────────────
    view1_bgr = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
    for (x, y, bw, bh, area, nuc_cnt) in raw_candidates:
        cv2.rectangle(view1_bgr, (x, y), (x + bw, y + bh), (255, 0, 255), 1)
    view1_img = Image.fromarray(cv2.cvtColor(view1_bgr, cv2.COLOR_BGR2RGB))

    # ── View 2: Filtered accepted detections (Cyan / Orange) ──────────────────
    view2_img = annotate_detections_on_image(pil_image, filtered_cells)

    # ── View 3: Contact sheet of final detection crops ────────────────────────
    # Columns: CELL ID | BBox | Crop | Detection Score + diagnostic metrics
    crops_to_show = filtered_cells[:max_crops_in_sheet]
    n_crops = len(crops_to_show)
    if n_crops == 0:
        view3_img = Image.new("RGB", (300, 100), (255, 255, 255))
    else:
        cols = 5
        rows = (n_crops + cols - 1) // cols
        tile_w, tile_h = 140, 178   # taller to fit all diagnostic rows
        sheet_w = cols * tile_w
        hdr_h = 28
        sheet_h = rows * tile_h + hdr_h

        view3_img = Image.new("RGB", (sheet_w, sheet_h), (240, 242, 248))
        draw = ImageDraw.Draw(view3_img)

        # Header bar
        draw.rectangle([0, 0, sheet_w, hdr_h - 1], fill=(30, 58, 138))
        draw.text(
            (8, 7),
            f"Contact Sheet  |  {n_crops} cells shown (of {len(filtered_cells)} accepted)"
            f"  |  CELL ID | BBox | Crop | Score | FG% | AR | Ctr | Ovlp",
            fill=(220, 230, 255),
        )

        for idx, cell in enumerate(crops_to_show):
            c_col = idx % cols
            c_row = idx // cols
            tx = c_col * tile_w
            ty = c_row * tile_h + hdr_h

            # Tile background
            bg_col = (248, 249, 252) if (idx % 2 == 0) else (242, 244, 250)
            draw.rectangle([tx, ty, tx + tile_w - 1, ty + tile_h - 1], fill=bg_col)

            # Crop thumbnail (104x104)
            if cell.crop:
                thumb = cell.crop.resize((104, 104), Image.Resampling.BICUBIC)
                view3_img.paste(thumb, (tx + 18, ty + 6))

            # Border: cyan=confident, orange=uncertain
            border_col = (2, 132, 199) if not cell.is_uncertain else (234, 88, 12)
            draw.rectangle([tx + 18, ty + 6, tx + 122, ty + 110], outline=border_col, width=2)

            # --- Text rows: CELL ID | BBox | Detection Score | FG% AR Ctr Ovlp ---
            short_id = cell.cell_id.split("_")[-1]
            bx_x, bx_y, bx_w, bx_h = cell.bbox
            ry = ty + 114
            # Row 0: Cell ID (bold-like via colour)
            draw.text((tx + 4, ry),      f"{short_id}", fill=(15, 23, 42))
            # Row 1: BBox location + size
            draw.text((tx + 4, ry + 12), f"BBox:{bx_w}x{bx_h} @{bx_x},{bx_y}", fill=(71, 85, 105))
            # Row 2: Detection score
            draw.text((tx + 4, ry + 24), f"Score:{cell.confidence:.0%}  AR:{cell.aspect_ratio:.1f}", fill=(2, 132, 199))
            # Row 3: Foreground%, centred, overlaps
            draw.text((tx + 4, ry + 36), f"FG%:{cell.foreground_pct:.0f}  Ctr:{'Y' if cell.is_centered else 'N'}  Ovlp:{cell.neighbour_overlap_count}", fill=(100, 116, 139))
            # Row 4: Nuc pixels
            draw.text((tx + 4, ry + 48), f"Nuc:{cell.nuc_pixel_count}px", fill=(130, 140, 160))

    return view1_img, view2_img, view3_img


def annotate_detections_on_image(
    pil_image: Image.Image,
    cells: List[DetectedCell],
    max_display: int = 500,
) -> Image.Image:
    """
    Draw bounding boxes on the original image with Cell ID tags.
    """
    img = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    font_scale = max(0.30, min(0.55, img.shape[1] / 3000.0))
    thickness = max(1, int(img.shape[1] / 1400.0))

    for i, cell in enumerate(cells[:max_display]):
        x, y, bw, bh = cell.bbox
        color = (0, 220, 255) if not cell.is_uncertain else (0, 140, 255)
        cv2.rectangle(img_bgr, (x, y), (x + bw, y + bh), color, thickness)

        label = cell.cell_id.split("_")[-1]
        text_y = max(y - 3, 12)
        cv2.putText(
            img_bgr, label, (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA,
        )
        cv2.putText(
            img_bgr, label, (x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA,
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img_rgb)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _to_rgb_array(pil_image: Image.Image) -> np.ndarray:
    img = pil_image.convert("RGB")
    return np.array(img, dtype=np.uint8)


def _extract_padded_crop(
    img_rgb: np.ndarray,
    x: int, y: int, bw: int, bh: int,
    margin_ratio: float,
    target_size: int,
) -> Optional[Image.Image]:
    """Extract a square padded crop centred on the bounding box."""
    H, W = img_rgb.shape[:2]
    side = max(bw, bh)
    pad = int(side * margin_ratio)
    side_padded = side + 2 * pad

    cx = x + bw // 2
    cy = y + bh // 2

    x0 = max(cx - side_padded // 2, 0)
    y0 = max(cy - side_padded // 2, 0)
    x1 = min(cx + side_padded // 2, W)
    y1 = min(cy + side_padded // 2, H)

    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    patch = img_rgb[y0:y1, x0:x1]
    pil_crop = Image.fromarray(patch).resize((target_size, target_size), Image.Resampling.BICUBIC)
    return pil_crop
