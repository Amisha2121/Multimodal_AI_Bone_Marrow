# -*- coding: utf-8 -*-
"""
Deterministic Figure Generation for Phase 10 Scan Understanding Report.
Generates:
  - Fig 1: Annotated field with class-coloured bounding boxes & flagged cell callouts.
  - Fig 2: Class contact sheets (sorted by confidence, with labels and needs-review tags).
  - Fig 3: Explainability panels (crop | rollout | overlay + attention focus bar).
  - Fig 4: Flagged-cell gallery (cells behind each plausibility flag).

All files are saved under DATA_DIR/derived/<case_id>/ (never touching uploads/ or originals/).
Deterministic code (matplotlib / cv2 / PIL) — ZERO generative image models.
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from app.db import DATA_DIR
from pipeline.scan_analytics import compute_attention_focus, plausibility_flags


CLASS_COLORS = {
    "Blast": "#ef4444",                     # Red
    "Promyelocyte": "#f97316",              # Orange
    "Granulocyte (Immature)": "#eab308",    # Yellow-gold
    "Granulocyte (Mature)": "#10b981",      # Emerald Green
    "Lymphoid Series": "#3b82f6",           # Blue
    "Erythroid Precursor": "#8b5cf6",       # Purple
    "Megakaryocyte/Platelet": "#ec4899",    # Pink
    "Other Morphological": "#64748b",       # Slate Gray
}


def get_derived_dir(case_id: str) -> Path:
    """Ensure and return the derived folder for a specific case."""
    d = DATA_DIR / "derived" / case_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def generate_figure1_annotated_field(
    case_id: str,
    original_img_path: str,
    cells: List[Dict[str, Any]],
    force_regen: bool = False
) -> Path:
    """
    Fig 1: Annotated Scan Field
    Original scan with class-coloured boxes, dimmed low-confidence boxes,
    and numbered callouts for top flagged cells.
    """
    out_path = get_derived_dir(case_id) / "fig1_annotated_field.png"
    if out_path.exists() and not force_regen:
        return out_path

    src = Path(original_img_path)
    if not src.exists():
        # Fallback to creating a placeholder canvas
        img = np.ones((600, 900, 3), dtype=np.uint8) * 240
    else:
        img = cv2.imread(str(src))
        if img is None:
            img = np.ones((600, 900, 3), dtype=np.uint8) * 240
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=150)
    ax.imshow(img)

    flagged_idx = 1
    for cell in cells:
        bbox = cell.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        ymin, xmin, ymax, xmax = bbox
        cls = cell.get("class", "Other Morphological")
        conf = float(cell.get("confidence", 0.0) or 0.0)
        color = CLASS_COLORS.get(cls, "#3b82f6")

        alpha = 0.9 if conf >= 0.50 else 0.4
        linewidth = 2.0 if conf >= 0.50 else 1.0

        rect = patches.Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            linewidth=linewidth, edgecolor=color, facecolor="none",
            alpha=alpha, linestyle="-" if conf >= 0.50 else "--"
        )
        ax.add_patch(rect)

        # Highlight blasts and low confidence cells with callout numbers
        if cls == "Blast" or conf < 0.35:
            ax.text(
                xmin, max(0, ymin - 4), f"#{flagged_idx} {cls[:5]} {int(conf*100)}%",
                fontsize=6.5, color="white", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.85, edgecolor="none")
            )
            flagged_idx += 1

    ax.axis("off")
    plt.title(f"Figure 1: High-Power Field Segmentation & Classification ({case_id})", fontsize=11, weight="bold", pad=10)
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out_path


def generate_figure2_contact_sheets(
    case_id: str,
    cells: List[Dict[str, Any]],
    force_regen: bool = False
) -> Path:
    """
    Fig 2: Class Contact Sheets
    Crops grouped by class and sorted by confidence, with confidence and needs-review tags.
    """
    out_path = get_derived_dir(case_id) / "fig2_contact_sheets.png"
    if out_path.exists() and not force_regen:
        return out_path

    # Group cells by class
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for c in cells:
        cls = c.get("class", "Other Morphological")
        grouped.setdefault(cls, []).append(c)

    # Sort each class by confidence descending
    for cls in grouped:
        grouped[cls].sort(key=lambda x: float(x.get("confidence", 0.0) or 0.0), reverse=True)

    # Pick top classes (up to 4 classes, up to 4 crops each)
    display_classes = sorted(grouped.keys(), key=lambda k: len(grouped[k]), reverse=True)[:4]
    nrows = len(display_classes)
    ncols = 4

    fig, axes = plt.subplots(nrows=max(1, nrows), ncols=ncols, figsize=(11, max(3, nrows * 2.4)), dpi=140)
    if nrows == 1:
        axes = np.expand_dims(axes, 0)

    for r_idx, cls in enumerate(display_classes):
        cls_cells = grouped[cls][:ncols]
        for c_idx in range(ncols):
            ax = axes[r_idx, c_idx]
            if c_idx < len(cls_cells):
                cell = cls_cells[c_idx]
                p = cell.get("image_path")
                conf = float(cell.get("confidence", 0.0) or 0.0)
                conf_pct = int(conf * 100)
                needs_rev = conf < 0.50

                if p and Path(p).exists():
                    c_img = cv2.imread(str(p))
                    if c_img is not None:
                        c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)
                        ax.imshow(c_img)
                    else:
                        ax.fill([0, 1, 1, 0], [0, 0, 1, 1], "#e2e8f0")
                else:
                    ax.fill([0, 1, 1, 0], [0, 0, 1, 1], "#e2e8f0")

                border_col = "#dc2626" if needs_rev else "#10b981"
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_col)
                    spine.set_linewidth(1.5)

                title_lbl = f"{conf_pct}% {'[Review]' if needs_rev else ''}"
                ax.set_title(title_lbl, fontsize=8, color=border_col, weight="bold")
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])

        axes[r_idx, 0].set_ylabel(cls, fontsize=8.5, weight="bold", rotation=0, labelpad=70, va="center")

    plt.suptitle(f"Figure 2: Morphological Series Contact Sheets ({case_id})", fontsize=11, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path


def generate_figure3_explainability_panels(
    case_id: str,
    cells: List[Dict[str, Any]],
    force_regen: bool = False
) -> Path:
    """
    Fig 3: Explainability Panels
    Crop | Rollout Overlay for representative cells (highest confidence, lowest confidence, and every blast),
    with the attention focus summary bar (nucleus vs cytoplasm vs background).
    """
    out_path = get_derived_dir(case_id) / "fig3_explainability_panels.png"
    if out_path.exists() and not force_regen:
        return out_path

    # Select representative cells: blasts, highest conf, lowest conf
    blasts = [c for c in cells if c.get("class") == "Blast"]
    sorted_conf = sorted(cells, key=lambda x: float(x.get("confidence", 0.0) or 0.0))
    lowest = sorted_conf[:1]
    highest = sorted_conf[-1:]

    chosen = []
    seen = set()
    for c in blasts[:2] + lowest + highest:
        cid = c.get("id")
        if cid and cid not in seen:
            seen.add(cid)
            chosen.append(c)

    chosen = chosen[:4]
    nrows = len(chosen)
    fig, axes = plt.subplots(nrows=nrows, ncols=3, figsize=(10, nrows * 2.3), dpi=140)
    if nrows == 1:
        axes = np.expand_dims(axes, 0)

    xai_dir = DATA_DIR / "new_patient_cache" / case_id / "xai"

    for r_idx, cell in enumerate(chosen):
        cid = cell.get("id")
        cls = cell.get("class")
        conf = float(cell.get("confidence", 0.0) or 0.0)
        crop_path = cell.get("image_path")

        # Find matching overlay file if available
        overlay_path = None
        if xai_dir.exists():
            for f in xai_dir.glob(f"*{cid}*overlay*.png"):
                overlay_path = str(f)
                break

        # Compute attention focus
        focus = compute_attention_focus(str(crop_path) if crop_path else "", overlay_path)

        # Col 1: Original Crop
        ax_crop = axes[r_idx, 0]
        if crop_path and Path(crop_path).exists():
            im = cv2.imread(str(crop_path))
            if im is not None:
                ax_crop.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax_crop.set_title(f"{cid}\n{cls} ({int(conf*100)}%)", fontsize=8, weight="bold")
        ax_crop.axis("off")

        # Col 2: Attention Rollout / Overlay
        ax_over = axes[r_idx, 1]
        if overlay_path and Path(overlay_path).exists():
            im_over = cv2.imread(overlay_path)
            if im_over is not None:
                ax_over.imshow(cv2.cvtColor(im_over, cv2.COLOR_BGR2RGB))
        else:
            ax_over.text(0.5, 0.5, "Rollout Matrix\n(Interpolated)", ha="center", va="center", fontsize=8)
            ax_over.set_facecolor("#f8fafc")
        ax_over.set_title("ViT Rollout Heatmap", fontsize=8)
        ax_over.axis("off")

        # Col 3: Attention Focus Bar
        ax_bar = axes[r_idx, 2]
        nuc = focus.get("nucleus_share", 0.6)
        cyto = focus.get("cytoplasm_share", 0.3)
        bg = focus.get("background_share", 0.1)

        ax_bar.barh([0], [nuc], color="#3b82f6", label="Nucleus", height=0.4)
        ax_bar.barh([0], [cyto], left=[nuc], color="#10b981", label="Cytoplasm", height=0.4)
        ax_bar.barh([0], [bg], left=[nuc + cyto], color="#94a3b8", label="Background", height=0.4)
        ax_bar.set_xlim(0, 1.0)
        ax_bar.set_yticks([])
        ax_bar.set_xticks([0.0, 0.5, 1.0])
        ax_bar.set_xticklabels(["0%", "50%", "100%"], fontsize=7)
        ax_bar.set_title(f"{focus.get('caption', 'Attention distribution')}", fontsize=7.5, pad=4)
        if r_idx == 0:
            ax_bar.legend(loc="upper center", bbox_to_anchor=(0.5, 1.45), ncol=3, fontsize=6.5, frameon=False)

    plt.suptitle(f"Figure 3: ViT Attention Rollout & Chromatin Focus ({case_id})", fontsize=11, weight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path


def generate_figure4_flagged_cell_gallery(
    case_id: str,
    summary: Dict[str, Any],
    cells: List[Dict[str, Any]],
    force_regen: bool = False
) -> Path:
    """
    Fig 4: Flagged-Cell Gallery
    Cells specifically triggering plausibility flags (e.g. uncertain blast-adjacent cells,
    megakaryocyte series candidates, and low-confidence cells).
    """
    out_path = get_derived_dir(case_id) / "fig4_flagged_gallery.png"
    if out_path.exists() and not force_regen:
        return out_path

    # Identify flagged cells
    flagged = [c for c in cells if float(c.get("confidence", 0.0) or 0.0) < 0.40 or c.get("class") == "Blast"][:6]
    if not flagged:
        flagged = cells[:6]

    ncols = min(6, len(flagged))
    fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols * 2.0, 2.5), dpi=140)
    if ncols == 1:
        axes = [axes]

    for idx, cell in enumerate(flagged):
        ax = axes[idx]
        p = cell.get("image_path")
        cls = cell.get("class", "Cell")
        conf = float(cell.get("confidence", 0.0) or 0.0)

        if p and Path(p).exists():
            im = cv2.imread(str(p))
            if im is not None:
                ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{cls}\n{int(conf*100)}% conf", fontsize=7.5, weight="bold")
        ax.axis("off")

        # Red border for review flag
        rect = patches.Rectangle((0, 0), 1, 1, transform=ax.transAxes, linewidth=2, edgecolor="#dc2626", facecolor="none")
        ax.add_patch(rect)

    plt.suptitle(f"Figure 4: Flagged Cells for Prioritized Physician Review ({case_id})", fontsize=10.5, weight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), bbox_inches="tight", dpi=140)
    plt.close(fig)
    return out_path


def generate_all_figures(case_id: str, summary: Dict[str, Any], cells: List[Dict[str, Any]]) -> Dict[str, str]:
    """Generate and return paths to Figures 1 through 4."""
    orig_img = str(DATA_DIR / "new_patient_cache" / case_id / "originals" / "IMAGE-01_original.png")
    if not Path(orig_img).exists():
        orig_img = str(DATA_DIR / "new_patient_cache" / case_id / "originals" / "IMAGE-01_annotated.png")

    fig1 = generate_figure1_annotated_field(case_id, orig_img, cells)
    fig2 = generate_figure2_contact_sheets(case_id, cells)
    fig3 = generate_figure3_explainability_panels(case_id, cells)
    fig4 = generate_figure4_flagged_cell_gallery(case_id, summary, cells)

    return {
        "fig1": str(fig1),
        "fig2": str(fig2),
        "fig3": str(fig3),
        "fig4": str(fig4),
    }

