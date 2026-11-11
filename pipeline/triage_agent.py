# -*- coding: utf-8 -*-
"""
Triage Agent: Human-in-the-loop review queue for uncertain or critical cell classifications.
Ranks detected cells using:
  - Low confidence (<0.50)
  - Blast-adjacent lineages (Blast vs Promyelocyte vs Myeloid)
  - Overlap flags (IoU > 0.50)
  - Top-1/Top-2 margin

Follows global agent rules:
  - Reads data only via build_morphology_summary() and stored cell results.
  - Proposals only; modifications require explicit physician confirmation.
  - Every step logged to agent_events table.
  - Capped at 6 tool calls and 20 seconds.
"""

from typing import Any, Dict, List, Optional
from pipeline.summary import build_morphology_summary, MORPH_CLASSES
from pipeline.agent_guard import AgentExecutionGuard, sanitize_agent_language


def compute_iou(box1: List[int], box2: List[int]) -> float:
    """Compute IoU between two [x, y, w, h] bounding boxes."""
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0

    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return float(inter_area / union_area)


def build_triage_queue(case_or_analysis_id: str, case_dict: Optional[dict] = None) -> Dict[str, Any]:
    """
    Triage Agent: builds an ordered queue of candidate cells requiring physician verification.
    """
    guard = AgentExecutionGuard(agent_name="Triage agent", case_id=str(case_or_analysis_id))

    # Tool Call 1: Load authoritative summary
    def _tool_get_summary():
        return build_morphology_summary(case_or_analysis_id, case_dict=case_dict, use_reviewer_labels=False)

    summary = guard.execute_tool("build_morphology_summary", _tool_get_summary)

    # Tool Call 2: Load cells and detection images
    def _tool_get_cells():
        resolved_case = case_dict
        if resolved_case is None:
            try:
                from app.db import get_case_from_db
                resolved_case = get_case_from_db(case_or_analysis_id)
            except Exception:
                pass
        if resolved_case is None:
            try:
                from app.flask_app import analyses
                resolved_case = analyses.get(case_or_analysis_id)
            except Exception:
                pass

        cells = (resolved_case or {}).get("cell_images") or []
        if not cells and resolved_case and resolved_case.get("analysis"):
            from app.flask_app import _cell_to_dict
            cells = [_cell_to_dict(c) for c in resolved_case["analysis"].cell_predictions]
        return cells or [], (resolved_case or {}).get("xai_examples") or []

    raw_cells, xai_examples = guard.execute_tool("get_cells", _tool_get_cells)

    # Map overlay paths from XAI examples
    overlay_map = {}
    for x in xai_examples:
        cid = getattr(x, "cell_id", None) or (x.get("cell_id") if isinstance(x, dict) else None)
        path = getattr(x, "overlay_path", None) or (x.get("overlay_path") if isinstance(x, dict) else None)
        if cid and path:
            overlay_map[cid] = str(path)

    # Tool Call 3: Find overlapping boxes
    def _tool_check_overlaps():
        overlaps = set()
        for i in range(len(raw_cells)):
            b1 = raw_cells[i].get("bbox")
            if not b1:
                continue
            for j in range(i + 1, len(raw_cells)):
                b2 = raw_cells[j].get("bbox")
                if b2 and compute_iou(b1, b2) > 0.50:
                    cid1 = raw_cells[i].get("id") or raw_cells[i].get("cell_id") or str(i)
                    cid2 = raw_cells[j].get("id") or raw_cells[j].get("cell_id") or str(j)
                    overlaps.add(cid1)
                    overlaps.add(cid2)
        return overlaps

    overlap_cids = guard.execute_tool("check_overlaps", _tool_check_overlaps)

    # Tool Call 4: Rank cells and build triage queue
    def _tool_rank_queue():
        candidates = []
        blast_lineage = {"Blast", "Promyelocyte", "Myeloid (Maturing)"}

        for i, c in enumerate(raw_cells):
            cid = c.get("id") or c.get("cell_id") or f"CELL-{i+1:03d}"
            ai_cls = c.get("class") or c.get("predicted_class") or "Other Morphological"
            conf = c.get("confidence")
            conf_val = float(conf) if conf is not None else 1.0
            rev_label = c.get("reviewer_label")

            is_low_conf = conf_val < 0.50
            is_blast_adjacent = ai_cls in blast_lineage
            is_overlap = cid in overlap_cids

            # Primary review criteria: confidence < 0.50 or spatial overlap
            if is_low_conf or is_overlap:
                # Score formula: lower confidence gives higher priority
                score = (1.0 - conf_val) * 100.0
                if ai_cls == "Blast":
                    score += 40.0
                elif is_blast_adjacent:
                    score += 20.0
                if is_overlap:
                    score += 15.0

                # Formulate clinical rationale
                if ai_cls == "Blast" and conf_val < 0.50:
                    reason = f"{conf_val*100:.1f}% blast, close to promyelocyte; blast-adjacent cell"
                elif is_overlap:
                    reason = f"Overlapping box (IoU > 0.50) with adjacent cell; check for double-counting"
                elif ai_cls == "Blast":
                    reason = f"Blast cell ({conf_val*100:.1f}% confidence); manual confirmation required"
                elif is_blast_adjacent and is_low_conf:
                    reason = f"{conf_val*100:.1f}% confidence ({ai_cls}) - blast-adjacent lineage; review recommended"
                else:
                    reason = f"{conf_val*100:.1f}% confidence ({ai_cls}) - below 50% confidence threshold"

                img_path = c.get("image_path")
                overlay_path = c.get("overlay_path") or overlay_map.get(cid)

                candidates.append({
                    "cell_id": cid,
                    "index": i,
                    "ai_class": ai_cls,
                    "current_label": rev_label or ai_cls,
                    "reviewer_label": rev_label,
                    "is_reviewed": bool(rev_label),
                    "confidence": round(conf_val, 4),
                    "priority_score": round(score, 2),
                    "reason": sanitize_agent_language(reason),
                    "is_blast_adjacent": is_blast_adjacent,
                    "is_overlap": is_overlap,
                    "image_path": img_path,
                    "overlay_path": overlay_path,
                })

        # Sort highest priority first
        candidates.sort(key=lambda x: x["priority_score"], reverse=True)
        return candidates

    ranked_queue = guard.execute_tool("rank_queue", _tool_rank_queue)

    # Tool Call 5: Summarize queue stats
    def _tool_summarize_queue():
        return {
            "total_candidates": len(ranked_queue),
            "low_conf_count": sum(1 for c in ranked_queue if c["confidence"] < 0.50),
            "blast_count": sum(1 for c in ranked_queue if c["ai_class"] == "Blast"),
            "reviewed_count": sum(1 for c in ranked_queue if c["is_reviewed"]),
        }

    stats = guard.execute_tool("summarize_queue", _tool_summarize_queue)

    return {
        "case_id": summary["case_id"],
        "analysis_id": summary["analysis_id"],
        "total_cells": summary["total_cells"],
        "initial_blast_fraction": summary["blast_fraction"],
        "initial_blast_count": summary["blast_count"],
        "queue": ranked_queue,
        "queue_length": len(ranked_queue),
        "stats": stats,
        "classes": MORPH_CLASSES,
        "history": guard.history,
    }
