"""
agents/critic.py
─────────────────────────────────────────────────────────────────────────────
Critic & Safety Verifier Agent Node.

Evaluates the draft synthesis on TWO INDEPENDENT AXES:

  A. EVIDENCE GROUNDING (0.0–1.0)
     Are the specific claims in the draft supported by the retrieved
     WHO/ICC guideline chunks?  Citation correctness check.

  B. EVIDENCE SUFFICIENCY (HIGH | MODERATE | LOW | INSUFFICIENT)
     Does the image evidence provide enough information to make any
     population-level morphological statement?
     A draft can be perfectly grounded (1.00) but still insufficient
     if only 1 cell was detected.

Final verification states:
  APPROVED           — grounding ≥ 0.75 AND sufficiency is MODERATE or HIGH
  INSUFFICIENT_EVIDENCE — too few cells, even if well-grounded
  REVIEW_REQUIRED    — grounding failed; routed to synthesizer for revision
  ABSTAINED          — max revisions exceeded; human review triggered
"""

import logging
import re
from typing import List

from agents.state import AgentState
from agents.tools import format_scientific_disclaimers

logger = logging.getLogger("agent_critic")

# ── Evidence sufficiency thresholds ──────────────────────────────────────────
_CELL_THRESHOLD_INSUFFICIENT = 5    # < 5 cells → INSUFFICIENT
_CELL_THRESHOLD_LOW          = 50   # < 50 cells → LOW
_CELL_THRESHOLD_MODERATE     = 150  # < 150 cells → MODERATE; ≥ 150 → HIGH


def _compute_evidence_sufficiency(total_cells: int, low_evidence_flag: bool) -> str:
    if total_cells < _CELL_THRESHOLD_INSUFFICIENT:
        return "INSUFFICIENT"
    if total_cells < _CELL_THRESHOLD_LOW or low_evidence_flag:
        return "LOW"
    if total_cells < _CELL_THRESHOLD_MODERATE:
        return "MODERATE"
    return "HIGH"


def critic_node(state: AgentState) -> AgentState:
    """
    Evaluates draft against guidelines (grounding) and image evidence
    volume (sufficiency) independently.
    """
    pid           = state.get("patient_id", "UNKNOWN")
    draft         = state.get("draft_differential", "")
    evidence      = state.get("retrieved_evidence", [])
    valid_cids    = {e["chunk_id"] for e in evidence}
    low_evidence  = state.get("low_evidence_flag", False)
    total_cells   = state.get("total_cells", 0)
    revisions     = state.get("revision_count", 0)

    logger.info(f"[{pid}] Critic verifying draft (revision #{revisions}) — {total_cells} cells...")

    # ── A. Evidence sufficiency (independent of grounding) ────────────────────
    sufficiency = _compute_evidence_sufficiency(total_cells, low_evidence)
    state["evidence_sufficiency"] = sufficiency

    # ── B. Citation grounding check ───────────────────────────────────────────
    hallucination_flags: List[str] = []
    penalties = 0.0

    cited_chunks = re.findall(r"\[(WHO2022_[A-Za-z0-9_]+|DISCLAIMER_[A-Za-z0-9_]+)\]", draft)
    if not cited_chunks:
        hallucination_flags.append("No authoritative clinical guideline chunks cited in draft.")
        penalties += 0.3
    else:
        for cid in cited_chunks:
            if cid not in valid_cids:
                hallucination_flags.append(f"Fabricated / unretrieved citation: [{cid}].")
                penalties += 0.4

    # Check for fabricated clinical data
    lower_draft = draft.lower()
    if "cbc reveals" in lower_draft or "karyotype showed" in lower_draft:
        hallucination_flags.append("Draft asserts CBC or Karyotype results that were not provided.")
        penalties += 0.5

    pred_cls = state.get("predicted_class", "")
    if pred_cls and pred_cls not in draft:
        hallucination_flags.append(f"Primary predicted class '{pred_cls}' not acknowledged.")
        penalties += 0.2

    grounding_score = max(0.0, min(1.0, 1.0 - penalties))

    state["critic_grounding_score"] = round(grounding_score, 3)
    state["critic_hallucination_flags"] = hallucination_flags
    state["revision_count"] = revisions + 1

    disclaimer_block = format_scientific_disclaimers(
        low_evidence=low_evidence,
        total_cells=total_cells,
        evidence_sufficiency=sufficiency,
    )

    # ── C. Final approval decision ────────────────────────────────────────────
    grounding_ok   = grounding_score >= 0.75 and len(hallucination_flags) == 0
    sufficient_ev  = sufficiency in ("MODERATE", "HIGH")

    if sufficiency == "INSUFFICIENT":
        # Well-grounded doesn't matter — not enough cells
        logger.warning(f"[{pid}] INSUFFICIENT EVIDENCE ({total_cells} cells). Overriding to abstain.")
        state["is_approved"] = False
        state["requires_human_abstention"] = True
        state["final_report"] = (
            f"> [!CAUTION]\n"
            f"> **INSUFFICIENT EVIDENCE:** Only {total_cells} cell(s) detected. "
            f"Population-level morphology cannot be assessed.\n\n"
            f"> Individual cell classifications may be displayed for reference "
            f"but population statistics are not meaningful.\n\n---\n\n{disclaimer_block}"
        )

    elif grounding_ok and sufficient_ev:
        logger.info(f"[{pid}] APPROVED (grounding={grounding_score:.2f}, sufficiency={sufficiency}).")
        state["is_approved"] = True
        state["requires_human_abstention"] = False
        state["final_report"] = f"{draft}\n\n---\n\n{disclaimer_block}"

    elif grounding_ok and not sufficient_ev:
        # Grounded but low evidence — show with prominent warning
        logger.warning(f"[{pid}] Approved with LOW EVIDENCE warning ({total_cells} cells).")
        state["is_approved"] = True
        state["requires_human_abstention"] = False
        state["final_report"] = (
            f"> [!WARNING]\n"
            f"> **LOW-EVIDENCE RESULT:** Only {total_cells} cells were detected. "
            f"The morphological distribution below carries high sampling uncertainty.\n\n"
            f"{draft}\n\n---\n\n{disclaimer_block}"
        )

    else:
        logger.warning(f"[{pid}] REJECTED (grounding={grounding_score:.2f}): {hallucination_flags}")
        state["is_approved"] = False
        state["critic_comments"] = f"Corrections needed: {'; '.join(hallucination_flags)}"

        if state["revision_count"] >= 2:
            logger.warning(f"[{pid}] Max revisions reached — triggering abstention.")
            state["requires_human_abstention"] = True
            state["final_report"] = (
                f"> [!CAUTION]\n> **AUTOMATED ABSTENTION:** Evidence validation failed after {state['revision_count']} revisions.\n\n"
                f"{draft}\n\n---\n\n{disclaimer_block}"
            )

    return state
