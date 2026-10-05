"""
context/assembler.py
─────────────────────────────────────────────────────────────────────────────
Deterministic Structured Patient Context Assembler.

Assembles multimodal clinical information:
- Vision-derived cell-differential vector & blast %
- Quantitative laboratory panel (CBC, LDH, etc.)
- Cytogenetic & molecular diagnostic findings
- Clinical narrative notes & patient demographics

Formats a unified, grounded structured context block passed directly
to the LLM Synthesizer without learned cross-modal weights.
"""

from typing import Any, Dict, List, Optional
import json


def format_differential_summary(differential: Dict[str, float], blast_pct: float, total_crops: int) -> str:
    """Formats cell differential vector into clinical text."""
    # Sort classes by frequency descending
    sorted_items = sorted(differential.items(), key=lambda kv: kv[1], reverse=True)
    lines = [
        f"Analyzed Crops: N = {total_crops} cells",
        f"WHO 5th Blast Percentage: {blast_pct * 100:.1f}%",
        "Differential Breakdown:",
    ]
    for cls_name, fraction in sorted_items:
        if fraction > 0.005:  # Display classes with >= 0.5% presence
            lines.append(f"  - {cls_name}: {fraction * 100:.1f}%")
    return "\n".join(lines)


def format_laboratory_panel(labs: Dict[str, Any]) -> str:
    """Formats quantitative laboratory panel into readable clinical report lines."""
    lines = []
    unit_map = {
        "wbc_10e9_l": ("White Blood Count (WBC)", "× 10⁹/L"),
        "hb_g_dl": ("Hemoglobin (Hb)", "g/dL"),
        "platelets_10e9_l": ("Platelet Count", "× 10⁹/L"),
        "neutrophil_abs_10e9_l": ("Absolute Neutrophil Count (ANC)", "× 10⁹/L"),
        "ldh_u_l": ("Lactate Dehydrogenase (LDH)", "U/L"),
        "fibrinogen_mg_dl": ("Fibrinogen", "mg/dL"),
        "d_dimer_ug_ml": ("D-Dimer", "µg/mL"),
    }
    for key, val in labs.items():
        name, unit = unit_map.get(key, (key.replace("_", " ").title(), ""))
        lines.append(f"  - {name}: {val} {unit}".rstrip())
    return "\n".join(lines) if lines else "  - No laboratory panel recorded."


def assemble_patient_context(
    case_id: str,
    patient_profile: Dict[str, Any],
    differential: Dict[str, float],
    blast_pct: float,
    total_crops: int,
    laboratory: Dict[str, Any],
    cytogenetics: List[str],
    clinical_notes: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Assembles a deterministic structured patient context.

    Returns:
        Dict with:
          - 'case_id': string
          - 'text_block': formatted markdown text for prompt injection
          - 'json_payload': raw structured dictionary for programmatic inspection
    """
    diff_text = format_differential_summary(differential, blast_pct, total_crops)
    lab_text = format_laboratory_panel(laboratory)
    cyto_text = "\n".join([f"  - {c}" for c in cytogenetics]) if cytogenetics else "  - None reported"
    notes_text = clinical_notes or patient_profile.get("clinical_presentation", "No prior notes provided.")

    age = patient_profile.get("age", "Unknown")
    sex = patient_profile.get("sex", "Unknown")

    text_block = f"""### Structured Patient Context (Case ID: {case_id})

#### 1. Patient Profile
- Age: {age}
- Sex: {sex}

#### 2. Clinical Notes & Presentation
{notes_text}

#### 3. Bone Marrow Aspirate Cytomorphology (Vision Pipeline)
{diff_text}

#### 4. Laboratory Blood & Serum Panel
{lab_text}

#### 5. Cytogenetic & Molecular Diagnostic Findings
{cyto_text}
"""

    json_payload = {
        "case_id": case_id,
        "profile": {"age": age, "sex": sex},
        "clinical_notes": notes_text,
        "morphology": {
            "total_crops": total_crops,
            "blast_pct": blast_pct,
            "differential": differential,
        },
        "laboratory": laboratory,
        "cytogenetics": cytogenetics,
    }

    return {
        "case_id": case_id,
        "text_block": text_block,
        "json_payload": json_payload,
    }
