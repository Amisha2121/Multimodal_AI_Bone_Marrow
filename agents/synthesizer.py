"""
agents/synthesizer.py
─────────────────────────────────────────────────────────────────────────────
Synthesizer Agent Node.

Evidence provenance is explicit throughout:
  OBSERVED_FROM_IMAGE  — detected cells, blast fraction
  PROVIDED_BY_USER     — age (if entered), sex (if entered)
  RETRIEVED            — WHO/ICC guideline passages
  NOT_AVAILABLE        — CBC, cytogenetics, molecular, flow cytometry

Rules:
  - Never say "pediatric" unless age < 18 is explicitly confirmed.
  - Never infer age, sex, or diagnosis from morphology alone.
  - Never assert CBC / cytogenetics / molecular data that was not provided.
  - Distinguish MORPHOLOGY OBSERVATION from DISEASE DIAGNOSIS.
"""

import logging
from typing import Dict, Any, List

from agents.state import AgentState
from agents.tools import evaluate_blast_cutoffs

logger = logging.getLogger("agent_synthesizer")


def _age_context(age_years: float | None, age_months: float | None) -> str:
    """
    Build an honest age description string.
    Returns 'Not provided' when age is unknown — never fabricates context.
    """
    if age_years is not None and age_years > 0:
        return f"{age_years:.1f} years"
    if age_months is not None and age_months > 0:
        return f"{age_months / 12.0:.1f} years"
    return "Not provided"


def _is_pediatric(age_years: float | None, age_months: float | None) -> bool | None:
    """
    Returns True only if age is confirmed < 18.
    Returns None (unknown) if age is not provided.
    """
    if age_years is not None:
        return age_years < 18
    if age_months is not None:
        return age_months < 216   # 18 × 12
    return None   # unknown


def build_deterministic_synthesis(state: AgentState) -> Dict[str, Any]:
    """
    Case-specific clinical synthesis grounded strictly in retrieved evidence.
    """
    pid           = state.get("patient_id", "UNKNOWN")
    cls           = state.get("predicted_class", "Other")
    probs         = state.get("class_probabilities", {})
    blast_fraction = state.get("annotated_blast_fraction", 0.0)
    total_cells   = state.get("total_cells", 0)
    evidence      = state.get("retrieved_evidence", [])
    age_months    = state.get("age_months", None)
    age_years     = state.get("age_years", None)
    sex           = state.get("sex", None)
    low_ev        = state.get("low_evidence_flag", False)

    age_str    = _age_context(age_years, age_months)
    pediatric  = _is_pediatric(age_years, age_months)   # True | False | None
    blast_eval = evaluate_blast_cutoffs(blast_fraction, cls)

    # Evidence chunk IDs actually retrieved
    retrieved_cids = {e["chunk_id"] for e in evidence}
    cited_ids = [e["chunk_id"] for e in evidence[:3]]
    cited_str = ", ".join(f"[{cid}]" for cid in cited_ids) if cited_ids else "(retrieved guidelines)"

    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    primary_dx   = sorted_probs[0][0] if sorted_probs else cls
    primary_conf = sorted_probs[0][1] if sorted_probs else 0.0
    secondary_dx = sorted_probs[1][0] if len(sorted_probs) > 1 else "—"

    sex_str = sex if sex else "Not provided"

    lines = [
        f"### Cytomorphology Assessment — Case `{pid}`",
        "",
        "#### Evidence Provenance",
        f"| Source | Information |",
        f"|--------|-------------|",
        f"| **Observed from image** | {total_cells} automatically detected regions; image-derived cell distribution as shown below |",
        f"| **Provided by user** | Age: {age_str} \u00b7 Sex: {sex_str} |",
        f"| **Retrieved guidelines** | WHO 5th Ed / ICC 2022 ({len(evidence)} passages) |",
        f"| **Not available** | CBC \u00b7 Cytogenetics \u00b7 Molecular genetics \u00b7 Flow cytometry |",
        "",
        "#### 1. Image-Derived Morphological Findings",
        f"Automated image analysis of {total_cells} detected regions yielded a predominant morphological classification of "
        f"**{primary_dx}** (derived from per-cell classifier votes).",
        f"The detected-cell blast proportion is **{blast_fraction:.1%}** ({blast_eval['rules_matched'][0]}).",
        "",
        "> [!IMPORTANT]",
        "> **Scope limitation:** This assessment describes the morphological distribution observed in the submitted image only.",
        "> It does NOT establish a disease diagnosis, differential, or clinical recommendation.",
        "> Automated image analysis is currently insufficient for a reliable population-level morphological interpretation",
        "> without validation of cell-detection quality, model calibration, and expert haematopathologist review.",
    ]

    if low_ev:
        lines.append(
            f"\n>  **Low-evidence caution:** Only {total_cells} cells were detected. "
            "Population-level statistics carry high sampling uncertainty."
        )

    if total_cells < 50:
        lines += [
            "",
            "#### 2. Visual Evidence & Sampling Limitation",
            f"The submitted image(s) yielded **{total_cells} automatically detected and classified regions**.",
            "- **Population Assessment:** Automated image analysis is currently insufficient for a reliable",
            "  population-level morphological interpretation with this sample size.",
            "- **Cell-Level Findings:** Individual region morphology classifications and XAI attribution maps are presented for examination.",
            "- **Recommendation:** High-power aspirate examination across multiple fields and standard 500-cell manual differential.",
        ]
    else:
        lines += ["", "#### 2. Image Morphology Context (NOT a diagnostic differential)"]

        if primary_dx == "ALL":
            haem_cite = f"[WHO2022_HAEMATOGONES_009]" if "WHO2022_HAEMATOGONES_009" in retrieved_cids else f"({cited_str})"
            lines.append(
                f"The image morphology is consistent with features described for "
                f"**Acute Lymphoblastic Leukemia (ALL)** ({cited_str}). "
                "Small-to-medium lymphoblasts with high nuclear-to-cytoplasmic ratios and condensed chromatin characterise this presentation."
            )
            lines.append(f"- **WHO Blast Criteria:** {blast_eval['rules_matched'][0]}")
            if pediatric is True:
                lines.append(
                    f"- **Age-specific consideration:** Patient age confirmed < 18 years. "
                    f"In pediatric presentations, benign haematogones {haem_cite} must be excluded by "
                    "immunophenotypic demonstration of a monomorphic maturational arrest."
                )
            elif pediatric is None:
                lines.append(
                    "- **Age context:** Patient age was not provided. Age-specific differential considerations "
                    "(e.g., haematogone burden in young children, adult ALL variants) cannot be applied."
                )
            # adult: no pediatric language

        elif primary_dx == "AML":
            apl_cite = f"[WHO2022_AML_SUB_005]" if "WHO2022_AML_SUB_005" in retrieved_cids else f"({cited_str})"
            lines.append(
                f"The image morphology is consistent with features described for "
                f"**Acute Myeloid Leukemia (AML)** ({cited_str}). "
                "Myeloblasts characteristically exhibit delicate chromatin, distinct nucleoli, and variable granulation."
            )
            lines.append(f"- **WHO Blast Criteria:** {blast_eval['rules_matched'][0]}")
            lines.append(
                f"- **Urgent note:** If hypergranular promyelocytes or multiple Auer rods are visualised, "
                f"emergency evaluation for APL {apl_cite} with immediate ATRA protocol should be considered."
            )

        elif primary_dx == "CML":
            cml_cite   = f"[WHO2022_CML_CP_007]"  if "WHO2022_CML_CP_007"   in retrieved_cids else f"({cited_str})"
            react_cite = f"[WHO2022_REACTIVE_010]" if "WHO2022_REACTIVE_010" in retrieved_cids else f"({cited_str})"
            lines.append(
                f"The cellular distribution is consistent with features of "
                f"**Chronic Myeloid Leukemia (CML)** ({cited_str}). "
                "Granulocytic hyperplasia across all maturation stages is the predominant finding."
            )
            lines.append(f"- **Cytological hallmarks:** Myelocyte bulge pattern with blast proportion < 10% {cml_cite}.")
            lines.append(
                f"- **Differential consideration:** Reactive leukemoid reactions {react_cite} "
                "require exclusion by BCR::ABL1 genetic testing."
            )

        else:
            mpal_cite  = f"[WHO2022_MPAL_011]"    if "WHO2022_MPAL_011"    in retrieved_cids else f"({cited_str})"
            react_cite = f"[WHO2022_REACTIVE_010]" if "WHO2022_REACTIVE_010" in retrieved_cids else f"({cited_str})"
            lines.append(
                f"The cytological image distribution falls within **Non-malignant / Reactive / Other morphology** ({cited_str}). "
                "Detected blast proportions are low or demonstrate ambiguous lineage features."
            )
            lines.append(
                f"- Evaluation for Mixed Phenotype Acute Leukemia (MPAL) {mpal_cite} "
                f"or reactive marrow expansion {react_cite} may be appropriate; confirmation requires ancillary testing."
            )

    # Clinical flags
    if blast_eval["clinical_flags"]:
        lines.append("\n#### 3. Clinical Caveats")
        for flag in blast_eval["clinical_flags"]:
            lines.append(f"-  {flag}")

    # What cannot be determined
    lines += [
        "",
        "#### 4. Cannot Be Determined From This Image",
        "The following were **not provided** and have NOT been inferred or fabricated:",
        "- Complete Blood Count (CBC) — **Not provided**",
        "- Flow Cytometric Immunophenotyping — **Not provided**",
        "- Cytogenetics / Karyotype — **Not provided**",
        "- Molecular / NGS data — **Not provided**",
        "",
        "#### 5. Recommended Confirmatory Workup",
        "1. **Flow Cytometric Immunophenotyping** — definitive lineage assignment (CD19, CD10, CD3, MPO, etc.).",
        "2. **Cytogenetics & Molecular Genetics** — karyotyping, FISH, RT-PCR for key fusions.",
        "3. **Complete Blood Count (CBC)** — peripheral blast count, anemia, thrombocytopenia staging.",
        "4. **Haematopathologist Review** — morphological review of the full aspirate slide.",
    ]

    draft_text = "\n".join(lines)
    return {
        "draft_differential": draft_text,
        "differential_diagnosis_hierarchy": [primary_dx, secondary_dx],
        "morphological_justification": (
            f"Image-derived blast fraction {blast_fraction:.1%} with predominant model classification {primary_dx}."
        ),
    }


def synthesizer_node(state: AgentState) -> AgentState:
    logger.info(f"[{state.get('patient_id')}] Synthesizer drafting assessment...")
    synthesis = build_deterministic_synthesis(state)
    state["draft_differential"] = synthesis["draft_differential"]
    state["differential_diagnosis_hierarchy"] = synthesis["differential_diagnosis_hierarchy"]
    state["morphological_justification"] = synthesis["morphological_justification"]
    return state
