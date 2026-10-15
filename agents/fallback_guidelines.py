"""
agents/fallback_guidelines.py
─────────────────────────────────────────────────────────────────────────────
Built-in clinical guidelines as emergency fallback when RAG retrieval fails.

Provides basic WHO diagnostic criteria and pediatric considerations.
"""

from typing import Dict, List, Any, Optional


# WHO 5th Edition Blast Thresholds
BLAST_THRESHOLDS = {
    "ALL": {
        "threshold": 0.20,
        "description": "≥20% lymphoblasts in bone marrow required for ALL diagnosis",
        "subtypes": ["B-ALL", "T-ALL"],
        "pediatric_note": "Most common pediatric malignancy (peak 2-5 years)",
    },
    "AML": {
        "threshold": 0.20,
        "description": "≥20% myeloblasts required for AML diagnosis (WHO 5th)",
        "exceptions": [
            "APL with t(15;17) - diagnosis regardless of blast count",
            "AML with defining genetic abnormalities - may have <20% blasts",
        ],
    },
    "CML": {
        "chronic_phase": "Blasts <10% with BCR::ABL1 fusion",
        "accelerated_phase": "Blasts 10-19%",
        "blast_phase": "Blasts ≥20%",
    },
}


PEDIATRIC_CONSIDERATIONS = """
**Pediatric Acute Leukemia Epidemiology:**
- ALL accounts for 75-80% of pediatric acute leukemias
- Peak incidence: 2-5 years of age
- B-cell ALL is most common subtype (80-85% of pediatric ALL)
- T-cell ALL accounts for 15-20% of pediatric ALL
- AML represents 15-20% of pediatric acute leukemias

**Age-Specific Risk Factors:**
- Infant leukemia (<1 year): Associated with KMT2A (MLL) rearrangements
- Age 1-10 years: Best prognosis group for B-ALL
- Adolescents/young adults: Intermediate prognosis

**Critical Diagnostic Requirements:**
- Flow cytometry immunophenotyping (mandatory)
- Cytogenetics/FISH for risk stratification
- Molecular testing (BCR-ABL1, KMT2A rearrangements, ETV6-RUNX1)
- CNS assessment (lumbar puncture)
"""


ADULT_CONSIDERATIONS = """
**Adult Acute Leukemia Patterns:**
- AML more common than ALL in adults
- ALL prognosis generally worse in adults vs children
- Ph+ ALL (BCR-ABL1) more common in adults (25-30% vs 3-5% in children)
- Secondary AML from prior chemotherapy or MDS increases with age

**Key Diagnostic Considerations:**
- Prior hematologic disorder (MDS, MPN)
- Prior chemotherapy/radiation exposure
- Cytogenetic risk stratification critical for treatment decisions
- Consider molecular testing (FLT3, NPM1, CEBPA in AML)
"""


MORPHOLOGY_PATTERNS = {
    "Blast": {
        "significance": "Immature precursor cells",
        "high_percentage_concern": "≥20% suggests acute leukemia",
        "differential": ["ALL", "AML", "Blast phase CML", "High-grade MDS"],
    },
    "Lymphoid (Mature)": {
        "significance": "Mature lymphocytes",
        "elevated_consider": "Reactive lymphocytosis, CLL, lymphoproliferative disorders",
    },
    "Myeloid (Maturing)": {
        "significance": "Maturing granulocyte precursors",
        "left_shift": "Increased immature forms suggests stress response or CML",
    },
    "Megakaryocyte / Platelet": {
        "significance": "Platelet precursors and platelets",
        "elevated_consider": "Essential thrombocythemia, reactive thrombocytosis",
        "decreased_consider": "ITP, aplastic anemia, chemotherapy effect",
    },
}


def get_fallback_evidence(
    blast_fraction: float,
    age_years: Optional[float] = None,
    predominant_morphology: Optional[str] = None,
    total_cells: int = 0,
) -> List[Dict[str, Any]]:
    """
    Returns basic clinical guidelines when RAG retrieval fails.
    
    Args:
        blast_fraction: Proportion of blast cells [0.0-1.0]
        age_years: Patient age in years (None if unknown)
        predominant_morphology: Most common cell type observed
        total_cells: Total cells analyzed
    
    Returns:
        List of evidence chunks mimicking RAG format
    """
    evidence = []
    
    # Evidence 1: Blast threshold assessment
    blast_pct = blast_fraction * 100.0
    
    if blast_fraction >= 0.20:
        evidence.append({
            "chunk_id": "FALLBACK_BLAST_THRESHOLD",
            "title": "WHO Blast Percentage Diagnostic Criteria",
            "source": "WHO Classification of Haematolymphoid Tumours (5th Edition, 2022)",
            "text": (
                f"**Observed Finding:** {blast_pct:.1f}% blast cells detected.\n\n"
                f"**WHO Diagnostic Threshold:** ≥20% blasts in bone marrow is required for "
                f"acute leukemia diagnosis (ALL or AML).\n\n"
                f"**Clinical Significance:** The observed blast percentage ({blast_pct:.1f}%) "
                f"EXCEEDS the diagnostic threshold. This finding is consistent with acute leukemia, "
                f"pending confirmatory immunophenotyping and molecular studies.\n\n"
                f"**Important Exceptions:**\n"
                f"- Acute promyelocytic leukemia (APL) with t(15;17): Diagnosis established regardless of blast count\n"
                f"- AML with recurrent genetic abnormalities: May be diagnosed with <20% blasts\n"
                f"- High-grade myelodysplastic syndrome (MDS-EB-2): 10-19% blasts"
            ),
            "blast_threshold_rule": {
                "threshold": 0.20,
                "observed": blast_fraction,
                "threshold_met": True,
                "interpretation": "Exceeds WHO criteria for acute leukemia"
            },
            "hybrid_rrf_score": 1.0,
        })
    elif 0.10 <= blast_fraction < 0.20:
        evidence.append({
            "chunk_id": "FALLBACK_INTERMEDIATE_BLASTS",
            "title": "Intermediate Blast Percentage Interpretation",
            "source": "WHO Classification Guidelines",
            "text": (
                f"**Observed Finding:** {blast_pct:.1f}% blast cells detected.\n\n"
                f"**Interpretation:** Blast percentage is between 10-19%, which is BELOW the "
                f"acute leukemia threshold (≥20%) but above normal.\n\n"
                f"**Differential Diagnosis:**\n"
                f"- High-grade Myelodysplastic Syndrome (MDS-EB-2): 10-19% blasts\n"
                f"- Chronic Myeloid Leukemia - Accelerated Phase: 10-19% blasts\n"
                f"- Recovering bone marrow (post-chemotherapy)\n"
                f"- Reactive conditions with left shift\n\n"
                f"**Recommended:** Cytogenetic and molecular studies to distinguish between entities."
            ),
            "blast_threshold_rule": {
                "threshold": 0.20,
                "observed": blast_fraction,
                "threshold_met": False,
                "interpretation": "Below acute leukemia threshold, consider MDS or CML"
            },
            "hybrid_rrf_score": 0.95,
        })
    else:
        evidence.append({
            "chunk_id": "FALLBACK_LOW_BLASTS",
            "title": "Low Blast Percentage - Non-Acute Pattern",
            "source": "Diagnostic Hematopathology Guidelines",
            "text": (
                f"**Observed Finding:** {blast_pct:.1f}% blast cells detected.\n\n"
                f"**Interpretation:** Blast percentage is <10%, which is BELOW the threshold "
                f"for both acute leukemia (≥20%) and high-grade myelodysplasia (≥10%).\n\n"
                f"**Possible Interpretations:**\n"
                f"- Normal or reactive bone marrow\n"
                f"- Chronic myeloid disorders (CML chronic phase, MPN)\n"
                f"- Low-grade MDS\n"
                f"- Sampling artifact or field selection bias\n\n"
                f"**Note:** This analysis is based on detected cells in submitted images only. "
                f"A formal 500-cell bone marrow differential count may yield different results."
            ),
            "blast_threshold_rule": {
                "threshold": 0.20,
                "observed": blast_fraction,
                "threshold_met": False,
                "interpretation": "Does not meet acute leukemia criteria"
            },
            "hybrid_rrf_score": 0.90,
        })
    
    # Evidence 2: Age-specific considerations
    if age_years is not None:
        if age_years < 18:
            evidence.append({
                "chunk_id": "FALLBACK_PEDIATRIC",
                "title": "Pediatric Acute Leukemia Considerations",
                "source": "Pediatric Hematology-Oncology Guidelines",
                "text": PEDIATRIC_CONSIDERATIONS,
                "morphological_keywords": ["pediatric", "ALL", "childhood leukemia"],
                "hybrid_rrf_score": 0.85,
            })
        else:
            evidence.append({
                "chunk_id": "FALLBACK_ADULT",
                "title": "Adult Acute Leukemia Patterns",
                "source": "Adult Hematology Clinical Guidelines",
                "text": ADULT_CONSIDERATIONS,
                "morphological_keywords": ["adult", "AML", "secondary leukemia"],
                "hybrid_rrf_score": 0.85,
            })
    
    # Evidence 3: Required diagnostic workup
    evidence.append({
        "chunk_id": "FALLBACK_WORKUP",
        "title": "Essential Diagnostic Testing for Acute Leukemia",
        "source": "NCCN Clinical Practice Guidelines",
        "text": (
            "**Mandatory Diagnostic Studies:**\n\n"
            "1. **Flow Cytometry Immunophenotyping:**\n"
            "   - Distinguishes ALL from AML (essential)\n"
            "   - Identifies lineage (B-cell, T-cell, myeloid)\n"
            "   - Detects aberrant antigen expression\n\n"
            "2. **Cytogenetics/Karyotype:**\n"
            "   - Risk stratification\n"
            "   - Treatment selection\n"
            "   - Prognostic information\n\n"
            "3. **Molecular Studies:**\n"
            "   - BCR-ABL1 (Ph+ leukemia)\n"
            "   - KMT2A rearrangements (infant/pediatric)\n"
            "   - FLT3, NPM1, CEBPA (AML)\n"
            "   - Minimal residual disease markers\n\n"
            "4. **Complete Blood Count:**\n"
            "   - Peripheral blast count\n"
            "   - Hemoglobin/platelet levels\n"
            "   - White blood cell count\n\n"
            "5. **Clinical Chemistry:**\n"
            "   - LDH, uric acid (tumor burden)\n"
            "   - Renal/hepatic function\n\n"
            "**Note:** Morphology alone is INSUFFICIENT for definitive subtype diagnosis."
        ),
        "morphological_keywords": ["diagnostic workup", "flow cytometry", "molecular testing"],
        "hybrid_rrf_score": 0.80,
    })
    
    # Evidence 4: Morphology-specific guidance
    if predominant_morphology and predominant_morphology in MORPHOLOGY_PATTERNS:
        pattern_info = MORPHOLOGY_PATTERNS[predominant_morphology]
        evidence.append({
            "chunk_id": f"FALLBACK_MORPH_{predominant_morphology.upper().replace(' ', '_')}",
            "title": f"Clinical Significance of {predominant_morphology}",
            "source": "Bone Marrow Interpretation Guidelines",
            "text": (
                f"**Predominant Cell Type:** {predominant_morphology}\n\n"
                f"**Significance:** {pattern_info['significance']}\n\n"
                f"**Differential Diagnosis:** {', '.join(pattern_info.get('differential', ['Requires clinical correlation']))}"
            ),
            "morphological_keywords": [predominant_morphology.lower()],
            "hybrid_rrf_score": 0.75,
        })
    
    # Evidence 5: Sample adequacy warning
    if total_cells < 50:
        evidence.append({
            "chunk_id": "FALLBACK_LOW_SAMPLE",
            "title": "Sample Adequacy Considerations",
            "source": "Laboratory Quality Standards",
            "text": (
                f"**Sample Size:** {total_cells} cells analyzed\n\n"
                f"**Adequacy Concern:** A standard bone marrow differential requires counting "
                f"≥500 cells for statistical reliability. The current sample of {total_cells} cells "
                f"has HIGH SAMPLING VARIABILITY.\n\n"
                f"**Clinical Impact:**\n"
                f"- Small samples may not represent overall marrow composition\n"
                f"- Focal lesions or sampling artifacts can skew results\n"
                f"- Percentage estimates have wide confidence intervals\n\n"
                f"**Recommendation:** Results should be interpreted with caution. Consider "
                f"correlation with peripheral blood findings and additional marrow sampling if discordant."
            ),
            "hybrid_rrf_score": 0.70,
        })
    
    return evidence


def format_fallback_notice() -> str:
    """Returns a notice that fallback guidelines are being used."""
    return (
        "\n\n---\n\n"
        "> ⚠️ **Note:** Knowledge base retrieval did not return results for this query. "
        "The assessment above is based on built-in WHO classification criteria and general "
        "hematopathology guidelines. For the most current disease-specific recommendations, "
        "consult primary literature and institutional protocols.\n"
    )
