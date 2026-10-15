"""
agents/tools.py
─────────────────────────────────────────────────────────────────────────────
Deterministic Clinical Pathology Tools for Agentic Reasoning.
"""

import json
import logging
from typing import Dict, List, Optional, Any

from retrieval.retrieve import HybridRetriever

logger = logging.getLogger("agent_tools")

_GLOBAL_RETRIEVER: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    """Singleton getter for HybridRetriever to avoid reloading FAISS/BGE repeatedly."""
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = HybridRetriever()
    return _GLOBAL_RETRIEVER


def retrieve_guidelines(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """Retrieve top-K clinical passages using hybrid dense/sparse search."""
    retriever = get_retriever()
    return retriever.retrieve(query=query, top_k=top_k)


def evaluate_blast_cutoffs(
    blast_fraction: float,
    predicted_class: str,
) -> Dict[str, Any]:
    """
    Evaluates observed annotated blast fraction against standard WHO/ICC diagnostic rules.
    """
    pct = blast_fraction * 100.0

    rules_matched = []
    flags = []

    if pct >= 20.0:
        rules_matched.append("Meets standard >=20% blast threshold for Acute Leukemia (ALL / AML).")
    else:
        rules_matched.append(f"Annotated blast fraction is {pct:.1f}% (<20%).")
        if predicted_class in ["ALL", "AML"]:
            flags.append(
                "Sub-threshold blast fraction (<20%) for acute leukemia. "
                "Requires evaluation for defining genetic abnormalities (e.g. APL PML::RARA, CBF-AML) "
                "or sampling limitation of selected monolayer fields."
            )
        elif predicted_class == "CML":
            rules_matched.append("Blast percentage <10% is consistent with Chronic Phase CML (CML-CP).")

    return {
        "observed_blast_pct": round(pct, 2),
        "rules_matched": rules_matched,
        "clinical_flags": flags,
    }


def format_scientific_disclaimers(
    low_evidence: bool = False,
    total_cells: int = 0,
    evidence_sufficiency: str = "UNKNOWN",
) -> str:
    """Generates mandatory regulatory and scientific disclaimers."""
    disclaimers = [
        "> [!WARNING]",
        "> **Image-Derived Blast Proportion ≠ Clinical Whole-Slide Differential:**  ",
        "> This proportion is derived from automatically detected cells in the submitted image(s).  ",
        "> It does NOT replace an exhaustive 500-cell haematopathologist aspirate differential count.",
        "",
        "> [!IMPORTANT]",
        "> **Ancillary Diagnostic Modalities Not Provided:**  ",
        "> Complete Blood Count (CBC), flow cytometric immunophenotyping, cytogenetics/karyotype, and  ",
        "> molecular/NGS data were not submitted. Final definitive disease subtyping requires these modalities.",
        "",
        "> [!NOTE]",
        "> **Model Training Context:**  ",
        "> The vision model was developed using a pediatric bone-marrow research cohort.  ",
        "> It has not been validated for adult patients or general clinical use.",
    ]

    if evidence_sufficiency in ("LOW", "INSUFFICIENT") or low_evidence:
        disclaimers.insert(
            0,
            f"> [!CAUTION]\n"
            f"> **LOW-EVIDENCE WARNING:** Only {total_cells} cell(s) were detected ({evidence_sufficiency}).  \n"
            f"> Morphological distributions carry high statistical uncertainty and should not be used for clinical decisions.\n"
        )

    return "\n".join(disclaimers)
