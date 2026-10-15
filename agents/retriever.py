"""
agents/retriever.py
─────────────────────────────────────────────────────────────────────────────
Retriever Agent Node: Translates patient morphological profiles into targeted
clinical queries and retrieves evidence from WHO 5th Ed / ICC guidelines.
"""

import logging
from typing import Dict, Any

from agents.state import AgentState
from agents.tools import get_retriever, retrieve_guidelines
from agents.fallback_guidelines import get_fallback_evidence, format_fallback_notice

logger = logging.getLogger("agent_retriever")


def retriever_node(state: AgentState) -> AgentState:
    """
    Analyzes patient quantitative morphology metrics and retrieves authoritative clinical literature.
    """
    patient_id = state.get("patient_id", "UNKNOWN")
    predicted_class = state.get("predicted_class", "Leukemia")
    blast_fraction = state.get("annotated_blast_fraction", 0.0)
    age_months = state.get("age_months", None)
    cell_counts = state.get("cell_counts_by_category", {})

    # Sort predominant cell types
    sorted_cells = sorted(cell_counts.items(), key=lambda x: x[1], reverse=True)
    top_cell_types = [c[0] for c in sorted_cells[:3]]

    # Formulate clinical query
    retriever = get_retriever()
    query = retriever.build_query_from_morphology(
        predicted_class=predicted_class,
        blast_fraction=blast_fraction,
        predominant_cells=top_cell_types,
        patient_age_months=age_months,
    )

    logger.info(f"[{patient_id}] Retriever Query: '{query}'")
    retrieved_chunks = retrieve_guidelines(query=query, top_k=4)

    # FALLBACK: If RAG returns no results, use built-in guidelines
    if not retrieved_chunks or len(retrieved_chunks) == 0:
        logger.warning(f"[{patient_id}] RAG retrieval failed, using fallback guidelines")
        age_years = (age_months / 12.0) if age_months else None
        predominant_morph = top_cell_types[0] if top_cell_types else None
        total_cells = state.get("total_cells", 0)
        
        retrieved_chunks = get_fallback_evidence(
            blast_fraction=blast_fraction,
            age_years=age_years,
            predominant_morphology=predominant_morph,
            total_cells=total_cells,
        )
        
        # Mark that fallback was used
        state["used_fallback_guidelines"] = True
    else:
        state["used_fallback_guidelines"] = False

    formatted_context = retriever.format_context_for_llm(retrieved_chunks)

    state["search_query"] = query
    state["retrieved_evidence"] = retrieved_chunks
    state["formatted_guidelines_context"] = formatted_context

    return state
