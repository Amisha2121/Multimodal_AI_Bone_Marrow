"""
agents/graph.py
─────────────────────────────────────────────────────────────────────────────
LangGraph Workflow: Connects Retriever, Synthesizer, and Critic Agents into an
iterative, self-reflecting, evidence-grounded clinical decision-support loop.

Workflow:
  START -> [Retriever Node] -> [Synthesizer Node] -> [Critic Node]
                                    ^                      |
                                    |--- (Refine if rejected)
                                    |
                                    v (If approved / max revisions)
                                   END
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Literal

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from langgraph.graph import StateGraph, START, END

from agents.state import AgentState, PatientMorphologyInput
from agents.retriever import retriever_node
from agents.synthesizer import synthesizer_node
from agents.critic import critic_node

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("agent_graph")


def critic_router(state: AgentState) -> Literal["synthesizer", "__end__"]:
    """
    Decides whether to finalize the report or route back to the Synthesizer for revision.
    """
    is_approved = state.get("is_approved", False)
    revisions = state.get("revision_count", 0)

    if is_approved or revisions >= 2:
        return "__end__"
    else:
        logger.info(f"Routing back to Synthesizer for revision #{revisions + 1}...")
        return "synthesizer"


def build_clinical_agent_graph():
    """Compiles the LangGraph clinical multi-agent workflow."""
    builder = StateGraph(AgentState)

    # Register Nodes
    builder.add_node("retriever", retriever_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("critic", critic_node)

    # Define Linear Flow
    builder.add_edge(START, "retriever")
    builder.add_edge("retriever", "synthesizer")
    builder.add_edge("synthesizer", "critic")

    # Conditional Self-Reflecting Edge
    builder.add_conditional_edges(
        "critic",
        critic_router,
        {
            "synthesizer": "synthesizer",
            "__end__": END,
        }
    )

    return builder.compile()


def run_agentic_assessment(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience API to execute the multi-agent assessment for a single patient profile.
    """
    graph = build_clinical_agent_graph()

    # Initial State
    initial_state: AgentState = {
        "patient_id": patient_data.get("patient_id", "UNKNOWN"),
        "age_months": patient_data.get("age_months"),
        "age_years": patient_data.get("age_years"),
        "sex": patient_data.get("sex"),
        "annotated_blast_fraction": patient_data.get("annotated_blast_fraction", 0.0),
        "predicted_class": patient_data.get("predicted_class", "Other"),
        "class_probabilities": patient_data.get("class_probabilities", {}),
        "cell_counts_by_category": patient_data.get("cell_counts_by_category", {}),
        "total_cells": patient_data.get("total_cells", 0),
        "total_images": patient_data.get("total_images", 1),
        "low_evidence_flag": patient_data.get("low_evidence_flag", False),
        "revision_count": 0,
        "is_approved": False,
    }

    final_state = graph.invoke(initial_state)
    return dict(final_state)


# ── Quick Verification Demo CLI ──────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  AGENTIC MULTIMODAL RAG CLINICAL ASSESSMENT DEMO")
    print("=" * 70)

    # Sample mock patient based on real cohort statistics
    sample_patient = {
        "patient_id": "02522A2B2726FB0A03BB19F2D8D9524D",
        "age_months": 72.0,  # 6.0 years
        "annotated_blast_fraction": 0.684,
        "predicted_class": "ALL",
        "class_probabilities": {"ALL": 0.92, "AML": 0.05, "CML": 0.02, "Other": 0.01},
        "cell_counts_by_category": {
            "Blast": 71,
            "Lymphoid Series": 15,
            "Granulocyte (Mature/Band)": 10,
            "Erythroid Precursor": 8,
        },
        "total_cells": 104,
        "low_evidence_flag": False,
    }

    print(f"\nEvaluating Patient: {sample_patient['patient_id']} (Age: {sample_patient['age_months']/12:.1f} yrs)")
    print(f"Model Consensus: {sample_patient['predicted_class']} (Blast Fraction: {sample_patient['annotated_blast_fraction']:.1%})")

    result = run_agentic_assessment(sample_patient)

    print("\n" + "-" * 70)
    print("FINAL AGENTIC CLINICAL DECISION SUPPORT REPORT:")
    print("-" * 70)
    print(result.get("final_report", "No report generated."))
    print("-" * 70)
    print(f"Critic Grounding Score: {result.get('critic_grounding_score')}")
    print(f"Total Revisions: {result.get('revision_count')}")
    print(f"Requires Human Abstention: {result.get('requires_human_abstention')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
