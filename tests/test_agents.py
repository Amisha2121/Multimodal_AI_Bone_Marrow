"""
tests/test_agents.py
─────────────────────────────────────────────────────────────────────────────
Unit and Integration Tests for the Multi-Agent Clinical Assessment Workflow.
"""

import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from context.assembler import load_patient_multimodal_context
from agents.graph import run_agentic_assessment, build_clinical_agent_graph

# A real consensus patient ID present in data/patient_manifest.csv
_TEST_PATIENT_ID = "02522A2B2726FB0A03BB19F2D8D9524D"


class TestAgentWorkflow(unittest.TestCase):

    def test_context_assembler(self):
        """Context assembler returns real patient data with correct schema."""
        ctx = load_patient_multimodal_context(_TEST_PATIENT_ID)
        # Top-level schema
        self.assertEqual(ctx["patient_id"], _TEST_PATIENT_ID)
        self.assertIn("demographics", ctx)
        self.assertIn("morphology", ctx)
        self.assertIn("vision_predictions", ctx)
        self.assertIn("missing_modalities", ctx)
        self.assertIn("regulatory_safety", ctx)
        # Morphology sub-fields
        morph = ctx["morphology"]
        self.assertGreater(morph["total_annotated_cells"], 0)
        self.assertGreaterEqual(morph["annotated_blast_fraction"], 0.0)
        # Missing modalities must be null (never fabricated)
        self.assertIsNone(ctx["missing_modalities"]["complete_blood_count"]["data"])
        self.assertIsNone(ctx["missing_modalities"]["cytogenetics"]["data"])

    def test_agent_graph_compilation(self):
        """LangGraph agent graph compiles without errors."""
        graph = build_clinical_agent_graph()
        self.assertIsNotNone(graph)

    def test_end_to_end_agentic_assessment(self):
        """Full agentic assessment returns approved report with grounding score >= 0.75."""
        ctx = load_patient_multimodal_context(_TEST_PATIENT_ID)
        morph = ctx["morphology"]
        # Build minimal agent_input dict from assembled context
        agent_input = {
            "patient_id": ctx["patient_id"],
            "age_months": ctx["demographics"]["age_months"],
            "annotated_blast_fraction": morph["annotated_blast_fraction"],
            "predicted_class": ctx["demographics"]["ground_truth"]["primary_4class"],
            "class_probabilities": {},
            "cell_counts_by_category": morph["category_counts"],
            "total_cells": morph["total_annotated_cells"],
            "low_evidence_flag": morph["low_evidence_flag"],
        }
        result = run_agentic_assessment(agent_input)

        self.assertIn("final_report", result)
        self.assertIn("critic_grounding_score", result)
        self.assertGreaterEqual(result["critic_grounding_score"], 0.75)
        self.assertTrue(result["is_approved"])
        self.assertFalse(result["requires_human_abstention"])
        self.assertIn(
            "Annotated-Cell Blast Fraction \u2260 Whole-Slide Clinical Aspirate Differential",
            result["final_report"],
        )


if __name__ == "__main__":
    unittest.main()
