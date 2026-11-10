# -*- coding: utf-8 -*-
"""
Phase 7 Verification Tests: Triage Agent & Foundation.
Tests:
  1. Triage queue on CASE-CECDBCC7 lists the 6 cells under 50% confidence.
  2. Relabelling one updates counts only when 'Use reviewer labels' is ON.
  3. AI labels in storage remain unchanged.
  4. Language guard neutralizes definitive diagnostic assertions.
  5. Agent events are logged and tool calls per request capped at <= 6.
  6. Preserved cases integrity: CASE-F09AFAC4 (49 cells, 10.2%) and CASE-CECDBCC7 (12 cells, 8.3%).
"""

import sys
import unittest
import requests
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.triage_agent import build_triage_queue
from pipeline.summary import build_morphology_summary
from pipeline.agent_guard import sanitize_agent_language
from app.db import get_case_from_db, update_cell_reviewer_label_in_db, get_agent_events_from_db


class TestPhase7TriageAgent(unittest.TestCase):
    def test_triage_queue_case_cecdbcc7(self):
        result = build_triage_queue("CASE-CECDBCC7")
        self.assertEqual(result["case_id"], "CASE-CECDBCC7")
        self.assertEqual(result["total_cells"], 12)
        queue = result["queue"]
        # Exactly 6 cells are under 50% confidence in CASE-CECDBCC7
        self.assertEqual(len(queue), 6)

        # Blast cell IMAGE-01_CELL-010 (conf 0.42) ranked first
        first_cell = queue[0]
        self.assertEqual(first_cell["cell_id"], "IMAGE-01_CELL-010")
        self.assertEqual(first_cell["ai_class"], "Blast")
        self.assertIn("blast", first_cell["reason"].lower())

        # Check all queued cells have confidence < 0.50
        for c in queue:
            self.assertLess(c["confidence"], 0.50)

    def test_reviewer_label_isolation_and_toggle(self):
        # 1. Clean slate check: AI counts
        summary_ai = build_morphology_summary("CASE-CECDBCC7", use_reviewer_labels=False)
        self.assertEqual(summary_ai["total_cells"], 12)
        self.assertEqual(summary_ai["blast_count"], 1)
        self.assertAlmostEqual(summary_ai["blast_fraction"], 0.0833, places=3)

        # 2. Relabel cell IMAGE-01_CELL-007 from 'Myeloid (Maturing)' to 'Blast'
        cell_id = "IMAGE-01_CELL-007"
        update_cell_reviewer_label_in_db("CASE-CECDBCC7", cell_id, "Blast")

        # 3. Check with use_reviewer_labels=False (MUST REMAIN 8.3% blasts)
        summary_off = build_morphology_summary("CASE-CECDBCC7", use_reviewer_labels=False)
        self.assertEqual(summary_off["blast_count"], 1)
        self.assertAlmostEqual(summary_off["blast_fraction"], 0.0833, places=3)

        # 4. Check with use_reviewer_labels=True (MUST BE 2 blasts = 16.7%)
        summary_on = build_morphology_summary("CASE-CECDBCC7", use_reviewer_labels=True)
        self.assertEqual(summary_on["blast_count"], 2)
        self.assertAlmostEqual(summary_on["blast_fraction"], 0.1667, places=3)

        # 5. Verify stored AI label is untouched
        db_case = get_case_from_db("CASE-CECDBCC7")
        stored_cell = next(c for c in db_case["cell_images"] if c["id"] == cell_id)
        self.assertEqual(stored_cell["class"], "Myeloid (Maturing)")
        self.assertEqual(stored_cell.get("reviewer_label"), "Blast")

        # 6. Reset reviewer_label to keep clean persistent state
        update_cell_reviewer_label_in_db("CASE-CECDBCC7", cell_id, None)
        summary_reset = build_morphology_summary("CASE-CECDBCC7", use_reviewer_labels=True)
        self.assertEqual(summary_reset["blast_count"], 1)
        self.assertAlmostEqual(summary_reset["blast_fraction"], 0.0833, places=3)

    def test_language_guard(self):
        # Diagnostic claim must be neutralized
        dirty_text = "The patient has AML based on the increased blast count."
        clean_text = sanitize_agent_language(dirty_text)
        self.assertNotIn("patient has AML", clean_text)
        self.assertIn("Clinical note:", clean_text)
        self.assertIn("physician evaluation", clean_text)

        # Descriptive cytomorphology must remain unaffected
        pure_text = "5 cells classified as Blast with high nucleocytoplasmic ratio."
        self.assertEqual(sanitize_agent_language(pure_text), pure_text)

    def test_agent_events_logged(self):
        # Run triage queue and check logged events
        result = build_triage_queue("CASE-CECDBCC7")
        self.assertLessEqual(len(result["history"]), 6)
        events = get_agent_events_from_db(case_id="CASE-CECDBCC7", limit=10)
        self.assertGreater(len(events), 0)
        triage_events = [e for e in events if e.get("agent") == "Triage agent"]
        self.assertGreater(len(triage_events), 0)

    def test_preserved_saved_cases_integrity(self):
        c1 = build_morphology_summary("CASE-F09AFAC4", use_reviewer_labels=False)
        self.assertEqual(c1["total_cells"], 49)
        self.assertEqual(c1["blast_count"], 5)
        self.assertAlmostEqual(c1["blast_fraction"], 0.102, places=2)

        c2 = build_morphology_summary("CASE-CECDBCC7", use_reviewer_labels=False)
        self.assertEqual(c2["total_cells"], 12)
        self.assertEqual(c2["blast_count"], 1)
        self.assertAlmostEqual(c2["blast_fraction"], 0.083, places=2)


if __name__ == "__main__":
    unittest.main()
