# -*- coding: utf-8 -*-
"""
Tests for Phase 10 Step 8 Schema, scan_facts, and Chapters Fallback.
"""

import unittest
from pipeline.scan_facts import build_scan_facts, generate_deterministic_chapters
from pipeline.summary import build_morphology_summary
from app.db import DB_PATH
import sqlite3
import json


class TestPhase10SchemaAndChapters(unittest.TestCase):

    def setUp(self):
        self.cid = "CASE-F09AFAC4"
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT cell_results_json FROM cases WHERE case_id=?", (self.cid,))
            row = cur.fetchone()
            self.cells = json.loads(row[0]) if row and row[0] else []
        self.summary = build_morphology_summary(self.cid)

    def test_scan_facts_fields(self):
        facts = build_scan_facts(self.summary, self.cells)
        required_keys = [
            "case_id", "total_cells", "blast_count", "blast_pct", "blast_ci",
            "sensitivity_range", "plausibility_flags", "confusion_neighbours",
            "mean_confidence", "confidence_bands"
        ]
        for k in required_keys:
            self.assertIn(k, facts)

    def test_eight_chapters_generated_with_claims(self):
        facts = build_scan_facts(self.summary, self.cells)
        chapters = generate_deterministic_chapters(facts, ["WHO2022_BALL_001", "WHO2022_MDS_EB2_014"])
        self.assertEqual(len(chapters), 8)
        expected_ids = [f"ch{i}" for i in range(1, 9)]
        actual_ids = [ch["id"] for ch in chapters]
        self.assertEqual(actual_ids, expected_ids)

        for ch in chapters:
            self.assertGreater(len(ch["claims"]), 0, f"Chapter {ch['id']} must have claims")
            for c in ch["claims"]:
                self.assertIn("id", c)
                self.assertIn("type", c)
                self.assertIn("text", c)
                self.assertTrue(c.get("is_supported"))
                # Language guard: must not assert diagnosis
                forbidden = ["diagnosed with", "suffers from", "positive for acute leukemia"]
                for f in forbidden:
                    self.assertNotIn(f, c["text"].lower())


if __name__ == "__main__":
    unittest.main()

