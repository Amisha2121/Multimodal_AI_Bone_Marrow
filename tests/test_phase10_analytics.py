# -*- coding: utf-8 -*-
"""
Tests for Phase 10 Step 3 & 4:
  - blast_sensitivity_range()
  - plausibility_flags()
  - confusion_neighbours()
  - compute_attention_focus()
"""

import unittest
from pipeline.scan_analytics import (
    blast_sensitivity_range,
    plausibility_flags,
    confusion_neighbours,
    compute_attention_focus,
    PLAUSIBILITY_THRESHOLDS,
)
from pipeline.summary import build_morphology_summary
from app.db import DB_PATH
import sqlite3
import json


class TestPhase10Analytics(unittest.TestCase):

    def setUp(self):
        # Load cells for CASE-F09AFAC4
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT cell_results_json FROM cases WHERE case_id=?", ("CASE-F09AFAC4",))
            row = cur.fetchone()
            self.cells_f09 = json.loads(row[0]) if row and row[0] else []

    def test_blast_sensitivity_range_calculation(self):
        total = 49
        res = blast_sensitivity_range(self.cells_f09, total)
        self.assertIn("lower_pct", res)
        self.assertIn("upper_pct", res)
        self.assertIn("label", res)
        self.assertEqual(res["label"], "heuristic, not a statistical interval")
        self.assertGreaterEqual(res["upper_pct"], res["lower_pct"])
        self.assertGreater(res["uncertain_cells_count"], 0)

    def test_plausibility_flags_megakaryocyte_triggers(self):
        summary = build_morphology_summary("CASE-F09AFAC4")
        flags = plausibility_flags(summary, self.cells_f09)
        flag_ids = [f["id"] for f in flags]

        # Megakaryocyte share is 30.6% > 5%, must trigger
        self.assertIn("FLAG-MEGAKARYOCYTE", flag_ids)
        # Total cells is 49 < 200, must trigger
        self.assertIn("FLAG-CELL-COUNT", flag_ids)
        # Low confidence ratio is > 40%, must trigger
        self.assertIn("FLAG-LOW-CONF", flag_ids)

        mega_flag = next(f for f in flags if f["id"] == "FLAG-MEGAKARYOCYTE")
        self.assertIn("Megakaryocyte", mega_flag["title"])
        self.assertIn("General haematology convention", mega_flag["source_label"])

    def test_confusion_neighbours_structure(self):
        neighbours = confusion_neighbours(self.cells_f09)
        self.assertIn("Blast", neighbours)
        self.assertIsInstance(neighbours["Blast"], list)
        self.assertGreater(len(neighbours["Blast"]), 0)
        top_pair, weight = neighbours["Blast"][0]
        self.assertEqual(top_pair, "Promyelocyte")

    def test_compute_attention_focus_on_real_blast(self):
        crop = "data/new_patient_cache/CASE-F09AFAC4/xai/IMAGE-01_CELL-041_Blast_original.png"
        overlay = "data/new_patient_cache/CASE-F09AFAC4/xai/IMAGE-01_CELL-041_Blast_overlay.png"
        focus = compute_attention_focus(crop, overlay)
        self.assertIn("nucleus_share", focus)
        self.assertIn("cytoplasm_share", focus)
        self.assertIn("background_share", focus)
        self.assertIn("caption", focus)
        # Sum of shares should approximate 1.0
        tot = focus["nucleus_share"] + focus["cytoplasm_share"] + focus["background_share"]
        self.assertAlmostEqual(tot, 1.0, places=1)
        # Blast CELL-041 has dense nuclear attention
        self.assertGreater(focus["nucleus_share"], 0.6)
        self.assertTrue(focus["reliable"])

    def test_compute_attention_focus_missing_file_graceful_fallback(self):
        focus = compute_attention_focus("non_existent_path.png", None)
        self.assertIn("nucleus_share", focus)
        self.assertTrue(focus["reliable"])


if __name__ == "__main__":
    unittest.main()

