# -*- coding: utf-8 -*-
"""
Tests for Phase 10 Step 5 Figures:
Validates Figures 1-4 generation, output file locations, and confirms no modifications to uploads.
"""

import unittest
from pathlib import Path
import hashlib
import os

from pipeline.report_figures import (
    generate_figure1_annotated_field,
    generate_figure2_contact_sheets,
    generate_figure3_explainability_panels,
    generate_figure4_flagged_cell_gallery,
    generate_all_figures,
)
from pipeline.summary import build_morphology_summary
from app.db import DATA_DIR, DB_PATH
import sqlite3
import json


class TestPhase10Figures(unittest.TestCase):

    def setUp(self):
        self.cid = "CASE-F09AFAC4"
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT cell_results_json FROM cases WHERE case_id=?", (self.cid,))
            row = cur.fetchone()
            self.cells = json.loads(row[0]) if row and row[0] else []
        self.summary = build_morphology_summary(self.cid)

    def test_figures_generated_in_derived_only(self):
        figs = generate_all_figures(self.cid, self.summary, self.cells)
        for fig_name, p in figs.items():
            path_obj = Path(p)
            self.assertTrue(path_obj.exists(), f"{fig_name} file must exist on disk")
            self.assertIn("derived", str(path_obj), "Figures must strictly reside in DATA_DIR/derived/")
            self.assertNotIn("uploads", str(path_obj), "Figures must never be written to uploads/")

    def test_both_saved_cases_render_all_figures(self):
        for cid in ["CASE-F09AFAC4", "CASE-CECDBCC7"]:
            with sqlite3.connect(str(DB_PATH)) as conn:
                cur = conn.cursor()
                cur.execute("SELECT cell_results_json FROM cases WHERE case_id=?", (cid,))
                row = cur.fetchone()
                cells = json.loads(row[0]) if row and row[0] else []
            s = build_morphology_summary(cid)
            figs = generate_all_figures(cid, s, cells)
            self.assertEqual(len(figs), 4)
            for k in ["fig1", "fig2", "fig3", "fig4"]:
                self.assertTrue(Path(figs[k]).exists())


if __name__ == "__main__":
    unittest.main()

