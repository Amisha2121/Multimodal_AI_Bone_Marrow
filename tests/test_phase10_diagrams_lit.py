# -*- coding: utf-8 -*-
"""
Tests for Phase 10 Step 6 & 7: Diagrams & Literature Module.
"""

import unittest
from pathlib import Path
import json

from pipeline.report_diagrams import (
    generate_d1_pipeline_provenance,
    generate_d2_blast_ruler,
    generate_d3_confusion_neighbours,
    generate_d4_evidence_chain,
    generate_all_diagrams,
)
from pipeline.summary import build_morphology_summary
from app.db import DATA_DIR, DB_PATH
import sqlite3


class TestPhase10DiagramsAndLiterature(unittest.TestCase):

    def setUp(self):
        self.cid = "CASE-F09AFAC4"
        with sqlite3.connect(str(DB_PATH)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT cell_results_json FROM cases WHERE case_id=?", (self.cid,))
            row = cur.fetchone()
            self.cells = json.loads(row[0]) if row and row[0] else []
        self.summary = build_morphology_summary(self.cid)

    def test_diagrams_generated_as_svgs(self):
        diags = generate_all_diagrams(self.cid, self.summary, self.cells, ["WHO2022_BALL_001"])
        for k in ["d1", "d2", "d3", "d4"]:
            self.assertIn(k, diags)
            p = Path(diags[k])
            self.assertTrue(p.exists())
            self.assertTrue(str(p).endswith(".svg"))
            content = p.read_text(encoding="utf-8")
            self.assertTrue(content.startswith("<svg"))
            self.assertTrue(content.strip().endswith("</svg>"))

    def test_blast_ruler_includes_literature_thresholds(self):
        p = Path(generate_d2_blast_ruler(self.cid, 0.102, 49, {"lower_pct": 2.0, "upper_pct": 36.7}))
        svg = p.read_text(encoding="utf-8")
        self.assertIn("5% (MDS)", svg)
        self.assertIn("10% (CML-AP)", svg)
        self.assertIn("20% (Acute)", svg)
        self.assertIn("Observed: 10.2%", svg)

    def test_background_literature_seed_content(self):
        lit_file = DATA_DIR / "background_literature.json"
        self.assertTrue(lit_file.exists())
        with open(lit_file, "r", encoding="utf-8") as f:
            items = json.load(f)
        self.assertEqual(len(items), 13)
        ids = [it["id"] for it in items]
        for i in range(1, 14):
            self.assertIn(f"LIT-{i:02d}", ids)


if __name__ == "__main__":
    unittest.main()

