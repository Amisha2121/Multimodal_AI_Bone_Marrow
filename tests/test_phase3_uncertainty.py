# -*- coding: utf-8 -*-
"""
Phase 3 Verification Tests - Uncertainty and Statistical Limits.
Tests Wilson score CI calculation, confidence bands, needs_review flagging,
report "Confidence and Limitations" section, and API endpoints.
"""

import sys
import unittest
import requests
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.uncertainty import (
    wilson_interval,
    confidence_band,
    needs_review,
    build_confidence_and_limits_section,
)
from pipeline.summary import build_morphology_summary


class TestPhase3Uncertainty(unittest.TestCase):
    def test_pure_helpers(self):
        # Wilson interval test
        ci_49 = wilson_interval(5, 49)
        self.assertIsNotNone(ci_49)
        low, high = ci_49
        observed = 5 / 49  # 0.10204
        self.assertLessEqual(low, observed)
        self.assertGreaterEqual(high, observed)
        # Low should be ~0.044, high ~0.218
        self.assertAlmostEqual(low, 0.0444, places=3)
        self.assertAlmostEqual(high, 0.2176, places=3)

        # 12 cells test (CASE-CECDBCC7: 1 blast, 12 cells = 8.3%)
        ci_12 = wilson_interval(1, 12)
        self.assertIsNotNone(ci_12)
        low_12, high_12 = ci_12
        obs_12 = 1 / 12  # 0.0833
        self.assertLessEqual(low_12, obs_12)
        self.assertGreaterEqual(high_12, obs_12)
        self.assertAlmostEqual(low_12, 0.0149, places=3)
        self.assertAlmostEqual(high_12, 0.3538, places=3)

        # Confidence bands test
        self.assertEqual(confidence_band(0.85), "high")
        self.assertEqual(confidence_band(0.70), "high")
        self.assertEqual(confidence_band(0.69), "medium")
        self.assertEqual(confidence_band(0.40), "medium")
        self.assertEqual(confidence_band(0.39), "low")
        self.assertEqual(confidence_band(0.10), "low")
        self.assertIsNone(confidence_band(None))

        # Needs review test
        self.assertTrue(needs_review(0.49))
        self.assertTrue(needs_review(0.20))
        self.assertFalse(needs_review(0.50))
        self.assertFalse(needs_review(0.85))
        self.assertFalse(needs_review(None))

    def test_summary_and_report_section_case_f09afac4(self):
        summary = build_morphology_summary("CASE-F09AFAC4")
        self.assertEqual(summary["total_cells"], 49)
        self.assertEqual(summary["blast_count"], 5)
        self.assertAlmostEqual(summary["blast_fraction"], 0.102, places=2)

        section = build_confidence_and_limits_section(summary)
        self.assertEqual(section["title"], "Confidence and Limitations")
        content = section["content"]
        self.assertIn("Blast classification is the least reliable class in published automated systems", content)
        self.assertIn("95% CI: 4.4%–21.8%", content)
        self.assertIn("Low cell count (<200 cells)", content)
        self.assertIn("Total Evaluated Nucleated Cells:** 49", content)

    def test_summary_and_report_section_case_cecdbcc7(self):
        summary = build_morphology_summary("CASE-CECDBCC7")
        self.assertEqual(summary["total_cells"], 12)
        self.assertEqual(summary["blast_count"], 1)
        self.assertAlmostEqual(summary["blast_fraction"], 0.083, places=2)

        section = build_confidence_and_limits_section(summary)
        content = section["content"]
        self.assertIn("95% CI: 1.5%–35.4%", content)
        self.assertIn("Low cell count (<200 cells)", content)
        self.assertIn("Total Evaluated Nucleated Cells:** 12", content)

    def test_live_api_endpoints(self):
        base_url = "http://127.0.0.1:5000"
        try:
            r = requests.get(f"{base_url}/api/health", timeout=10)
        except Exception:
            self.skipTest("Flask server is not currently running.")

        # Test case endpoint returns blast_ci
        r1 = requests.get(f"{base_url}/api/case/CASE-F09AFAC4")
        self.assertEqual(r1.status_code, 200)
        d1 = r1.json()
        self.assertEqual(d1["metrics"]["cells_detected"], 49)
        self.assertAlmostEqual(d1["metrics"]["blast_fraction"], 0.102, places=2)
        self.assertIsNotNone(d1["metrics"].get("blast_ci"))
        ci1 = d1["metrics"]["blast_ci"]
        self.assertAlmostEqual(ci1[0], 0.0444, places=3)
        self.assertAlmostEqual(ci1[1], 0.2176, places=3)

        # Test report endpoint includes Confidence and Limitations section
        r_rep = requests.get(f"{base_url}/api/analysis-report/CASE-F09AFAC4")
        self.assertEqual(r_rep.status_code, 200)
        rep = r_rep.json()
        sec_titles = [s["title"] for s in rep.get("sections", [])]
        self.assertIn("Confidence and Limitations", sec_titles)
        conf_sec = next(s for s in rep["sections"] if s["title"] == "Confidence and Limitations")
        self.assertIn("Blast classification is the least reliable class", conf_sec["content"])


if __name__ == "__main__":
    unittest.main()
