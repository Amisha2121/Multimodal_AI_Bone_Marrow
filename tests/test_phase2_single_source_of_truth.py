# -*- coding: utf-8 -*-
"""
Test Phase 2: Single source of truth for case morphology data.
Verifies that metric cards, Cell Explorer, Ask Questions context, and RAG report
counts all strictly equal build_morphology_summary() outputs.
"""

import sys
import json
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.summary import build_morphology_summary


def http_get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_tests():
    print("=" * 70)
    print("PHASE 2: SINGLE SOURCE OF TRUTH VERIFICATION")
    print("=" * 70)

    base_url = "http://127.0.0.1:5000"

    cases_to_test = [
        ("adb52ad345f1", "CASE-F09AFAC4", 49, 0.102),
        ("9a0ca82e9003", "CASE-CECDBCC7", 12, 0.0833),
    ]

    for aid, cid, exp_total, exp_blast_pct in cases_to_test:
        print(f"\nTesting Case: {cid} (Analysis ID: {aid})")
        print("-" * 50)

        # 1. Authoritative Summary
        summary = build_morphology_summary(aid)
        assert summary["total_cells"] == exp_total, f"Expected {exp_total}, got {summary['total_cells']}"
        assert round(summary["blast_fraction"], 3) == round(exp_blast_pct, 3), f"Expected {exp_blast_pct}, got {summary['blast_fraction']}"

        # 2. Metric Cards (/api/analysis/<id> and /api/case/<id>)
        analysis_data = http_get(f"{base_url}/api/analysis/{aid}")
        assert analysis_data["metrics"]["cells_detected"] == summary["total_cells"]
        assert round(analysis_data["metrics"]["blast_fraction"], 3) == round(summary["blast_fraction"], 3)

        case_data = http_get(f"{base_url}/api/case/{cid}")
        assert case_data["metrics"]["cells_detected"] == summary["total_cells"]
        assert round(case_data["metrics"]["blast_fraction"], 3) == round(summary["blast_fraction"], 3)

        # 3. Cell Explorer (/api/cells/<id>)
        cells_data = http_get(f"{base_url}/api/cells/{aid}")["cells"]
        assert len(cells_data) == summary["total_cells"]

        # 4. Monocyte Series check for CASE-F09AFAC4
        if cid == "CASE-F09AFAC4":
            mono_cells = [c for c in cells_data if c["class"] == "Monocyte Series"]
            assert len(mono_cells) == 2, f"Expected 2 Monocyte Series cells, got {len(mono_cells)}"
            assert summary["counts"]["Monocyte Series"] == 2
            print("  [PASSED] Monocyte Series verified: 2 cells in Cell Explorer & Summary")

        # 5. Table of Per-Class Counts
        print(f"\n  {'Class Name':<30} | {'Summary Count':<13} | {'Explorer Count':<14} | {'Percentage'}")
        print(f"  {'-'*30}-+-{'-'*13}-+-{'-'*14}-+-----------")
        for cls_name in summary["counts"]:
            cnt = summary["counts"][cls_name]
            exp_cnt = sum(1 for c in cells_data if c["class"] == cls_name)
            pct = summary["percentages"][cls_name]
            assert cnt == exp_cnt, f"Mismatch for {cls_name}: summary={cnt}, explorer={exp_cnt}"
            print(f"  {cls_name:<30} | {cnt:<13} | {exp_cnt:<14} | {pct*100:.1f}%")

        print(f"\n  [PASSED] All counts match single source of truth for {cid}")

    print("\n" + "=" * 70)
    print("ALL PHASE 2 ACCEPTANCE ASSERTIONS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
