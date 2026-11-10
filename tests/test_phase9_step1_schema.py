"""
Phase 9 Step 1 – Claim-Level Schema Tests
==========================================
Run with:
    .\.venv\Scripts\python.exe -m unittest tests/test_phase9_step1_schema.py -v
"""

import sys
import json
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from pipeline.claim_extractor import (
    split_sentences,
    classify_claim_type,
    detect_metric_refs,
    detect_cell_ids,
    extract_claims_from_section,
    extract_claims_from_report,
)


# ---------------------------------------------------------------------------
# Unit tests: sentence splitter
# ---------------------------------------------------------------------------

class TestSplitSentences(unittest.TestCase):

    def test_basic(self):
        text = "There are 5 blasts. This represents 10.2% of total cells. Manual review is recommended."
        sents = split_sentences(text)
        self.assertEqual(len(sents), 3)

    def test_empty(self):
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences(None), [])

    def test_single(self):
        sents = split_sentences("Only one sentence here.")
        self.assertEqual(len(sents), 1)

    def test_abbreviations_not_split(self):
        text = "The case involves Dr. Smith who noted 5 blasts. Manual review required."
        sents = split_sentences(text)
        # Should not split at "Dr." — expect 2 sentences not 3
        self.assertLessEqual(len(sents), 3)


# ---------------------------------------------------------------------------
# Unit tests: type classifier
# ---------------------------------------------------------------------------

class TestClaimType(unittest.TestCase):

    def test_observation(self):
        self.assertEqual(classify_claim_type("Five blasts were detected."), 'observation')

    def test_guideline(self):
        t = classify_claim_type("According to WHO 2022, blasts ≥20% define AML.")
        self.assertEqual(t, 'guideline')

    def test_limitation(self):
        t = classify_claim_type("Interpret these results with caution given the low cell count.")
        self.assertEqual(t, 'limitation')

    def test_followup(self):
        t = classify_claim_type("Flow cytometry and cytogenetics are recommended for further workup.")
        self.assertEqual(t, 'followup')


# ---------------------------------------------------------------------------
# Unit tests: metric-ref detector
# ---------------------------------------------------------------------------

class TestMetricRefs(unittest.TestCase):

    def test_blast_pct(self):
        refs = detect_metric_refs("Blast proportion is 10.2% of 49 cells.")
        self.assertIn('blast_pct', refs)

    def test_total_cells(self):
        refs = detect_metric_refs("A total of 49 cells were classified.")
        self.assertIn('total_cells', refs)

    def test_blast_ci(self):
        refs = detect_metric_refs("95% CI 3.4%-23.6% using the Wilson interval.")
        self.assertIn('blast_ci', refs)

    def test_no_refs(self):
        refs = detect_metric_refs("The analysis employs a Vision Transformer.")
        self.assertEqual(refs, [])


# ---------------------------------------------------------------------------
# Unit tests: cell-id detector
# ---------------------------------------------------------------------------

class TestCellIds(unittest.TestCase):

    def test_short_format(self):
        ids = detect_cell_ids("CELL-001 and CELL-042 were reviewed.")
        self.assertIn('CELL-001', ids)
        self.assertIn('CELL-042', ids)

    def test_no_match_without_available(self):
        # Without restriction, any CELL-### matches
        ids = detect_cell_ids("CELL-999 flagged.")
        self.assertIn('CELL-999', ids)

    def test_filter_by_available(self):
        ids = detect_cell_ids("CELL-001 and CELL-999.", available_cell_ids=['CELL-001'])
        self.assertIn('CELL-001', ids)
        # CELL-999 not in available → should NOT be in result
        self.assertNotIn('CELL-999', ids)

    def test_no_ids(self):
        ids = detect_cell_ids("No cell references here.")
        self.assertEqual(ids, [])


# ---------------------------------------------------------------------------
# Integration test: extract_claims_from_section
# ---------------------------------------------------------------------------

class TestExtractClaimsFromSection(unittest.TestCase):

    SECTION_TEXT = (
        "Five blast cells were detected, representing 10.2% of 49 classified cells. "
        "According to WHO 2022 criteria, a blast proportion below 20% does not meet "
        "the diagnostic threshold for AML. Flow cytometry and cytogenetic analysis "
        "are recommended for further workup. Interpret these findings with caution "
        "given the low cell count of 49 cells."
    )

    def test_all_claims_have_id(self):
        claims = extract_claims_from_section(
            self.SECTION_TEXT, "Case Summary", summary=None, chunk_ids=["WHO2022_BALL_001"]
        )
        ids = [c['id'] for c in claims]
        self.assertEqual(len(ids), len(set(ids)), "Claim IDs must be unique")
        for cid in ids:
            self.assertRegex(cid, r'^c\d+$', "ID must be cN format")

    def test_required_fields(self):
        claims = extract_claims_from_section(
            self.SECTION_TEXT, "Case Summary", summary=None
        )
        for c in claims:
            for field in ('id', 'text', 'type', 'metric_refs', 'cell_ids', 'chunk_ids', 'is_supported', 'critic_status'):
                self.assertIn(field, c, f"Missing field: {field}")

    def test_blast_sentence_has_metric_ref(self):
        claims = extract_claims_from_section(
            self.SECTION_TEXT, "Case Summary", summary=None
        )
        blast_claims = [c for c in claims if 'blast' in c['text'].lower() and '10.2' in c['text']]
        self.assertTrue(len(blast_claims) > 0, "Blast sentence not found")
        self.assertIn('blast_pct', blast_claims[0]['metric_refs'])

    def test_guideline_claim_type(self):
        claims = extract_claims_from_section(
            self.SECTION_TEXT, "Case Summary", summary=None
        )
        who_claims = [c for c in claims if 'WHO' in c['text']]
        self.assertTrue(len(who_claims) > 0)
        self.assertEqual(who_claims[0]['type'], 'guideline')

    def test_followup_claim_type(self):
        claims = extract_claims_from_section(
            self.SECTION_TEXT, "Case Summary", summary=None
        )
        fu_claims = [c for c in claims if 'recommended' in c['text'].lower()]
        self.assertTrue(len(fu_claims) > 0)
        self.assertEqual(fu_claims[0]['type'], 'followup')

    def test_chunk_ids_propagated(self):
        claims = extract_claims_from_section(
            self.SECTION_TEXT, "Case Summary", summary=None,
            chunk_ids=["WHO2022_BALL_001", "ICC2022_AML_002"]
        )
        for c in claims:
            self.assertIn('WHO2022_BALL_001', c['chunk_ids'])

    def test_unsupported_claim_marked(self):
        claims = extract_claims_from_section(
            "The analysis uses a Vision Transformer.",
            "Analytical Methodology", summary=None, chunk_ids=[]
        )
        # No metric refs, no cell IDs, no chunk IDs → is_supported = False
        for c in claims:
            self.assertFalse(c['is_supported'])

    def test_id_offset(self):
        claims = extract_claims_from_section(
            "One sentence.", "Title", summary=None, id_offset=10
        )
        self.assertEqual(claims[0]['id'], 'c11')


# ---------------------------------------------------------------------------
# Integration test: extract_claims_from_report
# ---------------------------------------------------------------------------

class TestExtractClaimsFromReport(unittest.TestCase):

    def _make_report(self):
        return {
            'sections': [
                {
                    'title': 'Case Summary',
                    'content': (
                        "Five blasts were detected, representing 10.2% of 49 cells. "
                        "According to WHO 2022, blasts below 20% do not define AML."
                    ),
                    'cell_images': [],
                },
                {
                    'title': 'Analytical Methodology',
                    'content': "A Vision Transformer was used for classification.",
                    'cell_images': [],
                },
            ],
            'sources': ['WHO2022_BALL_001'],
            'case_id': 'CASE-TEST',
        }

    def test_schema_version_set(self):
        report = self._make_report()
        result = extract_claims_from_report(report)
        self.assertEqual(result['schema_version'], 'v2')
        self.assertTrue(result['has_claims'])

    def test_idempotent(self):
        report = self._make_report()
        extract_claims_from_report(report)
        # Second call should not re-extract (schema_version already v2)
        section_claims_before = [len(s.get('claims', [])) for s in report['sections']]
        extract_claims_from_report(report)
        section_claims_after = [len(s.get('claims', [])) for s in report['sections']]
        self.assertEqual(section_claims_before, section_claims_after)

    def test_every_section_has_claims(self):
        report = self._make_report()
        extract_claims_from_report(report)
        for sec in report['sections']:
            self.assertIn('claims', sec, f"Section '{sec['title']}' missing claims")

    def test_global_uniqueness(self):
        report = self._make_report()
        extract_claims_from_report(report)
        all_ids = [c['id'] for s in report['sections'] for c in s.get('claims', [])]
        self.assertEqual(len(all_ids), len(set(all_ids)), "Claim IDs must be globally unique")

    def test_old_report_without_claims_no_crash(self):
        """Old report without schema_version should get claims added gracefully."""
        old_report = {
            'sections': [{'title': 'Case Summary', 'content': 'Old content.', 'cell_images': []}],
            'sources': [],
            'case_id': 'CASE-OLD',
        }
        result = extract_claims_from_report(old_report)
        self.assertEqual(result['schema_version'], 'v2')

    def test_no_forbidden_diagnostic_phrases(self):
        """Claims must not contain diagnostic conclusions (language guard)."""
        report = self._make_report()
        extract_claims_from_report(report)
        forbidden = ['the patient has', 'diagnosed with', 'confirms aml', 'definitive diagnosis']
        for sec in report['sections']:
            for claim in sec.get('claims', []):
                text_lower = claim['text'].lower()
                for phrase in forbidden:
                    self.assertNotIn(phrase, text_lower,
                                     f"Forbidden phrase '{phrase}' in claim: {claim['text']}")


# ---------------------------------------------------------------------------
# DB schema tests
# ---------------------------------------------------------------------------

class TestDbPhase9Schema(unittest.TestCase):
    """Verify new DB tables and columns exist after init_db()."""

    def test_report_versions_table(self):
        from app.db import init_db, get_db_connection
        init_db()
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='report_versions'")
            self.assertIsNotNone(cur.fetchone(), "report_versions table missing")

    def test_annotations_json_column(self):
        from app.db import init_db, get_db_connection
        init_db()
        with get_db_connection() as conn:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(cases)")
            cols = {row['name'] for row in cur.fetchall()}
            self.assertIn('annotations_json', cols, "annotations_json column missing")

    def test_save_and_get_report_version(self):
        from app.db import save_report_version, get_report_versions, get_report_version
        fake_case = 'CASE-TEST-P9-SCHEMA'
        fake_aid  = 'aidtestschema001'
        fake_report = {'sections': [], 'case_id': fake_case, 'schema_version': 'v2'}
        v = save_report_version(fake_case, fake_aid, fake_report, trigger='test', generator='test')
        self.assertIsNotNone(v)
        versions = get_report_versions(fake_case)
        self.assertTrue(any(r['version_num'] == v for r in versions))
        fetched = get_report_version(fake_case, v)
        self.assertEqual(fetched['case_id'], fake_case)

    def test_save_and_get_annotations(self):
        from app.db import save_annotations, get_annotations, init_db, get_db_connection
        # Insert a dummy case row so the UPDATE has something to target
        init_db()
        fake_case = 'CASE-TEST-ANN'
        with get_db_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO cases (case_id, analysis_id, timestamp) VALUES (?,?,?)",
                (fake_case, 'aidann001', '2026-01-01')
            )
            conn.commit()
        ann = {'c1': {'action': 'agree', 'note': 'Looks correct.'}}
        ok = save_annotations(fake_case, ann)
        self.assertTrue(ok)
        fetched = get_annotations(fake_case)
        self.assertEqual(fetched['c1']['action'], 'agree')
        with get_db_connection() as conn:
            conn.execute("DELETE FROM cases WHERE case_id = ?", (fake_case,))
            conn.commit()


if __name__ == '__main__':
    unittest.main(verbosity=2)

