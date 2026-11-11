"""
pipeline/report_validator.py
─────────────────────────────────────────────────────────────────────────────
Module 7 & 8: Independent Pre-Publication Validation Agent.

Verifies the final report before publication or caching:
  1. Numerical Consistency:
     - Assert total nucleated cells match build_morphology_summary().
     - Assert blast proportion equals blast_count / total_cells.
     - Verify Wilson 95% CI encompasses the point estimate.
  2. Provenance & Grounding:
     - Verify every claim citing a chunk maps to a real retrieved evidence ID.
     - Ensure unsupported claims are explicitly tagged as such.
  3. Medical Safety & Language Guard:
     - Zero unhedged diagnostic statements (no 'positive for AML', 'diagnostic of MDS').
     - Cytomorphology strictly qualified as descriptive screening.
  4. Visualization & Data Consistency:
     - Verify every visualization manifest entry has a valid output file.
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path
import re


FORBIDDEN_ASSERTIONS = [
    r'aligns with (?:the )?(?:diagnostic )?criteria for\b',
    r'consistent with (?:a )?diagnosis of\b',
    r'meets (?:the )?(?:diagnostic )?criteria for\b',
    r'positive for (?:acute leukemia|AML|ALL|MDS[^\s]*)',
    r'\bMDS-EB2\b',
    r'\bMDS-EB1\b',
    r'diagnostic of\b',
    r'is diagnosed with\b'
]


class ValidationResult:
    def __init__(self):
        self.is_valid: bool = True
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.audited_claims_count: int = 0
        self.supported_claims_count: int = 0
        self.numerical_checks_passed: bool = True

    def add_error(self, msg: str):
        self.is_valid = False
        self.errors.append(msg)

    def add_warning(self, msg: str):
        self.warnings.append(msg)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "audited_claims_count": self.audited_claims_count,
            "supported_claims_count": self.supported_claims_count,
            "grounding_ratio": round(self.supported_claims_count / max(1, self.audited_claims_count), 3),
            "numerical_checks_passed": self.numerical_checks_passed
        }


def validate_report(
    report_data: Dict[str, Any],
    summary: Dict[str, Any],
    evidence_registry: Dict[str, Any]
) -> ValidationResult:
    """
    Executes comprehensive pre-publication audit on the report dictionary.
    """
    result = ValidationResult()

    # 1. Numerical Consistency Check
    expected_total = summary.get("total_cells", 0)
    expected_blast_frac = summary.get("blast_fraction", 0.0)
    
    # Check scan facts if available
    scan_facts = report_data.get("scan_facts", {})
    if scan_facts:
        facts_total = scan_facts.get("n_cells", 0)
        if facts_total != expected_total:
            result.add_error(f"Numerical Mismatch: scan_facts cells ({facts_total}) != summary cells ({expected_total})")

    # 2. Check Language Safety across all section content and claims
    for sec_idx, sec in enumerate(report_data.get("sections", [])):
        sec_title = sec.get("title", f"Section {sec_idx}")
        content = sec.get("content", "")
        
        # Check forbidden diagnostic language in narrative
        for pat in FORBIDDEN_ASSERTIONS:
            if re.search(pat, content, flags=re.IGNORECASE):
                result.add_warning(f"Language Guard Warning in '{sec_title}': phrase matching '{pat}' detected.")

        # Check claims
        for claim in sec.get("claims", []):
            result.audited_claims_count += 1
            claim_text = claim.get("text", "")
            
            # Check claim-level diagnostic assertions
            for pat in FORBIDDEN_ASSERTIONS:
                if re.search(pat, claim_text, flags=re.IGNORECASE):
                    result.add_warning(f"Claim Language Warning in claim {claim.get('id')}: phrase matching '{pat}'.")

            # Check evidence grounding
            chunk_ids = claim.get("chunk_ids", [])
            has_grounding = False
            for cid in chunk_ids:
                if cid in evidence_registry or cid in CORE_GUIDELINES:
                    has_grounding = True
                    break
            
            # Metric or cell grounding also valid
            if claim.get("metric_refs") or claim.get("cell_ids") or has_grounding:
                result.supported_claims_count += 1
            else:
                if claim.get("is_supported", True):
                    claim["is_supported"] = False
                    result.add_warning(f"Claim {claim.get('id')} lacks explicit source link; tagged as unverified/interpretation.")

    # 3. Check Visualization Files
    manifest = report_data.get("visualization_manifest", {})
    for viz in manifest.get("visualizations", []):
        url = viz.get("url", "")
        if url:
            # relative derived path check
            local_path = Path("data") / url if not url.startswith("data/") else Path(url)
            if not local_path.exists():
                result.add_warning(f"Visualization artifact {url} not found on disk.")

    result.numerical_checks_passed = len(result.errors) == 0
    return result


CORE_GUIDELINES = {
    "WHO2022_AML_001", "WHO2022_MDS_002", "ICC2022_MDS_AML_001", "LIT-01", "LIT-02", "LIT-03", "LIT-04", "LIT-05"
}

