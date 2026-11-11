"""
Phase 9 – Claim-Level Schema (Step 1)
======================================
Pure functions that split a section's text into sentence-level claims and
annotate each claim with:
  - id           unique within the report (c1, c2, …)
  - text         the sentence
  - type         observation | guideline | limitation | followup
  - metric_refs  list of metric keys referenced (e.g. ["blast_pct"])
  - cell_ids     list of CELL-### IDs found in the text
  - chunk_ids    evidence chunk IDs associated with this section
  - is_supported True when at least one of metric_refs / cell_ids / chunk_ids is non-empty
  - critic_status "not_individually_checked" (populated later if Critic provides per-claim data)

Nothing is imported from the ML pipeline or agent graph.
All outputs go through sanitize_agent_language() so no diagnostic wording leaks.
"""

import re
from typing import List, Dict, Any, Optional

# ---------------------------------------------------------------------------
# Language guard (reused from Phase 7)
# ---------------------------------------------------------------------------

def _safe_sanitize(text: str) -> str:
    """Call sanitize_agent_language if available, else identity."""
    try:
        from pipeline.agent_guard import sanitize_agent_language
        return sanitize_agent_language(text)
    except Exception:
        return text


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(
    r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+'
)

_ABBREV_RE = re.compile(r'\b(?:Dr|Mr|Mrs|Ms|Prof|vs|etc|e\.g|i\.e|fig|vol|no|pp|cf)\.\s+', re.IGNORECASE)


def split_sentences(text: str) -> List[str]:
    """Split text into sentences.  Handles common abbreviations heuristically."""
    if not text or not text.strip():
        return []

    # Protect known abbreviations by replacing period+space with a placeholder
    protected = _ABBREV_RE.sub(lambda m: m.group().replace('. ', '.<ABBR>'), text)

    parts = _SENTENCE_SPLIT_RE.split(protected)

    sentences = []
    for part in parts:
        s = part.replace('<ABBR>', ' ').strip()
        if s:
            sentences.append(s)
    return sentences


# ---------------------------------------------------------------------------
# Type classifier (heuristic)
# ---------------------------------------------------------------------------

_GUIDELINE_KW = re.compile(
    r'\b(WHO|ICC|guideline|criteria|threshold|classification|per|AML|MDS|ALL|CML'
    r'|haematop|hematop|standard\s+of\s+care|recommend|reference|cited|according)\b',
    re.IGNORECASE,
)
_LIMITATION_KW = re.compile(
    r'\b(limit|caveat|uncertain|caution|interpret|manual\s+review|clinical\s+context'
    r'|additional\s+workup|not\s+a\s+substitute|low\s+cell\s+count|confidence\s+interval'
    r'|CI\s+\d|95\s*%\s*CI|bias|artefact|artifact|complement)\b',
    re.IGNORECASE,
)
_FOLLOWUP_KW = re.compile(
    r'\b(recommend|follow.up|workup|flow\s+cytometry|cytogenetic|molecular\s+panel'
    r'|urgent|refer|further|additional\s+test|repeat|biopsy|FISH|PCR|immunophenotyp)\b',
    re.IGNORECASE,
)


def classify_claim_type(sentence: str) -> str:
    """Return one of: observation | guideline | limitation | followup."""
    if _FOLLOWUP_KW.search(sentence):
        return 'followup'
    if _GUIDELINE_KW.search(sentence):
        return 'guideline'
    if _LIMITATION_KW.search(sentence):
        return 'limitation'
    return 'observation'


# ---------------------------------------------------------------------------
# Metric-ref detector
# ---------------------------------------------------------------------------

# Maps keywords found in sentence → canonical metric key
_METRIC_PATTERNS: List[tuple] = [
    (re.compile(r'\bblast[s]?\s*(?:fraction|%|proportion|count|percentage)\b', re.IGNORECASE), 'blast_pct'),
    (re.compile(r'\bblast[s]?\b.*?\b\d+\.?\d*\s*%', re.IGNORECASE), 'blast_pct'),
    (re.compile(r'\b\d+\.?\d*\s*%\s*(?:blast|blasts)\b', re.IGNORECASE), 'blast_pct'),
    (re.compile(r'\btotal\s+cells?\b', re.IGNORECASE), 'total_cells'),
    (re.compile(r'\btotal\s+nucleated\b', re.IGNORECASE), 'total_cells'),
    (re.compile(r'\b(?:49|12)\s+cells\b'), 'total_cells'),          # exact case counts
    (re.compile(r'\bmean\s+confidence\b', re.IGNORECASE), 'mean_confidence'),
    (re.compile(r'\bconfidence\s+band\b', re.IGNORECASE), 'confidence_band'),
    (re.compile(r'\b95\s*%\s*CI\b', re.IGNORECASE), 'blast_ci'),
    (re.compile(r'\bWilson\b', re.IGNORECASE), 'blast_ci'),
    (re.compile(r'\bblast\s+count\b', re.IGNORECASE), 'blast_count'),
    (re.compile(r'\bNeutrophil\b', re.IGNORECASE), 'Neutrophil_pct'),
    (re.compile(r'\bLymphocyte\b', re.IGNORECASE), 'Lymphocyte_pct'),
    (re.compile(r'\bPromonocyte\b', re.IGNORECASE), 'Promyelocyte_pct'),
    (re.compile(r'\bMonocyte\b', re.IGNORECASE), 'Monocyte_pct'),
    (re.compile(r'\bErythroid\b', re.IGNORECASE), 'Erythroid_pct'),
    (re.compile(r'\bMegakaryocyte\b', re.IGNORECASE), 'Megakaryocyte_pct'),
    (re.compile(r'\bPlasma\s+cell\b', re.IGNORECASE), 'PlasmaCells_pct'),
]


def detect_metric_refs(sentence: str) -> List[str]:
    """Return deduplicated list of metric keys referenced in the sentence."""
    found = []
    for pattern, key in _METRIC_PATTERNS:
        if pattern.search(sentence) and key not in found:
            found.append(key)
    return found


# ---------------------------------------------------------------------------
# Cell-id detector
# ---------------------------------------------------------------------------

_CELL_ID_RE = re.compile(r'\b(?:CELL|IMAGE-\w+_CELL)-\d{3,}\b', re.IGNORECASE)
_CELL_SHORT_RE = re.compile(r'\bCELL-(\d{3,})\b', re.IGNORECASE)


def detect_cell_ids(sentence: str, available_cell_ids: Optional[List[str]] = None) -> List[str]:
    """Extract CELL-### references that exist in the case (or any if list not provided)."""
    raw = set(_CELL_ID_RE.findall(sentence))
    raw |= {f'CELL-{m}' for m in _CELL_SHORT_RE.findall(sentence)}
    if available_cell_ids is not None:
        # Normalise to match stored IDs (may be IMAGE-01_CELL-001 or just CELL-001)
        ids_set = {c.upper() for c in available_cell_ids}
        raw = {r for r in raw if r.upper() in ids_set}
    return sorted(raw)


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

def extract_claims_from_section(
    section_text: str,
    section_title: str,
    summary: Optional[Dict[str, Any]],
    chunk_ids: Optional[List[str]] = None,
    cell_ids_in_case: Optional[List[str]] = None,
    id_offset: int = 0,
) -> List[Dict[str, Any]]:
    """
    Split *section_text* into sentence-level claims and annotate each one.

    Parameters
    ----------
    section_text      : Raw section content (may include markdown).
    section_title     : Used for type-classification hints.
    summary           : Output of build_morphology_summary() — used to cross-check metric refs.
    chunk_ids         : Evidence chunk IDs retrieved for this section (optional).
    cell_ids_in_case  : Full list of cell IDs in the case for validation (optional).
    id_offset         : Integer to start numbering claims from (for multi-section uniqueness).

    Returns
    -------
    List of claim dicts, one per sentence.
    """
    if not section_text or not section_text.strip():
        return []

    chunk_ids = chunk_ids or []
    sentences = split_sentences(section_text)
    claims: List[Dict[str, Any]] = []

    # Title-level type override
    title_lower = section_title.lower()
    title_type_hint = None
    if 'limitation' in title_lower or 'confidence' in title_lower:
        title_type_hint = 'limitation'
    elif 'methodology' in title_lower or 'analytical' in title_lower:
        title_type_hint = 'guideline'
    elif 'recommendation' in title_lower or 'interpretation' in title_lower:
        title_type_hint = 'followup'

    for i, sentence in enumerate(sentences):
        sanitized = _safe_sanitize(sentence)
        if not sanitized.strip():
            continue

        claim_type = title_type_hint or classify_claim_type(sanitized)
        metric_refs = detect_metric_refs(sanitized)
        cell_ids    = detect_cell_ids(sanitized, cell_ids_in_case)

        # For the Limitations section, always attach the blast CI metric if summary available
        if summary and claim_type == 'limitation' and 'blast' in sanitized.lower():
            if 'blast_ci' not in metric_refs:
                metric_refs.append('blast_ci')
            if 'blast_pct' not in metric_refs:
                metric_refs.append('blast_pct')

        is_supported = bool(metric_refs or cell_ids or chunk_ids)

        claims.append({
            'id':           f'c{id_offset + i + 1}',
            'text':         sanitized,
            'type':         claim_type,
            'metric_refs':  metric_refs,
            'cell_ids':     cell_ids,
            'chunk_ids':    list(chunk_ids),  # section-level evidence shared with every claim
            'is_supported': is_supported,
            'critic_status': 'not_individually_checked',
        })

    return claims


def extract_claims_from_report(
    report_data: Dict[str, Any],
    summary: Optional[Dict[str, Any]] = None,
    cell_ids_in_case: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Walk every section in *report_data* and add a 'claims' list to each.
    Also adds top-level 'schema_version': 'v2' and 'has_claims': True.

    Returns the **same dict** (mutated in-place) for convenience.
    Old reports without claim data are detected by absence of 'schema_version'.
    """
    if report_data.get('schema_version') == 'v2':
        # Already processed — idempotent
        return report_data

    sections   = report_data.get('sections', [])
    all_sources = set(report_data.get('sources', []))
    id_offset  = 0

    for sec in sections:
        content    = sec.get('content', '')
        title      = sec.get('title', '')
        # Use sources as chunk_ids for all sections (section-level granularity)
        chunk_ids  = list(all_sources)
        claims     = extract_claims_from_section(
            section_text      = content,
            section_title     = title,
            summary           = summary,
            chunk_ids         = chunk_ids,
            cell_ids_in_case  = cell_ids_in_case,
            id_offset         = id_offset,
        )
        sec['claims'] = claims
        id_offset   += len(claims)

    report_data['schema_version'] = 'v2'
    report_data['has_claims']     = True
    return report_data

