"""
retrieval/ingest.py
─────────────────────────────────────────────────────────────────────────────
Ingestion and semantic chunk preparation for Pediatric Leukemia Clinical Guidelines.

Reads:
  data/knowledge_base/clinical_guidelines.json (or custom guidelines)

Writes:
  data/processed/retrieval_index/chunks.json (validated, structured corpus)
"""

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_KB_FILE = REPO_ROOT / "data" / "knowledge_base" / "clinical_guidelines.json"
OUTPUT_CHUNKS_FILE = REPO_ROOT / "data" / "processed" / "retrieval_index" / "chunks.json"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ingest")


def clean_text(text: str) -> str:
    """Normalize whitespace and clinical formatting."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate_chunk(chunk: Dict[str, Any], idx: int) -> Dict[str, Any]:
    """Ensures mandatory schema fields are present and well-formed."""
    chunk_id = chunk.get("chunk_id", f"CHUNK_{idx:03d}")
    category = chunk.get("disease_category", "General")
    entity = chunk.get("specific_entity", "Unspecified")
    title = chunk.get("title", "Clinical Guidance")
    text = clean_text(chunk.get("text", ""))
    source = chunk.get("source", "Standard Guidelines")
    keywords = chunk.get("morphological_keywords", [])
    blast_rule = chunk.get("blast_threshold_rule", {})

    if not text:
        raise ValueError(f"Chunk {chunk_id} has empty text!")

    # Rich representation combining title, entity, and text for dense embedding
    embedding_text = f"Entity: {entity} | Category: {category} | {title}: {text}"

    return {
        "chunk_id": chunk_id,
        "disease_category": category,
        "specific_entity": entity,
        "title": title,
        "text": text,
        "source": source,
        "morphological_keywords": keywords,
        "blast_threshold_rule": blast_rule,
        "embedding_text": embedding_text,
    }


def ingest_corpus(
    kb_path: Path = DEFAULT_KB_FILE,
    output_path: Path = OUTPUT_CHUNKS_FILE,
) -> List[Dict[str, Any]]:
    """Loads raw knowledge records, validates schema, and writes standardized chunks."""
    logger.info(f"Ingesting knowledge base from: {kb_path.relative_to(REPO_ROOT)}")
    if not kb_path.exists():
        raise FileNotFoundError(f"Knowledge base file not found at: {kb_path}")

    with open(kb_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if not isinstance(raw_data, list):
        raise ValueError(f"Expected list of chunk dictionaries in {kb_path}")

    processed_chunks = []
    for idx, item in enumerate(raw_data):
        chunk = validate_chunk(item, idx)
        processed_chunks.append(chunk)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(processed_chunks, f, indent=2, ensure_ascii=False)

    logger.info(f"Ingested {len(processed_chunks)} validated clinical chunks -> {output_path.relative_to(REPO_ROOT)}")
    return processed_chunks


if __name__ == "__main__":
    ingest_corpus()
