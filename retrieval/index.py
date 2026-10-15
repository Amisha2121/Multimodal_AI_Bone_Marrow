"""
retrieval/index.py
─────────────────────────────────────────────────────────────────────────────
Hybrid Dense + Sparse Indexing Engine for Pediatric Leukemia Guidelines.

Components:
  1. Dense Index: BAAI/bge-small-en-v1.5 (or all-MiniLM-L6-v2) embeddings + FAISS IndexFlatIP
  2. Sparse Index: BM25Okapi lexical matching on clinical terminology
  3. Metadata: Chunk mapping, embedding dimensionality, timestamp

Outputs:
  data/processed/retrieval_index/dense_index.faiss
  data/processed/retrieval_index/dense_embeddings.npy
  data/processed/retrieval_index/sparse_bm25.pkl
  data/processed/retrieval_index/index_metadata.json
"""

import json
import logging
import pickle
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

CHUNKS_FILE = REPO_ROOT / "data" / "processed" / "retrieval_index" / "chunks.json"
INDEX_DIR = REPO_ROOT / "data" / "processed" / "retrieval_index"

FAISS_PATH = INDEX_DIR / "dense_index.faiss"
EMBEDDINGS_PATH = INDEX_DIR / "dense_embeddings.npy"
BM25_PATH = INDEX_DIR / "sparse_bm25.pkl"
METADATA_PATH = INDEX_DIR / "index_metadata.json"

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"
FALLBACK_MODEL_NAME = "all-MiniLM-L6-v2"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("index")


def tokenize_text(text: str) -> List[str]:
    """Tokenize clinical text for BM25 with lowercase and alphanumeric extraction."""
    return [tok for tok in re.findall(r"\b[a-zA-Z0-9_\-\+]+(?:::)?\b", text.lower()) if len(tok) > 1]


def load_embedding_model(preferred_model: str = DEFAULT_MODEL_NAME) -> SentenceTransformer:
    """Loads embedding model with automatic fallback if offline or download issue."""
    try:
        logger.info(f"Loading embedding model: {preferred_model}...")
        return SentenceTransformer(preferred_model)
    except Exception as e:
        logger.warning(f"Could not load {preferred_model}: {e}. Trying fallback: {FALLBACK_MODEL_NAME}...")
        return SentenceTransformer(FALLBACK_MODEL_NAME)


def build_indices(
    chunks_path: Path = CHUNKS_FILE,
    model_name: str = DEFAULT_MODEL_NAME,
) -> None:
    """Builds and serializes both dense FAISS and sparse BM25 indices."""
    t0 = time.time()
    logger.info("=" * 65)
    logger.info("BUILDING HYBRID RETRIEVAL INDICES (Dense FAISS + Sparse BM25)")
    logger.info("=" * 65)

    if not chunks_path.exists():
        from retrieval.ingest import ingest_corpus
        ingest_corpus()

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    logger.info(f"Loaded {len(chunks)} chunks from {chunks_path.name}")
    embedding_texts = [c["embedding_text"] for c in chunks]

    # ── 1. Dense Indexing with SentenceTransformer + FAISS ───────────────
    model = load_embedding_model(model_name)
    logger.info(f"Encoding {len(embedding_texts)} chunks into dense vectors...")
    embeddings = model.encode(
        embedding_texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype("float32")

    dimension = embeddings.shape[1]
    logger.info(f"Embeddings shape: {embeddings.shape} (dim={dimension})")

    # FAISS Inner Product (cosine similarity since vectors are L2-normalized)
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)
    logger.info(f"FAISS index built. Total entries in index: {faiss_index.ntotal}")

    # ── 2. Sparse Lexical Indexing with BM25 ──────────────────────────────
    logger.info("Tokenizing chunks for BM25 sparse index...")
    tokenized_corpus = []
    for c in chunks:
        # Combine text and explicit morphological keywords for strong lexical matching
        doc_tokens = tokenize_text(c["embedding_text"] + " " + " ".join(c.get("morphological_keywords", [])))
        tokenized_corpus.append(doc_tokens)

    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("BM25 index built successfully.")

    # ── 3. Serialize Artifacts ───────────────────────────────────────────
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(faiss_index, str(FAISS_PATH))
    np.save(str(EMBEDDINGS_PATH), embeddings)

    with open(BM25_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "tokenized_corpus": tokenized_corpus}, f)

    metadata = {
        "num_chunks": len(chunks),
        "embedding_model": model_name,
        "dimension": dimension,
        "similarity_metric": "cosine",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chunk_ids": [c["chunk_id"] for c in chunks],
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    elapsed = time.time() - t0
    logger.info(f"Hybrid indices saved to {INDEX_DIR.relative_to(REPO_ROOT)} in {elapsed:.2f}s")
    logger.info("=" * 65)


if __name__ == "__main__":
    build_indices()
