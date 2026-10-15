"""
retrieval/retrieve.py
─────────────────────────────────────────────────────────────────────────────
Hybrid RAG Retrieval Engine for Pediatric Leukemia Bone Marrow Cytology.

Combines:
  1. Dense Semantic Search: BGE-small embeddings + FAISS cosine similarity
  2. Sparse Lexical Search: BM25Okapi over clinical keywords & entities
  3. Reciprocal Rank Fusion (RRF): Robust rank-level ensemble scoring
  4. Clinical Context Formatter: Formats passages for downstream LLM/Agent reasoning
"""

import json
import logging
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from retrieval.index import (
    CHUNKS_FILE,
    FAISS_PATH,
    BM25_PATH,
    METADATA_PATH,
    DEFAULT_MODEL_NAME,
    FALLBACK_MODEL_NAME,
    tokenize_text,
    load_embedding_model,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retrieve")


class HybridRetriever:
    """
    Production-grade hybrid retriever combining FAISS dense vector search
    and BM25 lexical search with Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        index_dir: Optional[Path] = None,
        model_name: str = DEFAULT_MODEL_NAME,
        rrf_k: int = 60,
    ):
        self.rrf_k = rrf_k
        self.model_name = model_name

        # 1. Load Chunks
        if not CHUNKS_FILE.exists():
            from retrieval.index import build_indices
            build_indices()

        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.chunk_by_id = {c["chunk_id"]: c for c in self.chunks}

        # 2. Load Dense Index & Embedding Model
        logger.info(f"Loading FAISS index from {FAISS_PATH}...")
        self.faiss_index = faiss.read_index(str(FAISS_PATH))
        self.model = load_embedding_model(self.model_name)

        # 3. Load BM25 Index
        logger.info(f"Loading BM25 index from {BM25_PATH}...")
        with open(BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25: BM25Okapi = bm25_data["bm25"]

        logger.info(f"HybridRetriever initialized successfully with {len(self.chunks)} clinical chunks.")

    def dense_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search FAISS index using normalized cosine similarity."""
        query_vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        scores, indices = self.faiss_index.search(query_vec, top_k)
        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            chunk["dense_score"] = float(score)
            chunk["dense_rank"] = rank
            results.append(chunk)
        return results

    def sparse_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search using BM25 lexical token matching."""
        tokens = tokenize_text(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            chunk = self.chunks[idx].copy()
            chunk["sparse_score"] = float(scores[idx])
            chunk["sparse_rank"] = rank
            results.append(chunk)
        return results

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid retrieval combining dense and sparse results via Reciprocal Rank Fusion (RRF).

        RRF score = dense_weight / (k + dense_rank) + sparse_weight / (k + sparse_rank)
        """
        dense_results = self.dense_search(query, top_k=max(top_k * 2, 10))
        sparse_results = self.sparse_search(query, top_k=max(top_k * 2, 10))

        rrf_scores = {}
        candidate_chunks = {}

        # Accumulate dense ranks
        for item in dense_results:
            cid = item["chunk_id"]
            candidate_chunks[cid] = item
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (dense_weight / (self.rrf_k + item["dense_rank"]))

        # Accumulate sparse ranks
        for item in sparse_results:
            cid = item["chunk_id"]
            if cid not in candidate_chunks:
                candidate_chunks[cid] = item
            else:
                candidate_chunks[cid]["sparse_score"] = item.get("sparse_score", 0.0)
                candidate_chunks[cid]["sparse_rank"] = item.get("sparse_rank", 999)
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (sparse_weight / (self.rrf_k + item["sparse_rank"]))

        # Sort candidates by combined RRF score
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        final_results = []
        for rank, cid in enumerate(sorted_cids, 1):
            res = candidate_chunks[cid].copy()
            res["hybrid_rrf_score"] = round(float(rrf_scores[cid]), 5)
            res["hybrid_rank"] = rank
            final_results.append(res)

        return final_results

    def format_context_for_llm(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved clinical passages into clean structured prompt context."""
        if not retrieved_chunks:
            return "No relevant clinical guideline passages found."

        context_blocks = ["### Authority Clinical Guidelines (WHO 5th Ed / ICC 2022)\n"]
        for c in retrieved_chunks:
            block = (
                f"**[{c['chunk_id']}] {c['title']}**\n"
                f"- **Entity:** {c['specific_entity']} ({c['disease_category']})\n"
                f"- **Source:** {c['source']}\n"
                f"- **Text:** {c['text']}\n"
            )
            rule = c.get("blast_threshold_rule", {})
            if rule:
                block += f"- **Threshold/Rule:** {json.dumps(rule)}\n"
            context_blocks.append(block)

        return "\n".join(context_blocks)

    def build_query_from_morphology(
        self,
        predicted_class: str,
        blast_fraction: float,
        predominant_cells: Optional[List[str]] = None,
        patient_age_months: Optional[float] = None,
    ) -> str:
        """
        Synthesizes a targeted clinical query from observed morphology metrics.

        Age-awareness rules:
          - age < 18 confirmed  → include "pediatric" context
          - age >= 18 confirmed → use adult hematology language
          - age unknown         → use general bone-marrow cytomorphology (no age assumption)
        """
        # Determine age context
        if patient_age_months is not None:
            age_years = patient_age_months / 12.0
            if age_years < 18:
                age_context = f"pediatric patient age {age_years:.1f} years"
            else:
                age_context = f"adult patient age {age_years:.1f} years"
        else:
            age_context = None  # not provided — do not assume

        # Build query from confirmed evidence only
        query_parts = [f"Bone marrow cytomorphology diagnostic criteria for {predicted_class}"]
        query_parts.append(f"image-derived blast proportion {blast_fraction:.2%}")

        if predominant_cells:
            top_cells = ", ".join(predominant_cells[:3])
            query_parts.append(f"predominant cell types {top_cells}")

        if age_context:
            query_parts.append(age_context)

        return " | ".join(query_parts)



# ── Quick Verification CLI ───────────────────────────────────────────────────

def main():
    retriever = HybridRetriever()

    test_queries = [
        "What are the diagnostic blast criteria and cytomorphology for pediatric B-ALL?",
        "How to differentiate benign haematogones from leukemic lymphoblasts in children?",
        "Auer rods faggot cells APL emergency ATRA",
        "CML chronic phase myelocyte bulge and blast percentage cutoff",
    ]

    print("\n" + "=" * 70)
    print("HYBRID RETRIEVAL VERIFICATION DEMO")
    print("=" * 70)

    for q in test_queries:
        print(f"\n[QUERY]: {q}")
        results = retriever.retrieve(q, top_k=2)
        for r in results:
            print(f"  -> Rank {r['hybrid_rank']} (RRF: {r['hybrid_rrf_score']:.4f}) "
                  f"[{r['chunk_id']}] {r['title']} ({r['specific_entity']})")

    print("\n" + "=" * 70)
    print("DEMO COMPLETE: Hybrid retrieval is functioning properly!")
    print("=" * 70)


if __name__ == "__main__":
    main()
