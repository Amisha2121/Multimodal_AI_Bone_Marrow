"""
tests/test_retrieval.py
─────────────────────────────────────────────────────────────────────────────
Unit and Integration Tests for the Pediatric Leukemia Medical RAG Engine.
"""

import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from retrieval.ingest import ingest_corpus, DEFAULT_KB_FILE, OUTPUT_CHUNKS_FILE
from retrieval.index import build_indices, CHUNKS_FILE, FAISS_PATH, BM25_PATH
from retrieval.retrieve import HybridRetriever


class TestRetrievalPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Ensure indices exist
        if not FAISS_PATH.exists() or not BM25_PATH.exists():
            build_indices()
        cls.retriever = HybridRetriever()

    def test_corpus_ingestion(self):
        chunks = ingest_corpus()
        self.assertGreaterEqual(len(chunks), 10)
        first_chunk = chunks[0]
        self.assertIn("chunk_id", first_chunk)
        self.assertIn("text", first_chunk)
        self.assertIn("embedding_text", first_chunk)

    def test_dense_search(self):
        query = "pediatric B-ALL diagnostic criteria and lymphoblast morphology"
        results = self.retriever.dense_search(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertIn("dense_score", results[0])
        self.assertGreater(results[0]["dense_score"], 0.4)

    def test_sparse_bm25_search(self):
        query = "Auer rods faggot cells PML-RARA"
        results = self.retriever.sparse_search(query, top_k=3)
        self.assertGreater(len(results), 0)
        top_ids = [r["chunk_id"] for r in results]
        self.assertIn("WHO2022_AML_SUB_005", top_ids)

    def test_hybrid_rrf_retrieval(self):
        query = "CML chronic phase myelocyte bulge"
        results = self.retriever.retrieve(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["chunk_id"], "WHO2022_CML_CP_007")
        self.assertIn("hybrid_rrf_score", results[0])


if __name__ == "__main__":
    unittest.main()
