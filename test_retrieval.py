"""
Verification script for BSM Hospital RAG Retrieval System.
Tests:
1. RetrievalEngine initialization & recursive chunking
2. BM25 sparse search
3. Dense semantic search
4. Reciprocal Rank Fusion (RRF)
5. Weighted Combination Fusion
6. FastAPI endpoints via TestClient
"""
import unittest
from fastapi.testclient import TestClient

from app import app, engine, ensure_engine_ready


class TestRetrievalSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ensure_engine_ready()
        cls.engine = engine
        cls.client = TestClient(app)

    def test_chunking_and_stats(self):
        stats = self.engine.get_stats()
        self.assertTrue(stats["is_ready"])
        self.assertEqual(stats["total_pages"], 15)
        self.assertGreater(stats["total_chunks"], 50)
        print(f"\n[PASS] Index created: {stats['total_chunks']} chunks from {stats['total_pages']} pages.")

    def test_bm25_search(self):
        query = "mislabeled specimen"
        results = self.engine.search_bm25(query=query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertIn("specimen", results[0]["content"].lower())
        print(f"[PASS] BM25 retrieved: Rank #1 Page {results[0]['page']} (Score: {results[0]['score']})")

    def test_dense_search(self):
        query = "emergency room triage order and urgent cases"
        results = self.engine.search_dense(query=query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertGreater(results[0]["score"], 0.4)
        print(f"[PASS] Dense retrieved: Rank #1 Page {results[0]['page']} (Sim: {results[0]['score']})")

    def test_rrf_fusion(self):
        query = "pharmacy operations and prescription dispensing"
        results = self.engine.search_rrf(query=query, top_k=3, rrf_k=60)
        self.assertGreater(len(results), 0)
        self.assertIn("bm25_rank", results[0])
        self.assertIn("dense_rank", results[0])
        print(f"[PASS] RRF retrieved: Rank #1 (RRF Score: {results[0]['score']}, BM25 #{results[0]['bm25_rank']}, Dense #{results[0]['dense_rank']})")

    def test_weighted_fusion(self):
        query = "cancellation and late arrival guidance for clinic appointments"
        results = self.engine.search_weighted(query=query, top_k=3, alpha=0.6)
        self.assertGreater(len(results), 0)
        self.assertIn("dense_norm", results[0])
        self.assertIn("bm25_norm", results[0])
        print(f"[PASS] Weighted retrieved: Rank #1 (Weighted Score: {results[0]['score']}, Alpha: {results[0]['alpha']})")

    def test_api_search_endpoint(self):
        payload = {
            "query": "billing dispute and refund procedure",
            "mode": "weighted",
            "top_k": 3,
            "alpha": 0.5,
            "rrf_k": 60
        }
        res = self.client.post("/api/search", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["mode"], "weighted")
        self.assertEqual(len(data["results"]), 3)
        print(f"[PASS] API Search returned 200 OK in {data['latency_ms']} ms.")

    def test_api_compare_endpoint(self):
        payload = {
            "query": "hospital incident reporting procedure",
            "mode": "compare",
            "top_k": 3,
            "alpha": 0.5,
            "rrf_k": 60
        }
        res = self.client.post("/api/search", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("comparison", data)
        self.assertIn("bm25", data["comparison"])
        self.assertIn("dense", data["comparison"])
        self.assertIn("rrf", data["comparison"])
        self.assertIn("weighted", data["comparison"])
        print(f"[PASS] API Compare returned all 4 modes in {data['latency_ms']} ms.")

    def test_api_stats_endpoint(self):
        res = self.client.get("/api/stats")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["document"], "BSM_Hospital_RAG_Knowledge_Base.pdf")
        self.assertEqual(data["total_pages"], 15)
        print(f"[PASS] API Stats endpoint returned valid metadata.")

    def test_api_examples_endpoint(self):
        res = self.client.get("/api/examples")
        self.assertEqual(res.status_code, 200)
        examples = res.json()
        self.assertGreaterEqual(len(examples), 5)
        print(f"[PASS] API Examples returned {len(examples)} curated sample queries.")


if __name__ == "__main__":
    unittest.main()
