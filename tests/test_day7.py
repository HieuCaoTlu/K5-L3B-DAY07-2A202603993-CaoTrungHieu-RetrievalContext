"""Test offline cho code Day 7 (không gọi pgvector hay Gemini; benchmark thật chạy bằng src/run_agent.py)."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag import config  # noqa: E402
from rag.agent import AgentRun, _clean_query, _finish  # noqa: E402
from rag.context import _block, pack  # noqa: E402
from rag.retrieval import BM25Retriever, check_lock, chunk_by_id, normalize, tokenize  # noqa: E402


class BenchmarkTest(unittest.TestCase):
    def test_lock_matches(self):
        check_lock()   # raise SystemExit nếu benchmark/corpus bị sửa sau khi khoá

    def test_split_sizes_and_gold_in_chunk(self):
        cases = json.loads(config.CASES_PATH.read_text(encoding="utf-8"))
        qs = json.loads(config.QUESTIONS_PATH.read_text(encoding="utf-8"))
        count = lambda s, a: sum(c["split"] == s and c["answerable"] == a for c in cases)  # noqa: E731
        self.assertGreaterEqual(count("dev", True), 15)
        self.assertGreaterEqual(count("dev", False), 2)
        self.assertGreaterEqual(count("test", True), 5)
        self.assertGreaterEqual(count("test", False), 2)
        chunks = chunk_by_id().values()
        for q in qs:
            for g in q["gold"]:
                self.assertTrue(any(c["page"] == g["page"] and g["evidence"] in c["text"] for c in chunks), q["id"])


class RetrievalTest(unittest.TestCase):
    def test_tokenize_joins_ocr_hyphen_and_drops_stopwords(self):
        self.assertEqual(normalize("ani-\n\nmals"), "animals")
        # stem nhẹ: chỉ bỏ 's' cuối khi từ > 4 ký tự (cats giữ nguyên; leopards -> leopard; Andes -> ande)
        self.assertEqual(tokenize("The cats and leopards of the Andes"), ["cats", "leopard", "ande"])

    def test_bm25_finds_exact_fact_and_respects_filter(self):
        bm25 = BM25Retriever()
        res = bm25.search("Wetland destruction primary threat fishing cat", top_k=3)
        self.assertEqual(res[0]["page"], 101)
        self.assertEqual([r["rank"] for r in res], [1, 2, 3])
        filtered = bm25.search("leopard colour", filters={"region": "Sub-Saharan Africa"}, top_k=5)
        self.assertTrue(filtered and all(r["metadata"]["region"] == "Sub-Saharan Africa" for r in filtered))


class ContextTest(unittest.TestCase):
    def test_pack_respects_budget_and_dedups(self):
        chunks = list(chunk_by_id().values())[:5]
        ctx, ids = pack(chunks + chunks, budget=1500)
        self.assertLessEqual(len(ctx), 1500)
        self.assertEqual(len(ids), len(set(ids)))

    def test_header_without_species(self):
        c = chunk_by_id()["iucn-wildcats-1996:p101:c03"]
        self.assertIn("section:", _block(c))
        self.assertNotIn("section:", _block(c, header="no_species"))
        self.assertIn("page 101", _block(c, header="no_species"))


class AgentParsingTest(unittest.TestCase):
    def test_valid_citation_answered(self):
        run = AgentRun(query_id="x")
        _finish(run, "Adult males weigh 11-14 kg [S1].", {"chunk-a": "S1"})
        self.assertEqual((run.status, run.citations), ("answered", ["chunk-a"]))

    def test_unknown_label_means_abstain(self):
        run = AgentRun(query_id="x")
        _finish(run, "Something [S9].", {"chunk-a": "S1"})
        self.assertEqual(run.status, "abstain")
        self.assertEqual(run.invalid_citations, ["S9"])

    def test_explicit_abstain(self):
        run = AgentRun(query_id="x")
        _finish(run, "INSUFFICIENT EVIDENCE: not stated [S1].", {"chunk-a": "S1"})
        self.assertEqual((run.status, run.citations), ("abstain", []))

    def test_grouped_labels_and_clean_query(self):
        run = AgentRun(query_id="x")
        _finish(run, "Three species [S1, S2].", {"a": "S1", "b": "S2"})
        self.assertEqual(run.citations, ["a", "b"])
        self.assertEqual(_clean_query("jaguar speed [S1]"), "jaguar speed")


if __name__ == "__main__":
    unittest.main()
