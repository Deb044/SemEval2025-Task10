"""Focused extension checks; invented fixtures are never experiment results."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from semeval.data import write_jsonl
from semeval.review import save_review, summarize_reviews


class ExtensionChecks(unittest.TestCase):
    def test_review_requires_real_explicit_submission_and_keeps_methods_blinded(self):
        with tempfile.TemporaryDirectory() as directory:
            queue_path, ratings_path = Path(directory) / "queue.jsonl", Path(directory) / "ratings.jsonl"
            queue = [{"review_id": "test", "variant": "hidden-method"}]
            write_jsonl(queue_path, queue)
            payload = {"review_id": "test", "reviewer": "test-only", "supported": 2, "attribution": 2, "relevance": 1, "readability": 1}
            with self.assertRaises(ValueError):
                save_review(payload, queue_path, ratings_path)
            self.assertFalse(ratings_path.exists())
            payload.update(human_reviewer=True, language_confident=True)
            save_review(payload, queue_path, ratings_path)
            save_review(dict(payload, supported=0), queue_path, ratings_path)
            ratings = [json.loads(line) for line in ratings_path.read_text().splitlines()]
            self.assertEqual(len(ratings), 1)
            summary = summarize_reviews(queue, ratings)
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["runs"][0]["metrics"]["human_grounding_rate"], 0.)
            self.assertNotIn("test-only", json.dumps(summary))
            self.assertEqual(summarize_reviews(queue, [])["runs"], [])

    def test_local_api_rejects_remote_origins_and_hosts(self):
        from types import SimpleNamespace
        from semeval.web import Handler
        self.assertTrue(Handler.local_request(SimpleNamespace(headers={"Host": "127.0.0.1:8765", "Origin": "http://127.0.0.1:8765"})))
        self.assertFalse(Handler.local_request(SimpleNamespace(headers={"Host": "attacker.example", "Origin": "http://127.0.0.1:8765"})))
        self.assertFalse(Handler.local_request(SimpleNamespace(headers={"Host": "127.0.0.1:8765", "Origin": "https://attacker.example"})))

    @unittest.skipUnless(importlib.util.find_spec("duckdb"), "Optional analytics dependencies missing")
    def test_parquet_preserves_real_query_semantics(self):
        from experiments.scalability import worker
        with tempfile.TemporaryDirectory() as directory:
            source, parquet = Path(directory) / "input.jsonl", Path(directory) / "input.parquet"
            source.write_text("\n".join(json.dumps(row) for row in [
                {"language": "english", "text": "Invented text", "propagandistic": False},
                {"language": "arabic", "text": "Invented other text", "propagandistic": True}]) + "\n")
            worker("convert", str(source), str(parquet))
            first = worker("json", str(source), directory)
            second = worker("parquet", str(parquet), directory)
            self.assertEqual(first["count"], 2)
            self.assertEqual(first["counts_by_language"], second["counts_by_language"])


if __name__ == "__main__":
    unittest.main()
