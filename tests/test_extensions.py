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
            with self.assertRaises(ValueError):
                summarize_reviews([dict(queue[0], explanation="changed output")], ratings)

    def test_model_audit_is_not_human_validation_and_exports_only_aggregates(self):
        from experiments.grounding import llm_summary
        queue = [{"review_id": "fixture", "variant": "fixture-method", "text": "PRIVATE_TEST_SENTINEL. Second sentence."}]
        rating = {"review_id": "fixture", "supported": 2, "attribution": 2, "relevance": 1,
                  "readability": 1, "evidence_ids": [0], "reviewer_type": "llm", "judge_model": "test-only",
                  "rationale": "PRIVATE_RATIONALE_SENTINEL"}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ratings.jsonl"
            write_jsonl(path, [rating])
            summary = llm_summary(queue, path)
            self.assertEqual(summary["runs"][0]["metrics"]["count"], 1)
            self.assertNotIn("PRIVATE_", json.dumps(summary))
            write_jsonl(path, [dict(rating, reviewer_type="human")])
            with self.assertRaises(ValueError):
                llm_summary(queue, path)

    def test_error_diagnostics_count_sibling_false_labels_without_text(self):
        from experiments.errors import diagnose
        gold = [{"task": 2, "article_id": "private-id-one", "labels": ["a"]},
                {"task": 2, "article_id": "private-id-two", "labels": ["c"]}]
        predicted = [dict(gold[0], labels=["b"]), dict(gold[1])]
        taxonomy = {"a": {"parent": "one"}, "b": {"parent": "one"}, "c": {"parent": "two"}}
        result = diagnose(gold, predicted, gold, taxonomy)
        self.assertEqual(result["per_class"]["a"]["fn"], 1)
        self.assertEqual(result["per_class"]["b"]["fp"], 1)
        self.assertEqual(result["per_class"]["c"]["f1"], 1.)
        self.assertEqual(result["sibling_false_positive_fraction"], 1.)
        self.assertNotIn("private-id", json.dumps(result))

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
