"""Small correctness checks using invented text, never benchmark measurements."""
import json
import importlib.util
import tempfile
import unittest
from pathlib import Path

from semeval.data import check_splits, import_official
from semeval.evaluate import align, export_tsv
from semeval.explain import extract, rank_evidence, select_context
from semeval.metrics import set_f1


class PipelineChecks(unittest.TestCase):
    def test_unicode_offsets_and_release_convention(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "HI_1.txt").write_text("शीर्षक\n\nमीरा helps.", encoding="utf-8", newline="")
            text = (root / "HI_1.txt").read_text(encoding="utf-8")
            start = text.index("मीरा")
            end = start + len("मीरा") - 1
            (root / "gold.tsv").write_text(f"HI_1.txt\tमीरा\t{start}\t{end}\tProtagonist\tGuardian\n", encoding="utf-8")
            rows = import_official(root, root / "gold.tsv", 1, "inclusive")
            self.assertEqual(rows[0]["text"][start:rows[0]["end_exclusive"]], "मीरा")
            with self.assertRaises(ValueError):
                import_official(root, root / "gold.tsv", 1, "exclusive")

    def test_cross_task_duplicate_leakage(self):
        first = {"task": 2, "article_id": "EN_1.txt", "language": "en", "text": "Invented article."}
        second = dict(first, task=3, article_id="EN_2.txt", text="Invented  article.")
        with self.assertRaises(ValueError):
            check_splits([first], [second])

    def test_alignment_and_document_f1(self):
        rows = [{"task": 2, "article_id": name} for name in ("a", "b")]
        self.assertEqual(align(rows, rows[::-1]), rows)
        with self.assertRaises(ValueError):
            align(rows, rows[:1])
        self.assertAlmostEqual(set_f1(["x", "y"], ["y", "z"]), .5)

    def test_extractive_evidence_and_tsv(self):
        class WordTokenizer:
            def encode(self, text, **kwargs):
                return text.split()
        units = [{"id": index, "text": "unit"} for index in (2, 0, 1, 3, 4)]
        contextual = select_context(units, WordTokenizer(), max_tokens=3, max_sentences=3)
        self.assertEqual([unit["id"] for unit in contextual], [1, 2, 3])
        ranked = rank_evidence("Invented title\n\nMira claims taxes hurt businesses. Others dispute that claim.", "", "textrank")
        text, ids = extract(ranked, 10)
        self.assertLessEqual(len(text.split()), 10)
        self.assertTrue(ids)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "prediction.tsv"
            export_tsv([{"task": 3, "article_id": "EN_1.txt", "explanation": text}], path)
            self.assertEqual(len(path.read_text(encoding="utf-8").strip().split("\t")), 2)

    def test_lexical_and_lsa_training(self):
        from semeval.classify import predict, train
        from semeval.evaluate import evaluate
        def rows(prefix, count):
            return [{"task": 2, "article_id": f"EN_{prefix}{i}.txt", "language": "en",
                     "text": f"Invented {prefix} article {i}. " + ("Policies cost money." if i % 2 else "Energy costs money."),
                     "labels": ["cost" if i % 2 else "energy"], "narratives": ["policy"]} for i in range(count)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "taxonomy.json"
            path.write_text(json.dumps({label: {"parent": "Other" if label == "Other" else "policy", "definition": label} for label in ("cost", "energy", "Other")}), encoding="utf-8")
            for method in ("tfidf", "lsa"):
                model = train(rows("train", 8), rows("dev", 4), method, path)
                output = predict(model, rows("test", 4))
                report = evaluate(rows("test", 4), output)
                self.assertEqual(report["metrics"]["count"], 4)
                self.assertFalse(report["benchmark_claim"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional model dependencies are not installed")
    def test_graph_shape_and_gradient(self):
        import torch
        from semeval.graph import LabelGCN
        model = LabelGCN(8, torch.randn(4, 8), torch.eye(4), torch.arange(2))
        logits = model(torch.randn(3, 8))
        self.assertEqual(tuple(logits.shape), (3, 2))
        logits.sum().backward()
        self.assertIsNotNone(model.layer1.weight.grad)


if __name__ == "__main__":
    unittest.main()
