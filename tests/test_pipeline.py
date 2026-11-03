"""Small correctness checks using invented text, never benchmark measurements."""
import json
import importlib.util
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from semeval.data import build_taxonomy, check_splits, import_official
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
        check_splits([first, dict(first, task=3)], [dict(first, article_id="EN_3.txt", text="Different article.")])
        translated = dict(first, article_id="HI_4.txt", language="hi", text="अलग भाषा में लेख।")
        with self.assertRaises(ValueError):
            check_splits([first], [translated], duplicate_groups={"EN_1.txt": "translation-a", "HI_4.txt": "translation-a"})
        with self.assertRaises(ValueError):
            check_splits([first, dict(first, task=3, text="Conflicting article body.")])

    def test_alignment_and_document_f1(self):
        rows = [{"task": 2, "article_id": name} for name in ("a", "b")]
        self.assertEqual(align(rows, rows[::-1]), rows)
        with self.assertRaises(ValueError):
            align(rows, rows[:1])
        with self.assertRaises(ValueError):
            align([rows[0], rows[0]], [rows[0]])
        self.assertAlmostEqual(set_f1(["x", "y"], ["y", "z"]), .5)

    def test_bertscore_hash_return_contract(self):
        import numpy as np
        from types import SimpleNamespace
        from semeval.evaluate import evaluate
        values = np.array([.7, .8, .9])
        scorer = SimpleNamespace(score=lambda *args, **kwargs: ((values, values, values), "unit-test-hash"))
        gold = [{"task": 3, "article_id": str(i), "language": "en", "explanation": "reference"} for i in range(3)]
        predictions = [dict(row, explanation="generated") for row in gold]
        with patch.dict("sys.modules", {"bert_score": scorer}):
            report = evaluate(gold, predictions, "unit-test-model", 9)
            self.assertAlmostEqual(report["metrics"]["bertscore_f1"], .8)
            from experiments.public_explanations import evaluate as evaluate_native
            native_gold = [{"label": label, "references": ["reference"]} for label in ("entailment", "neutral", "contradiction")]
            native = evaluate_native(native_gold, [dict(row, fallback=False) for row in predictions], "unit-test-model", 9)
            self.assertAlmostEqual(native["bertscore_f1"], .8)

    def test_extractive_evidence_and_tsv(self):
        class WordTokenizer:
            def encode(self, text, **kwargs):
                return text.split()
        units = [{"id": index, "text": "unit", "relevance": 1. / (index + 1)} for index in (2, 0, 1, 3, 4)]
        contextual = select_context(units, WordTokenizer(), max_tokens=3, max_sentences=3)
        self.assertEqual([unit["id"] for unit in contextual], [1, 2, 3])
        self.assertTrue(all(set(unit) == {"id", "text"} for unit in contextual))
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

    def test_release_taxonomy_mapping(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            coarse, fine = root / "coarse.txt", root / "fine.txt"
            coarse.write_text("Other\nCC: Policy\n", encoding="utf-8")
            fine.write_text("Other\nCC: Policy: Cost\n", encoding="utf-8")
            taxonomy = build_taxonomy(coarse, fine)
            self.assertEqual(taxonomy["CC: Policy: Cost"]["parent"], "CC: Policy")
            self.assertEqual(taxonomy["CC: Policy: Cost"]["definition_source"], "label_text_only")
            fine.write_text("Other\nCC: Unknown: Cost\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                build_taxonomy(coarse, fine)

    def test_native_labels_do_not_become_semeval_roles(self):
        from semeval.classify import decode
        import numpy as np
        self.assertEqual(decode(np.array([[.2, .8]]), ["positive", "negative"], .5, 1, "single"), [["negative"]])
        row = {"task": 1, "article_id": "EN_native.txt", "language": "en", "text": "Good food.",
               "mention": "food", "start": 5, "end": 9, "end_exclusive": 9,
               "labels": ["positive"], "label_schema": "native"}
        from semeval.data import validate
        validate([row])
        with self.assertRaises(ValueError):
            validate([dict(row, label_schema="semeval")])
        from experiments.publish import aggregate
        summary = {"dataset": "YASO clear-text subset", "label_schema": "native", "commit": "test-only",
                   "input_sha256": {"test": "test-only"}, "runs": [{"task": 1, "name": "test-only",
                   "primary_metric": "macro_f1", "metrics": {"macro_f1": .5, "count": 2}, "scoring_provenance": "test-only"}]}
        self.assertEqual(aggregate([summary])[0]["dataset"], "YASO clear-text subset")
        with self.assertRaises(ValueError):
            aggregate([dict(summary, dataset="SemEval-2025 Task 10")])
        with self.assertRaises(ValueError):
            aggregate([summary, summary])
        changed_split = dict(summary, input_sha256={"test": "different"})
        with self.assertRaises(ValueError):
            aggregate([summary, changed_split])
        invalid_run = dict(summary["runs"][0], metrics={"macro_f1": float("nan"), "count": 2})
        with self.assertRaises(ValueError):
            aggregate([dict(summary, runs=[invalid_run])])

    def test_experiment_resume_checks_inputs_and_reuses_completed_work(self):
        from experiments.run import run
        from semeval.data import write_json, write_jsonl
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {"dataset": "unit-test-only", "synthetic": True, "label_schema": "native", "output": str(root / "run")}
            rows = {}
            for split in ("train", "dev", "test"):
                text = split + " food example."
                start = text.index("food")
                rows[split] = [{"task": 1, "article_id": f"EN_{split}.txt", "language": "en", "text": text,
                    "mention": "food", "start": start, "end": start + 4, "end_exclusive": start + 4,
                    "labels": ["positive"], "label_schema": "native"}]
                config[split] = str(root / (split + ".jsonl"))
                write_jsonl(config[split], rows[split])
            path = root / "config.json"
            write_json(path, config)
            def fake_train(*args, **kwargs):
                return {"fit_seconds": 1., "threshold": .5, "regularization": 1.}
            def fake_predict(model, records):
                return [dict(row, scores={"positive": 1.}) for row in records]
            with patch("experiments.run.train", side_effect=fake_train) as training, patch("experiments.run.predict", side_effect=fake_predict) as prediction:
                run(path)
                run(path, resume=True)
                self.assertEqual(training.call_count, 3)
                self.assertEqual(prediction.call_count, 3)
                write_jsonl(config["dev"], [dict(rows["dev"][0], provenance="changed")])
                with self.assertRaises(ValueError):
                    run(path, resume=True)

    def test_native_multilabel_preserves_empty_sets_and_language_boundary(self):
        import numpy as np
        from semeval.classify import decode
        from semeval.data import validate
        from semeval.metrics import classification_scores
        output = decode(np.array([[.8, .7], [.1, .2]]), ["A", "B"], .5, 2, "native-multilabel")
        self.assertEqual(output, [["A", "B"], []])
        row = {"task": 2, "article_id": "AR_native.txt", "language": "ar", "text": "Invented text.",
               "labels": [], "narratives": [], "label_schema": "native"}
        validate([row])
        with self.assertRaises(ValueError):
            validate([dict(row, label_schema="semeval")])
        score = classification_scores([dict(row, labels=["A"]), dict(row, article_id="AR_two.txt")], [["A"], []], classes=["A", "B"])
        self.assertEqual(score["macro_f1"], .5)
        self.assertEqual(score["by_language"]["ar"]["macro_f1"], .5)

    def test_probel_grouping_keeps_source_articles_and_duplicates_together(self):
        from experiments.public_probel import group_split
        from semeval.data import fingerprint
        rows = [{"article_id": str(i), "source_group": str(i // 2),
                 "text": "Invented article " + str(i // 2) + " sentence " + str(i % 2)} for i in range(80)]
        rows.append(dict(rows[0], article_id="duplicate", source_group="duplicate-source"))
        partitions = group_split(rows)
        owners = {}
        for split, partition in partitions.items():
            for row in partition:
                for key in (row["source_group"], fingerprint(row["text"])):
                    self.assertIn(owners.setdefault(key, split), [split])

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
