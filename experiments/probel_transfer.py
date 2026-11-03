"""Frozen MiniLM and label-GCN transfer on the fixed ProBel news pilot.

Train and select settings in one language only; evaluate the same fitted model
in both languages. No target-language development data enters model selection.
"""
import hashlib
import json
import subprocess
from pathlib import Path

import joblib
from semeval.classify import predict, train
from semeval.data import check_splits, read_jsonl, write_json, write_jsonl
from semeval.evaluate import evaluate


def main():
    root, output = Path("data/public/probel"), Path("artifacts/public-probel-transfer")
    if (output / "summary.json").exists():
        raise ValueError("Preserve completed transfer run; choose a new output in the script for revised experiments")
    output.mkdir(parents=True, exist_ok=True)
    datasets = {split: read_jsonl(root / (split + ".jsonl")) for split in ("train", "dev", "test")}
    check_splits(*datasets.values(), duplicate_groups=json.loads((root / "duplicate-groups.json").read_text()))
    sources = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
        ("experiments/probel_transfer.py", "semeval/classify.py", "semeval/data.py", "semeval/features.py", "semeval/graph.py", "semeval/metrics.py", "semeval/evaluate.py")}
    inputs = {split: hashlib.sha256((root / (split + ".jsonl")).read_bytes()).hexdigest() for split in datasets}
    inputs["taxonomy"] = hashlib.sha256((root / "taxonomy.json").read_bytes()).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    reports = []
    for source in ("en", "ar"):
        training = [row for row in datasets["train"] if row["language"] == source]
        development = [row for row in datasets["dev"] if row["language"] == source]
        for method in ("transformer", "graph"):
            model = train(training, development, method, root / "taxonomy.json", seed=42, native=True, native_decision="multilabel")
            joblib.dump(model, output / f"{source}-{method}.joblib")
            for target in ("en", "ar"):
                evaluation = [row for row in datasets["test"] if row["language"] == target]
                predictions = predict(model, evaluation)
                name = f"{source}-to-{target}: {method}"
                report = evaluate(evaluation, predictions)
                report.update(name=name, seed=42, source_language=source, target_language=target,
                    fit_seconds=model["fit_seconds"],
                    scoring_provenance="Native 23-technique macro F1; source-language training/dev only; researcher article-grouped news pilot. NOT SemEval.")
                reports.append(report)
                write_jsonl(output / (name.replace(": ", "-") + ".jsonl"), predictions)
                print(name + " complete", flush=True)
    write_json(output / "summary.json", {"dataset": "ProBel news transfer pilot", "label_schema": "native", "runs": reports,
        "commit": commit, "input_sha256": inputs, "source_sha256": sources,
        "organizer_scorer_verified": False,
        "limitations": "Fixed small researcher article-grouped sample. One seed per transfer model, native persuasion techniques rather than narratives; EN/AR source and label distributions differ."})


if __name__ == "__main__":
    main()
