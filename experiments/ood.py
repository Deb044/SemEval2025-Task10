"""Apply fixed CARDS models to held-out EMM snippets, without retraining."""
import argparse
import json
import time
from pathlib import Path

import joblib
from experiments.common import provenance, sha256
from semeval.classify import predict
from semeval.data import check_splits, read_jsonl, write_json, write_jsonl
from semeval.evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="artifacts/extensions-cards")
    parser.add_argument("--out", default="artifacts/extensions-emm")
    args = parser.parse_args()
    folder, output = Path(args.models), Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preserve completed evaluation; choose a fresh output directory")
    paths = {name: Path("data/public/cards") / (name + ".jsonl") for name in ("train", "dev", "test", "ood")}
    rows = {name: read_jsonl(path) for name, path in paths.items()}
    check_splits(*rows.values())
    meta = provenance(["experiments/ood.py", "semeval/classify.py", "semeval/features.py", "semeval/graph.py", "semeval/evaluate.py", "semeval/metrics.py"], paths)
    files = sorted(folder.glob("*.joblib"))
    if not files or not (folder / "summary.json").exists():
        raise ValueError("Complete the classification matrix first")
    original = json.loads((folder / "summary.json").read_text())
    reports = []
    for path in files:
        model = joblib.load(path)
        for split in ("train", "dev"):
            if model.get("data_sha256", {}).get(split) != sha256(paths[split]):
                raise ValueError("Model training provenance does not match the audited splits")
        started = time.perf_counter()
        predictions = predict(model, rows["ood"])
        seconds = time.perf_counter() - started
        report = evaluate(rows["ood"], predictions)
        report.update(name=path.stem, method=model["method"], ablation=model["ablation"], seed=model["seed"],
            predict_seconds=seconds, source_model_sha256=sha256(path),
            scoring_provenance="Fixed CARDS model transferred to native EMM climate claims; no EMM training or tuning; not SemEval.")
        reports.append(report)
        write_jsonl(output / (path.stem + ".jsonl"), predictions)
        print(path.stem + " evaluated on EMM", flush=True)
    write_json(output / "summary.json", {**meta, "dataset": "EMM climate snippets", "experiment_id": "cards-to-emm",
        "label_schema": "native", "runs": reports, "training_commit": original["commit"],
        "organizer_scorer_verified": False, "limitations": "English snippet domain transfer; 18 native CARDS labels; exact-duplicate audits do not establish absence of near duplicates."})


if __name__ == "__main__":
    main()
