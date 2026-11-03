"""Long CARDS passages: evidence selection and supplied/predicted claim labels.

Native single-label climate claims are not the SemEval narrative taxonomy. No
human explanation references exist here, so only format/retrieval statistics are
automated; a blinded local review queue supports real human grounding judgments.
"""
import argparse
import hashlib
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import joblib
from experiments.common import provenance, sha256
from semeval.classify import predict
from semeval.data import read_jsonl, write_json, write_jsonl
from semeval.explain import explain_row, sentences

VARIANTS = {
    "textrank": ("textrank", "semantic", False),
    "semantic-extract": ("semantic", "semantic", False),
    "supplied-semantic": ("generate", "semantic", False),
    "supplied-lead": ("generate", "lead", False),
    "predicted-semantic": ("generate", "semantic", True),
}


def fixed_subset(rows, limit=24):
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda value: value["article_id"]):
        if row["labels"] != ["0_0"] and len(sentences(row["text"])) >= 8:
            groups[row["narratives"][0]].append(row)
    rng = random.Random(42)
    for key in sorted(groups):
        rng.shuffle(groups[key])
    chosen = []
    while any(groups.values()) and len(chosen) < limit:
        for key in sorted(groups):
            if groups[key] and len(chosen) < limit:
                chosen.append(groups[key].pop())
    if not chosen:
        raise ValueError("No passages meet the fixed length criterion")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--model", default="artifacts/public-cards-v2/graph-none-42.joblib")
    parser.add_argument("--out", default="artifacts/extensions-long-explanations")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    args = parser.parse_args()
    output = Path(args.out)
    rows = fixed_subset(read_jsonl("data/public/cards/test.jsonl"), args.limit)
    taxonomy = json.loads(Path("data/public/cards/taxonomy.json").read_text())
    meta = provenance(["experiments/long_explanations.py", "semeval/explain.py", "semeval/classify.py", "semeval/features.py"],
        {"evaluation": "data/public/cards/test.jsonl", "taxonomy": "data/public/cards/taxonomy.json", "classifier": args.model, "generator": args.gguf})
    signature = hashlib.sha256(json.dumps({"sources": meta["source_sha256"], "inputs": meta["input_sha256"], "limit": args.limit}, sort_keys=True).encode()).hexdigest()
    model = joblib.load(args.model)
    predictions = {row["article_id"]: row for row in predict(model, rows)}
    prepared = [dict(row, task=3, dominant_subnarrative=row["labels"][0], dominant_narrative=row["narratives"][0]) for row in rows]
    write_jsonl(output / "evaluation.jsonl", prepared)
    outputs, reports = {}, []
    for name, (method, selection, predicted) in VARIANTS.items():
        path = output / (name + ".jsonl")
        values = read_jsonl(path) if path.exists() else []
        seen = {value["article_id"] for value in values}
        if len(seen) != len(values) or not seen <= {row["article_id"] for row in rows} or any(value.get("signature") != signature for value in values):
            raise ValueError("Existing long-explanation checkpoint has different inputs/source or invalid IDs")
        for row in prepared:
            if row["article_id"] in seen:
                continue
            started = time.perf_counter()
            value = explain_row(row, method, taxonomy, args.endpoint, selection,
                                predictions[row["article_id"]] if predicted else None)
            value.update(variant=name, signature=signature, seconds=time.perf_counter() - started,
                         dataset="CARDS long-text explanations", label_schema="native")
            values.append(value)
            write_jsonl(path, values)
        lookup = {value["article_id"]: value for value in values}
        values = [lookup[row["article_id"]] for row in prepared]
        outputs[name] = values
        metrics = {"count": len(values), "length_compliance": sum(0 < value["words"] <= 80 for value in values) / len(values),
            "fallback_rate": sum(value["fallback"] for value in values) / len(values),
            "mean_words": sum(value["words"] for value in values) / len(values)}
        reports.append({"name": name, "task": 3, "primary_metric": "human_grounding_rate", "metrics": metrics,
            "status": "awaiting_human_review", "generation_seconds": sum(value["seconds"] for value in values),
            "scoring_provenance": "Format statistics only; no reference explanations or human quality ratings supplied."})
        print(name + " complete", flush=True)
    same = sum(a["evidence_ids"] == b["evidence_ids"] for a, b in zip(outputs["supplied-semantic"], outputs["supplied-lead"])) / len(rows)
    target_accuracy = sum(predictions[row["article_id"]]["labels"] == row["labels"] for row in rows) / len(rows)
    review = []
    # Six source passages, every method, stable blinded IDs. No scores are prefilled.
    for row in prepared[:6]:
        for variant, values in outputs.items():
            value = next(value for value in values if value["article_id"] == row["article_id"])
            review.append({"review_id": hashlib.sha256((row["article_id"] + variant).encode()).hexdigest()[:16],
                "variant": variant, "article_id": row["article_id"], "language": row["language"], "text": row["text"],
                "target": value["dominant_subnarrative"], "target_description": taxonomy[value["dominant_subnarrative"]]["definition"],
                "explanation": value["explanation"], "evidence_ids": value["evidence_ids"]})
    random.Random(42).shuffle(review)
    write_jsonl(output / "review-queue.jsonl", review)
    write_json(output / "summary.json", {**meta, "dataset": "CARDS long-text explanations", "label_schema": "native",
        "experiment_id": "long-evidence-and-predicted-labels", "runs": reports,
        "statistics": {"identical_evidence_rate": same, "predicted_claim_label_accuracy": target_accuracy,
                       "review_items": len(review), "source_passages": len(rows)},
        "limitations": "English long paragraphs, not full articles; native climate claims, not SemEval narratives. Format compliance is not grounding. Human quality evaluation is pending real reviewer input."})


if __name__ == "__main__":
    main()
