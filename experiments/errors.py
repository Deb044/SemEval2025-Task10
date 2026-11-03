"""Aggregate fixed-seed class supports, false labels and sibling confusions.

No article IDs, snippets or individual predictions are released. Diagnostic seed
42 is fixed in advance rather than selected for its test performance.
"""
import json
from collections import Counter
from pathlib import Path

from experiments.common import provenance
from semeval.data import read_jsonl, write_json
from semeval.evaluate import align

STUDIES = {
    "CARDS · extended hierarchy study": ("artifacts/extensions-cards", "data/public/cards/test.jsonl", "data/public/cards/train.jsonl", "data/public/cards/taxonomy.json"),
    "ProBel · extended bilingual study": ("artifacts/extensions-probel", "data/public/probel/test.jsonl", "data/public/probel/train.jsonl", "data/public/probel/taxonomy.json"),
    "CARDS → EMM source transfer": ("artifacts/extensions-emm", "data/public/cards/ood.jsonl", "data/public/cards/train.jsonl", "data/public/cards/taxonomy.json"),
}


def diagnose(gold, predictions, training, taxonomy):
    predictions = align(gold, predictions)
    training_support = Counter(label for row in training for label in row["labels"])
    counts = {label: {"training_support": training_support[label], "test_support": 0, "tp": 0, "fp": 0, "fn": 0} for label in taxonomy}
    confusions, siblings, false_positive_count = Counter(), 0, 0
    for row, prediction in zip(gold, predictions):
        actual, guessed = set(row["labels"]), set(prediction["labels"])
        for label in taxonomy:
            stats = counts[label]
            stats["test_support"] += label in actual
            stats["tp"] += label in actual & guessed
            stats["fp"] += label in guessed - actual
            stats["fn"] += label in actual - guessed
        parents = {taxonomy[label]["parent"] for label in actual}
        for wrong in guessed - actual:
            false_positive_count += 1
            siblings += taxonomy[wrong]["parent"] in parents
            for missed in actual - guessed:
                confusions[(missed, wrong)] += 1
    for stats in counts.values():
        denominator = 2 * stats["tp"] + stats["fp"] + stats["fn"]
        stats["f1"] = 2 * stats["tp"] / denominator if denominator else 0.
    rare = [stats["f1"] for stats in counts.values() if stats["training_support"] <= 10]
    common = [stats["f1"] for stats in counts.values() if stats["training_support"] > 10]
    return {"count": len(gold), "per_class": counts,
        "training_rare_threshold": 10, "rare_class_count": len(rare),
        "rare_class_macro_f1": sum(rare) / len(rare) if rare else None,
        "other_class_macro_f1": sum(common) / len(common) if common else None,
        "false_positive_label_count": false_positive_count,
        "sibling_false_positive_fraction": siblings / false_positive_count if false_positive_count else None,
        "top_confusions": [{"missed": pair[0], "predicted": pair[1], "pair_count": count} for pair, count in confusions.most_common(5)]}


def main():
    records = {}
    for study, (folder, test, train, taxonomy_path) in STUDIES.items():
        gold, training = read_jsonl(test), read_jsonl(train)
        taxonomy = json.loads(Path(taxonomy_path).read_text())
        records[study] = {}
        for prediction_path in sorted(Path(folder).glob("*-42.jsonl")):
            records[study][prediction_path.stem] = diagnose(gold, read_jsonl(prediction_path), training, taxonomy)
    write_json("results/error-analysis.json", {"seed": 42, "studies": records,
        "note": "Descriptive diagnostics at fixed seed 42; rare means <=10 training positives. Confusion pairs count each missed/extra-label combination, not disjoint article errors."})
    print("Exported aggregate native-label diagnostics")


if __name__ == "__main__":
    main()
