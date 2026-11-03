"""Transparent local metrics. Official parity requires the organizer scorer."""
import numpy as np


def set_f1(gold, predicted):
    gold, predicted = set(gold), set(predicted)
    return 2 * len(gold & predicted) / (len(gold) + len(predicted)) if gold or predicted else 0.0


def classification_scores(rows, predictions, classes=None):
    exact = [set(row["labels"]) == set(pred) for row, pred in zip(rows, predictions, strict=True)]
    f1 = [set_f1(row["labels"], pred) for row, pred in zip(rows, predictions, strict=True)]
    languages = sorted({row["language"] for row in rows})
    gold_sets, predicted_sets = [set(row["labels"]) for row in rows], [set(pred) for pred in predictions]
    observed = set().union(*gold_sets, *predicted_sets)
    all_labels = sorted(set(classes) if classes is not None else observed)
    true_positive = sum(len(g & p) for g, p in zip(gold_sets, predicted_sets))
    denominator = sum(len(g) + len(p) for g, p in zip(gold_sets, predicted_sets))
    label_scores = {}
    for label in all_labels:
        tp = sum(label in g and label in p for g, p in zip(gold_sets, predicted_sets))
        total = sum(label in g for g in gold_sets) + sum(label in p for p in predicted_sets)
        label_scores[label] = {"f1": 2 * tp / total if total else 0., "support": sum(label in g for g in gold_sets)}
    return {"count": len(rows), "exact_match": float(np.mean(exact)), "samples_f1": float(np.mean(f1)),
            "micro_f1": 2 * true_positive / denominator if denominator else 0.,
            "macro_f1": float(np.mean([v["f1"] for v in label_scores.values()])) if label_scores else 0.,
            "macro_f1_observed_labels": float(np.mean([v["f1"] for k, v in label_scores.items() if k in observed])) if observed else 0.,
            "per_label": label_scores,
            "by_language": {lang: {"count": sum(r["language"] == lang for r in rows),
                                  "samples_f1": float(np.mean([s for r, s in zip(rows, f1) if r["language"] == lang])),
                                  "exact_match": float(np.mean([s for r, s in zip(rows, exact) if r["language"] == lang]))}
                            for lang in languages}}


def paired_bootstrap(first, second, seed=42, repeats=2000):
    if len(first) != len(second) or not len(first):
        raise ValueError("Paired scores must have equal nonzero length")
    delta = np.asarray(second) - np.asarray(first)
    rng = np.random.default_rng(seed)
    means = [rng.choice(delta, len(delta), replace=True).mean() for _ in range(repeats)]
    return {"mean_difference": float(delta.mean()), "ci95": np.quantile(means, [.025, .975]).tolist()}

