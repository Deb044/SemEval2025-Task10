"""Transparent local metrics. Official parity requires the organizer scorer."""
import numpy as np


def set_f1(gold, predicted):
    gold, predicted = set(gold), set(predicted)
    return 2 * len(gold & predicted) / (len(gold) + len(predicted)) if gold or predicted else 0.0


def classification_scores(rows, predictions):
    exact = [set(row["labels"]) == set(pred) for row, pred in zip(rows, predictions, strict=True)]
    f1 = [set_f1(row["labels"], pred) for row, pred in zip(rows, predictions, strict=True)]
    languages = sorted({row["language"] for row in rows})
    return {"count": len(rows), "exact_match": float(np.mean(exact)), "samples_f1": float(np.mean(f1)),
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

