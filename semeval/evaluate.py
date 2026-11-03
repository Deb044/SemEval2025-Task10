"""Align by IDs, reject incomplete outputs, keep scoring provenance explicit."""
import platform
from pathlib import Path

from .data import identity
from .metrics import classification_scores


def align(gold, predicted):
    gold_ids = [identity(row) for row in gold]
    if len(set(gold_ids)) != len(gold_ids):
        raise ValueError("Gold examples must have unique IDs")
    lookup = {identity(row): row for row in predicted}
    if len(lookup) != len(predicted) or set(lookup) != set(gold_ids):
        raise ValueError("Predictions must match gold IDs exactly, with no duplicates")
    return [lookup[identity(row)] for row in gold]


def evaluate(gold, predicted, bert_model=None, bert_layers=None):
    predicted = align(gold, predicted)
    task = gold[0]["task"]
    if any(row["task"] != task for row in gold + predicted):
        raise ValueError("Evaluate one subtask at a time")
    if task in (1, 2):
        scores = classification_scores(gold, [r["labels"] for r in predicted])
        if task == 2:
            coarse = classification_scores([dict(r, labels=r["narratives"]) for r in gold], [r["narratives"] for r in predicted])
            scores["coarse_samples_f1"] = coarse["samples_f1"]
        primary = "exact_match" if task == 1 else "samples_f1"
        provenance = "local implementation of published metric; organizer scorer parity NOT verified"
    else:
        scores = {"count": len(gold), "length_compliance": sum(0 < len(r["explanation"].split()) <= 80 for r in predicted) / len(gold),
                  "fallback_rate": sum(r.get("fallback", False) for r in predicted) / len(gold)}
        primary = "bertscore_f1"
        provenance = "BERTScore not run"
        if bert_model:
            if bert_layers is None:
                raise ValueError("Specify BERTScore layers from the official scorer; do not guess defaults")
            from bert_score import score
            _, _, values, hashcode = score([r["explanation"] for r in predicted], [r["explanation"] for r in gold],
                model_type=bert_model, num_layers=bert_layers, device="cpu", batch_size=4,
                rescale_with_baseline=False, return_hash=True, verbose=False)
            scores[primary] = float(values.mean())
            scores["by_language"] = {lang: {"count": sum(r["language"] == lang for r in gold),
                primary: float(values[[r["language"] == lang for r in gold]].mean())}
                for lang in sorted({r["language"] for r in gold})}
            scores["bertscore_hash"] = hashcode
            provenance = "explicit unrescaled BERTScore configuration; verify against organizer scorer before official comparison"
    return {"task": task, "primary_metric": primary, "metrics": scores, "scoring_provenance": provenance,
            "python": platform.python_version(), "benchmark_claim": False}


def export_tsv(rows, path):
    lines = []
    for row in rows:
        if row["task"] == 1:
            fields = [row["article_id"], row["mention"], str(row["start"]), str(row["end"]), row["main_role"], *row["labels"]]
        elif row["task"] == 2:
            fields = [row["article_id"], ";".join(row["narratives"]), ";".join(row["labels"])]
        else:
            fields = [row["article_id"], row["explanation"]]
        if any("\t" in value or "\n" in value or "\r" in value for value in fields):
            raise ValueError("TSV fields must not contain tabs or newlines")
        lines.append("\t".join(fields))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")

