"""Export aggregate results and an SVG plot after manual provenance review.

Never publish articles, reference explanations, or predictions. This command does
not certify that a dataset is official; the researcher must verify that separately.
"""
import argparse
import html
import json
import math
from pathlib import Path
from statistics import mean, stdev

from semeval.data import write_json


def aggregate(summaries):
    """Preserve dataset identity; aggregate seeds only within the same variant."""
    groups = {}
    for summary in summaries:
        if summary.get("synthetic") or not summary.get("commit") or not summary.get("input_sha256"):
            raise ValueError("Only reviewed measured runs with source and data hashes may be exported")
        dataset = summary.get("dataset", "unspecified")
        if dataset == "unspecified":
            raise ValueError("Declare dataset identity in the experiment manifest before publication")
        schema = summary.get("label_schema", "semeval")
        if schema == "native" and dataset == "SemEval-2025 Task 10":
            raise ValueError("Native auxiliary labels cannot be presented as SemEval results")
        for run in summary["runs"]:
            metric = run["primary_metric"]
            value = run["metrics"].get(metric)
            if value is None:
                continue
            if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Invalid measured score; do not publish placeholders or non-finite values")
            name = run["method"] + (" / " + run["ablation"] if run.get("ablation") not in (None, "none") else "") if run.get("method") else run["name"]
            experiment_id = summary.get("experiment_id", "legacy")
            key = (dataset, experiment_id, run["task"], name, metric)
            if key not in groups:
                groups[key] = {"dataset": dataset, "experiment_id": experiment_id, "label_schema": schema, "measured": True,
                    "task": run["task"], "name": name, "metric": metric, "values": [], "seeds": [],
                    "count": run["metrics"]["count"], "split": "held-out evaluation; see protocol",
                    "commit": summary["commit"], "scoring_provenance": run["scoring_provenance"],
                    "source_sha256": summary.get("source_sha256", {}),
                    "input_sha256": summary["input_sha256"], "fit_seconds": [], "predict_seconds": [],
                    "generation_seconds": [], "run_names": [], "language_values": {}, "seed_values": []}
            record = groups[key]
            if record["count"] != run["metrics"]["count"]:
                raise ValueError("Cannot aggregate variants with different evaluation counts")
            if record["input_sha256"] != summary["input_sha256"] or run["name"] in record["run_names"]:
                raise ValueError("Cannot pool different splits or count the same run twice")
            record["run_names"].append(run["name"])
            record["values"].append(value)
            record["seed_values"].append({"name": run["name"], "seed": run.get("seed"), "value": value})
            for language, scores in run["metrics"].get("by_language", {}).items():
                if metric not in scores:
                    continue
                language_value = scores[metric]
                if not isinstance(language_value, (int, float)) or not math.isfinite(language_value) or not 0 <= language_value <= 1:
                    raise ValueError("Invalid per-language score")
                group = record["language_values"].setdefault(language, {"count": scores["count"], "values": []})
                if group["count"] != scores["count"]:
                    raise ValueError("Per-language evaluation counts differ between seeds")
                group["values"].append(language_value)
            if run.get("seed") is not None:
                record["seeds"].append(run["seed"])
            if run.get("fit_seconds") is not None:
                record["fit_seconds"].append(run["fit_seconds"])
            if run.get("predict_seconds") is not None:
                record["predict_seconds"].append(run["predict_seconds"])
            if run.get("generation_seconds") is not None:
                record["generation_seconds"].append(run["generation_seconds"])
    for record in groups.values():
        values = record.pop("values")
        record["value"] = mean(values)
        record["seed_std"] = stdev(values) if len(values) > 1 else None
        record["by_language"] = {language: {"count": group["count"], "value": mean(group["values"]),
            "seed_std": stdev(group["values"]) if len(group["values"]) > 1 else None}
            for language, group in record.pop("language_values").items()}
        for field in ("fit_seconds", "predict_seconds", "generation_seconds"):
            record[field] = mean(record[field]) if record[field] else None
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", nargs="+")
    parser.add_argument("--reviewed", action="store_true", help="Confirm data identity, split integrity, and aggregate release permission were reviewed")
    args = parser.parse_args()
    if not args.reviewed:
        parser.error("Review the source run and dataset identity before using --reviewed. Synthetic runs must never be published as benchmark results.")
    summaries = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.summary]
    runs = aggregate(summaries)
    if not runs:
        raise ValueError("No measured primary metrics; nothing to publish")
    payload = {"status": "measured_local_metrics", "runs": runs,
               "paired_comparisons": [{"dataset": summary["dataset"],
                    "comparisons": summary.get("paired_comparisons", {})} for summary in summaries],
               "organizer_scorer_verified": False,
               "note": "Compare methods within each dataset only. Auxiliary tasks are not SemEval benchmark scores."}
    write_json("results/measurements.json", payload)
    write_json("website/dist/results.json", payload)
    # Whitelist aggregate diagnostics: never export per-example scores or text.
    diagnostics = []
    fields = ("count", "exact_match", "samples_f1", "micro_f1", "macro_f1", "per_label", "by_relation", "by_language",
              "bertscore_f1", "bertscore_hash", "length_compliance", "fallback_rate")
    for summary in summaries:
        diagnostics.append({"dataset": summary["dataset"], "commit": summary["commit"],
            "input_sha256": summary["input_sha256"], "source_sha256": summary.get("source_sha256", {}),
            "runs": [{"name": run["name"], "metrics": {key: run["metrics"][key]
                for key in fields if key in run["metrics"]}} for run in summary["runs"]]})
    write_json("results/diagnostics.json", diagnostics)
    for dataset in sorted({run["dataset"] for run in runs}):
        plot([run for run in runs if run["dataset"] == dataset], dataset)
    print("Exported reviewed aggregate metrics, retaining native dataset identities.")


def plot(runs, dataset):
    height = 100 + 40 * len(runs)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 {height}" role="img">',
             '<rect width="100%" height="100%" fill="#f5f4ed"/>',
             '<g font-family="Arial,sans-serif" fill="#172e32">',
             f'<text x="25" y="30" font-size="18">{html.escape(dataset)} · {html.escape(runs[0]["metric"])}</text>',
             '<text x="25" y="52" font-size="11">' +
             ('Local Task 10 metrics; organizer scorer not verified.' if dataset == "SemEval-2025 Task 10" else
              'Native dataset evaluation. Not a SemEval-2025 Task 10 benchmark score.') + '</text>']
    for index, run in enumerate(runs):
        y = 85 + 40 * index
        parts += [f'<text x="25" y="{y + 14}" font-size="11">{html.escape(run["name"])}</text>',
                  f'<rect x="290" y="{y}" width="500" height="21" fill="#e1e7dc"/>',
                  f'<rect x="290" y="{y}" width="{500 * run["value"]:.2f}" height="21" fill="#275b53"/>',
                  f'<text x="810" y="{y + 15}" font-size="12">{run["value"]:.3f}</text>']
    parts += ['</g></svg>']
    slug = "".join(character.lower() if character.isalnum() else "-" for character in dataset).strip("-")
    Path(f"results/{slug}.svg").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
