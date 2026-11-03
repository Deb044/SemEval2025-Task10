"""Build the complete public experiment catalog without releasing source text.

Historical aggregates remain immutable. New model, format, systems and human
review evidence share one typed catalog; missing measurements are never zeros.
"""
import json
from pathlib import Path

from experiments.common import provenance
from experiments.publish import aggregate
from semeval.data import read_jsonl, write_json

STUDIES = [
    ("artifacts/extensions-cards/summary.json", "CARDS · extended hierarchy study", "classification"),
    ("artifacts/extensions-probel/summary.json", "ProBel · extended bilingual study", "classification"),
    ("artifacts/extensions-emm/summary.json", "CARDS → EMM source transfer", "classification"),
    ("artifacts/extensions-long-explanations/summary.json", "CARDS · longer-text explanations", "explanation"),
    ("artifacts/extensions-grounding/summary.json", "CARDS · independent grounding review", "review"),
]
NAMES = {"tfidf": "Character TF-IDF + logistic regression", "lsa": "LSA + logistic regression",
    "transformer": "MiniLM + logistic regression", "graph": "MiniLM + hierarchical GCN",
    "xlmr-transformer": "XLM-R + logistic regression", "xlmr-graph": "XLM-R + hierarchical GCN",
    "copy-input": "Copied input", "label-template": "Label-only template",
    "textrank": "TextRank extraction", "semantic": "Semantic extraction",
    "generate-semantic": "Qwen · semantic evidence", "generate-lead": "Qwen · leading evidence",
    "generate-full": "Qwen · full context", "generate-no-premise": "Qwen · premise withheld"}


def model_name(name):
    base, separator, ablation = name.partition(" / ")
    return NAMES.get(base, base) + (" · " + ablation if separator else "")


def model_entry(row, study, index):
    graph = "graph" in row["name"]
    role = "Core" if graph and " / " not in row["name"] else ("Ablation" if graph else "Comparison")
    return {"id": f"{study}-{index}", "study": study, "dataset": row["dataset"], "kind": "classification" if row["task"] in (1, 2) else "explanation",
        "role": role, "name": model_name(row["name"]), "metric": row["metric"], "value": row["value"], "unit": "score",
        "n": row["count"], "seeds": row.get("seeds", []), "seed_std": row.get("seed_std"), "status": "measured",
        "by_language": row.get("by_language", {}), "notes": row["scoring_provenance"],
        "provenance": {key: row.get(key) for key in ("commit", "source_sha256", "input_sha256")},
        "details": {"individual_runs": row.get("seed_values", []), "input_sha256": row["input_sha256"], "fit_seconds": row.get("fit_seconds"), "prediction_seconds": row.get("predict_seconds"), "generation_seconds": row.get("generation_seconds")}}


def build():
    historical = json.loads(Path("results/history.json").read_text(encoding="utf-8"))
    error_path = Path("results/error-analysis.json")
    errors = json.loads(error_path.read_text(encoding="utf-8"))["studies"] if error_path.exists() else {}
    entries = []
    for index, row in enumerate(historical["runs"]):
        entry = model_entry(row, row["dataset"] + " · initial study", index)
        comparisons = next((value.get("comparisons", {}) for value in historical.get("paired_comparisons", []) if value["dataset"] == row["dataset"]), {})
        entry["details"]["paired_comparisons"] = comparisons
        entries.append(entry)
    for path, study, kind in STUDIES:
        if not Path(path).exists():
            entries.append({"id": path, "study": study, "dataset": study.split(" · ")[0], "kind": kind,
                "name": "Study results not exported yet", "status": "not_measured", "value": None, "metric": "", "notes": "Run the fixed reproduction workflow."})
            continue
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        scored = aggregate([summary])
        for index, row in enumerate(scored):
            entry = model_entry(row, study, index)
            entry["kind"] = kind
            entry["details"]["paired_comparisons"] = summary.get("paired_comparisons", {})
            diagnostic_name = next((value["name"] for value in row.get("seed_values", []) if value["name"].endswith("-42")), None)
            if diagnostic_name in errors.get(study, {}):
                entry["details"]["fixed_seed_42_error_analysis"] = errors[study][diagnostic_name]
            if kind == "review":
                entry["role"] = "Model-based review"
                entry["details"]["evaluation_protocol"] = {key: summary.get(key) for key in ("nli_model", "nli_revision", "nli_weights_sha256", "protocol", "llm_audit", "limitations")}
            entries.append(entry)
        for report in summary["runs"]:
            if report["metrics"].get(report["primary_metric"]) is not None:
                continue
            metrics = report["metrics"]
            entries.append({"id": study + report["name"], "study": study, "dataset": summary["dataset"], "kind": kind,
                "name": report["name"], "metric": "length_compliance", "value": metrics["length_compliance"], "unit": "format fraction", "n": metrics["count"],
                "status": "measured_format_only", "role": "Evidence study",
                "format": f"Length compliance {metrics['length_compliance']:.0%}; fallback {metrics['fallback_rate']:.0%}",
                "notes": "Format statistics only. Independent model-based grounding is evaluated separately on a fixed blinded sample; no human validation is claimed.", "details": {"format_metrics": metrics, "study_statistics": summary.get("statistics", {})},
                "provenance": {key: summary.get(key) for key in ("commit", "input_sha256", "source_sha256")}})
    scale = Path("artifacts/extensions-scalability/summary.json")
    if scale.exists():
        summary = json.loads(scale.read_text(encoding="utf-8"))
        meta = {key: summary.get(key) for key in ("commit", "input_sha256", "source_sha256")}
        for record in summary["measurements"]:
            for operation in ("conversion", "json", "parquet"):
                result = record[operation]
                entries.append({"id": f"scale-{record['count']}-{operation}", "study": "Data processing scalability", "dataset": summary["dataset"],
                    "kind": "scalability", "name": {"conversion": "JSON → Parquet", "json": "JSON aggregation", "parquet": "Parquet aggregation"}[operation],
                    "metric": "conversion time" if operation == "conversion" else "first scan time", "unit": "seconds", "value": result.get("seconds", result.get("first_scan_seconds")),
                    "n": record["count"], "status": "measured", "details": result, "provenance": meta, "notes": summary["limitations"]})
        for record in summary["embeddings"]:
            for mode in ("uncached", "cached"):
                entries.append({"id": f"embeddings-{record['count']}-{mode}", "study": "Embedding cache scalability", "dataset": summary["dataset"],
                    "kind": "scalability", "name": "MiniLM · " + mode, "metric": "encoding time (weights loaded)", "unit": "seconds", "value": record[mode + "_seconds"],
                    "n": record["count"], "status": "measured", "details": record, "provenance": meta, "notes": summary["limitations"]})
    else:
        entries.append({"id": "scale-pending", "study": "Data processing scalability", "dataset": "Public news", "kind": "scalability", "name": "Benchmark results not exported yet", "value": None, "status": "not_measured"})
    throughput = Path("artifacts/extensions-throughput/summary.json")
    if throughput.exists():
        summary = json.loads(throughput.read_text(encoding="utf-8"))
        for record in summary["measurements"]:
            entries.append({"id": f"throughput-{record['model']}-{record['count']}", "study": "Core model inference throughput", "dataset": summary["dataset"],
                "kind": "scalability", "name": record["model"], "role": "Core", "metric": "repeat inference time (cached embeddings)", "unit": "seconds",
                "value": record["repeat_inference_seconds"], "n": record["count"], "status": "measured", "details": record,
                "notes": summary["limitations"], "provenance": {key: summary.get(key) for key in ("commit", "input_sha256", "source_sha256")}})
    else:
        entries.append({"id": "throughput-pending", "study": "Core model inference throughput", "dataset": "ProBel", "kind": "scalability", "name": "Timings not exported yet", "value": None, "status": "not_measured"})
    payload = {"experiments": entries,
               "review_method": "Independent blinded LLM audit plus pinned NLI diagnostics; not human validation.",
               "note": "Typed results have different units and tasks; compare within studies. Public alternatives are not SemEval benchmark scores."}
    write_json("results/experiments.json", payload)
    write_json("website/dist/experiments.json", payload)
    print(f"Published {len(entries)} typed experiment entries")


if __name__ == "__main__":
    build()
