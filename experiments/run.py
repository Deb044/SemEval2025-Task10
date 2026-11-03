"""Run controlled classification comparisons from a small JSON manifest.

Usage: python -m experiments.run experiments/example.json
All outputs remain private under artifacts/. No automatic benchmark claims.
"""
import hashlib
import json
import platform
import pickle
import subprocess
import sys
import time
from statistics import mean, stdev
from pathlib import Path

import joblib
from semeval.classify import predict, train
from semeval.data import check_splits, read_jsonl, write_json, write_jsonl
from semeval.evaluate import evaluate, export_tsv
from semeval.metrics import paired_bootstrap, set_f1
from semeval.resources import peak_rss_mb


def run(config_path, resume=False):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    datasets = {name: read_jsonl(config[name]) for name in ("train", "dev", "test")}
    check_splits(*datasets.values())
    task = datasets["train"][0]["task"]
    output = Path(config["output"])
    input_hashes = {name: hashlib.sha256(Path(config[name]).read_bytes()).hexdigest()
                    for name in (*datasets, "taxonomy") if config.get(name)}
    root = Path(__file__).resolve().parent.parent
    source_paths = [Path(__file__).resolve(), *(root / "semeval" / name for name in
        ("classify.py", "data.py", "features.py", "graph.py", "metrics.py", "evaluate.py"))]
    source_hashes = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    signature = hashlib.sha256(json.dumps({"config": config, "inputs": input_hashes,
        "source": source_hashes}, sort_keys=True).encode()).hexdigest()
    manifest_path = output / "run.json"
    if output.exists() and any(output.iterdir()):
        if not resume or not manifest_path.exists():
            raise ValueError("Nonempty output requires --resume and a matching run.json; preserve legacy runs and choose a fresh directory")
        if json.loads(manifest_path.read_text(encoding="utf-8"))["signature"] != signature:
            raise ValueError("Checkpoint inputs, configuration, or implementation differ; choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    write_json(manifest_path, {"signature": signature, "config": config,
                              "input_sha256": input_hashes, "source_sha256": source_hashes})
    reports, article_scores = [], {}
    variants = [(name, "none") for name in ("tfidf", "lsa", "transformer")]
    if task == 2:
        variants += [("graph", ablation) for ablation in ("none", "self-loops", "random-labels")]
    for method, ablation in variants:
        seeds = [13, 42, 87] if method == "graph" else [42]
        for seed in seeds:
            name = f"{method}-{ablation}-{seed}"
            model_path = output / (name + ".joblib")
            model = None
            if resume and model_path.exists():
                try:
                    model = joblib.load(model_path)
                except (OSError, EOFError, ValueError, pickle.UnpicklingError):
                    pass  # Incomplete checkpoint; retrain this variant only.
                if model is not None and model.get("run_signature") != signature:
                    raise ValueError(f"Model checkpoint provenance mismatch: {name}")
            if model is None:
                print("Training", name, flush=True)
                model = train(datasets["train"], datasets["dev"], method, config.get("taxonomy"), seed, ablation,
                              native=config.get("label_schema") == "native")
                model["run_signature"] = signature
                model["data_sha256"] = input_hashes
                temporary = model_path.with_suffix(".joblib.tmp")
                joblib.dump(model, temporary)
                temporary.replace(model_path)
            started = time.perf_counter()
            predictions_path = output / (name + ".jsonl")
            predictions = None
            if resume and predictions_path.exists():
                try:
                    candidate = read_jsonl(predictions_path)
                    from semeval.evaluate import align
                    predictions = align(datasets["test"], candidate)
                except (ValueError, json.JSONDecodeError):
                    pass  # Interrupted prediction write: regenerate from the checked model.
            reused_predictions = predictions is not None
            if not reused_predictions:
                predictions = predict(model, datasets["test"])
            predict_seconds = time.perf_counter() - started
            if reused_predictions:
                previous_path = output / (name + ".metrics.json")
                try:
                    predict_seconds = json.loads(previous_path.read_text(encoding="utf-8")).get("predict_seconds")
                except (OSError, ValueError):
                    predict_seconds = None
            write_jsonl(predictions_path, predictions)
            export_tsv(predictions, output / (name + ".tsv"))
            report = evaluate(datasets["test"], predictions)
            report.update(name=name, method=method, ablation=ablation, seed=seed,
                          fit_seconds=model["fit_seconds"], predict_seconds=predict_seconds,
                          process_peak_rss_mb=peak_rss_mb(),
                          threshold=model["threshold"], regularization=model["regularization"])
            report["reused_predictions"] = reused_predictions
            write_json(output / (name + ".metrics.json"), report)
            print("Completed", name, flush=True)
            reports.append(report)
            article_scores[name] = [float(set(g["labels"]) == set(p["labels"])) if task == 1
                                   else set_f1(g["labels"], p["labels"])
                                   for g, p in zip(datasets["test"], predictions, strict=True)]
    comparisons = {}
    if task == 2:
        averaged = {ablation: [mean(values) for values in zip(
            *(article_scores[f"graph-{ablation}-{seed}"] for seed in (13, 42, 87)), strict=True)]
            for ablation in ("none", "self-loops", "random-labels")}
        graph_mean = averaged["none"]
        comparisons["transformer_vs_graph_mean_of_three_seeds"] = paired_bootstrap(article_scores["transformer-none-42"], graph_mean)
        comparisons["self_loops_vs_taxonomy_graph"] = paired_bootstrap(averaged["self-loops"], graph_mean)
        comparisons["random_labels_vs_definition_labels"] = paired_bootstrap(averaged["random-labels"], graph_mean)
        for comparison in comparisons.values():
            comparison["metric"] = "accuracy" if config.get("label_schema") == "native" else "samples_f1"
    aggregates = []
    for method, ablation in variants:
        matching = [r for r in reports if r["method"] == method and r["ablation"] == ablation]
        values = [r["metrics"][r["primary_metric"]] for r in matching]
        aggregates.append({"method": method, "ablation": ablation, "seeds": [r["seed"] for r in matching],
                           "mean": mean(values), "sample_std": stdev(values) if len(values) > 1 else None})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    summary = {"config": config, "commit": commit, "python": platform.python_version(),
               "dataset": config.get("dataset", "unspecified"), "label_schema": config.get("label_schema", "semeval"),
               "synthetic": bool(config.get("synthetic", False)),
               "input_sha256": input_hashes, "source_sha256": source_hashes,
               "runs": reports, "aggregates": aggregates, "paired_comparisons": comparisons,
               "memory_note": "process_peak_rss_mb is cumulative high-water RSS across sequential runs, not isolated per-model peak",
               "status": "local_metrics_only", "organizer_scorer_verified": False}
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.config, args.resume)
