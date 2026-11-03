"""Run controlled classification comparisons from a small JSON manifest.

Usage: python -m experiments.run experiments/example.json
All outputs remain private under artifacts/. No automatic benchmark claims.
"""
import hashlib
import json
import platform
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


def run(config_path):
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    datasets = {name: read_jsonl(config[name]) for name in ("train", "dev", "test")}
    check_splits(*datasets.values())
    task = datasets["train"][0]["task"]
    output = Path(config["output"])
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose a new output directory; never overwrite an existing experiment")
    output.mkdir(parents=True, exist_ok=True)
    reports, article_scores = [], {}
    variants = [(name, "none") for name in ("tfidf", "lsa", "transformer")]
    if task == 2:
        variants += [("graph", ablation) for ablation in ("none", "self-loops", "random-labels")]
    for method, ablation in variants:
        seeds = [13, 42, 87] if method == "graph" else [42]
        for seed in seeds:
            name = f"{method}-{ablation}-{seed}"
            model = train(datasets["train"], datasets["dev"], method, config.get("taxonomy"), seed, ablation)
            joblib.dump(model, output / (name + ".joblib"))
            started = time.perf_counter()
            predictions = predict(model, datasets["test"])
            predict_seconds = time.perf_counter() - started
            write_jsonl(output / (name + ".jsonl"), predictions)
            export_tsv(predictions, output / (name + ".tsv"))
            report = evaluate(datasets["test"], predictions)
            report.update(name=name, method=method, ablation=ablation, seed=seed,
                          fit_seconds=model["fit_seconds"], predict_seconds=predict_seconds,
                          process_peak_rss_mb=peak_rss_mb(),
                          threshold=model["threshold"], regularization=model["regularization"])
            write_json(output / (name + ".metrics.json"), report)
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
               "input_sha256": {name: hashlib.sha256(Path(config[name]).read_bytes()).hexdigest()
                                for name in (*datasets, "taxonomy") if config.get(name)},
               "runs": reports, "aggregates": aggregates, "paired_comparisons": comparisons,
               "memory_note": "process_peak_rss_mb is cumulative high-water RSS across sequential runs, not isolated per-model peak",
               "status": "local_metrics_only", "organizer_scorer_verified": False}
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    run(sys.argv[1])
