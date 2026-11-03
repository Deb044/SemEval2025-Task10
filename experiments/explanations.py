"""Controlled ST3 comparison on one fixed language/domain-balanced subset."""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from semeval.data import read_jsonl, write_json, write_jsonl
from semeval.evaluate import evaluate


def balanced_subset(rows, limit):
    if limit < 1:
        raise ValueError("Subset size must be positive")
    groups = defaultdict(list)
    for row in rows:
        groups[(row["language"], row["dominant_narrative"].split(":", 1)[0])].append(row)
    rng = random.Random(42)
    for key in sorted(groups):
        groups[key].sort(key=lambda r: r["article_id"])
        rng.shuffle(groups[key])
    chosen = []
    while len(chosen) < min(limit, len(rows)):
        for key in sorted(groups):
            if groups[key] and len(chosen) < limit:
                chosen.append(groups[key].pop())
    return sorted(chosen, key=lambda r: r["article_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--predicted-labels")
    parser.add_argument("--limit", type=int, default=250, help="Fixed before examining outcomes; increase for full evaluation")
    parser.add_argument("--bert-model")
    parser.add_argument("--bert-layers", type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    subset = balanced_subset(read_jsonl(args.data), args.limit)
    subset_path = output / "evaluation.jsonl"
    if subset_path.exists() and read_jsonl(subset_path) != subset:
        raise ValueError("Existing output uses a different evaluation subset; choose a fresh directory")
    write_jsonl(subset_path, subset)
    variants = [("textrank", "textrank", "semantic", None),
                ("semantic", "semantic", "semantic", None),
                ("generate", "generate", "semantic", None),
                ("lead-ablation", "generate", "lead", None)]
    if args.predicted_labels:
        variants.append(("predicted-labels", "generate", "semantic", args.predicted_labels))
    reports = []
    for name, method, selection, labels in variants:
        predictions_path = output / (name + ".jsonl")
        command = [sys.executable, "-m", "semeval", "explain", "--data", str(subset_path),
                   "--taxonomy", args.taxonomy, "--method", method, "--selection", selection,
                   "--out", str(predictions_path)]
        if method == "generate":
            command += ["--gguf", args.gguf]
        if labels:
            command += ["--predicted-labels", labels]
        started = time.perf_counter()
        subprocess.run(command, check=True)
        predictions = read_jsonl(predictions_path)
        report = evaluate(subset, predictions, args.bert_model, args.bert_layers)
        report.update(name=name, method=method, seconds_including_resume=time.perf_counter() - started,
                      generation_seconds=sum(r.get("seconds", 0) for r in predictions))
        reports.append(report)
        write_json(output / (name + ".metrics.json"), report)
    summary = {"runs": reports, "subset_size": len(subset), "subset_seed": 42,
               "subset_sha256": hashlib.sha256(subset_path.read_bytes()).hexdigest(),
               "input_sha256": {"evaluation": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                                "taxonomy": hashlib.sha256(Path(args.taxonomy).read_bytes()).hexdigest()},
               "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "organizer_scorer_verified": False,
               "skipped": [] if args.predicted_labels else ["End-to-end mode: no ST2 predictions supplied"]}
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()
