"""Isolated inference timing for the two core transformer–GNN variants."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiments.common import provenance
from semeval.data import read_jsonl, write_json
from semeval.resources import peak_rss_mb

MODELS = {"minilm": "artifacts/extensions-probel/graph-none-42.joblib",
          "xlmr": "artifacts/extensions-probel/xlmr-graph-none-42.joblib"}


def worker(name, count):
    import joblib
    import torch
    from semeval.classify import predict
    torch.set_num_threads(4)
    rows = read_jsonl("data/public/probel/test.jsonl")[:count]
    started = time.perf_counter()
    model = joblib.load(MODELS[name])
    load = time.perf_counter() - started
    times, labels = [], None
    for _ in range(2):
        started = time.perf_counter()
        values = predict(model, rows)
        times.append(time.perf_counter() - started)
        current = [row["labels"] for row in values]
        if labels is not None and labels != current:
            raise ValueError("Repeated inference changed labels")
        labels = current
    if model["features"].encoder._model is not None:
        raise ValueError("Embedding cache was incomplete; finish classification before timing cache-backed inference")
    return {"model": name + " + hierarchical GCN", "count": len(rows), "model_load_seconds": load,
        "first_inference_seconds": times[0], "repeat_inference_seconds": times[1],
        "repeat_records_per_second": len(rows) / times[1], "peak_rss_mb": peak_rss_mb(),
        "repeated_labels_identical": True, "embedding_cache_complete": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts/extensions-throughput")
    parser.add_argument("--worker", choices=tuple(MODELS))
    parser.add_argument("--count", type=int)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.count)))
        return
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preserve completed timings; choose a fresh output folder")
    meta = provenance(["experiments/throughput.py", "semeval/classify.py", "semeval/features.py", "semeval/graph.py"],
                      {"evaluation": "data/public/probel/test.jsonl", **MODELS})
    records = []
    for name in MODELS:
        for count in (32, 128, 400):
            value = json.loads(subprocess.check_output([sys.executable, "-m", "experiments.throughput", "--worker", name, "--count", str(count)], text=True))
            records.append(value)
            print(f"Measured {name} GNN inference, n={count}", flush=True)
    write_json(output / "summary.json", {**meta, "experiment_id": "core-inference-throughput", "measurements": records,
        "dataset": "ProBel grouped held-out news", "threads": 4,
        "limitations": "Fresh subprocess per model/size. Model loading is separate; inference includes reading existing per-text embeddings, projection, graph scoring and decoding. It is not raw-text transformer throughput. First/repeat calls may benefit from OS caches; peak RSS includes imports."})


if __name__ == "__main__":
    main()
