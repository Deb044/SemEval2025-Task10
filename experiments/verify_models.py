"""Optional model integration probe using invented text, NOT a benchmark run."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib

from semeval.classify import predict, train
from semeval.data import write_json
from semeval.explain import explain_row
from semeval.features import get_encoder
from semeval.resources import peak_rss_mb


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", help="Optionally test a running local generator using this model file")
    args = parser.parse_args()
    output = Path("artifacts/verification")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    texts = ["Invented policy example.", "काल्पनिक नीति उदाहरण।", "Invented long article. " * 300]
    vectors = get_encoder().encode(texts)
    assert vectors.shape == (3, 384)
    taxonomy = {"Other": {"parent": "Other", "definition": "No specific narrative fits."},
                "cost": {"parent": "policy", "definition": "Claims that policies cause economic harm."},
                "energy": {"parent": "technology", "definition": "Claims that energy technology is unreliable."}}
    write_json(output / "synthetic-taxonomy.json", taxonomy)

    def rows(prefix, count):
        return [{"task": 2, "article_id": f"EN_synthetic_{prefix}{i}.txt", "language": "en",
                 "text": f"Invented {prefix} example {i}. " + ("Policy raises business costs." if i % 2 else "Energy supply is unreliable."),
                 "labels": ["cost" if i % 2 else "energy"]} for i in range(count)]

    checks = ["English, Hindi and long-article embedding shapes"]
    for ablation in ("none", "self-loops", "random-labels"):
        model = train(rows("train", 8), rows("dev", 4), "graph", output / "synthetic-taxonomy.json", ablation=ablation)
        path = output / (ablation + ".joblib")
        joblib.dump(model, path)
        predictions = predict(joblib.load(path), rows("test", 2))
        assert len(predictions) == 2 and all(prediction["labels"] for prediction in predictions)
        checks.append(f"Graph {ablation}: train/save/reload/predict")
    if args.gguf:
        label = "CC: Criticism of climate policies: Climate policies have negative impact on the economy"
        parent = "CC: Criticism of climate policies"
        row = {"task": 3, "article_id": "EN_synthetic_generation.txt", "language": "en",
               "dominant_narrative": parent, "dominant_subnarrative": label,
               "text": "Invented example. Mira says a climate policy raises business costs. She argues that higher taxes reduce hiring. Other speakers disagree and say the policy has benefits."}
        prediction = explain_row(row, "generate", {label: {"parent": parent, "definition": "Claims that climate policies harm the economy."}}, "http://127.0.0.1:8080/v1")
        assert 0 < prediction["words"] <= 80
        with open(args.gguf, "rb") as stream:
            model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        write_json(output / "synthetic-explanation.json", dict(prediction, synthetic=True, gguf_sha256=model_hash))
        checks.append("Local explanation output and length; grounding still requires manual review")
    report = {"synthetic": True, "benchmark_result": False, "checks": checks,
              "seconds": time.perf_counter() - started, "process_peak_rss_mb": peak_rss_mb()}
    write_json(output / "checks.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
