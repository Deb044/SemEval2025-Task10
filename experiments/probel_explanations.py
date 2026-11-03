"""Bilingual propaganda explanations; references are publisher model-authored.

Uses public news-only ProBel inputs, native technique labels, and no gold spans or
reference explanations in prompts. This is not dominant-narrative evaluation.
"""
import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from semeval.data import read_jsonl, write_json, write_jsonl
from semeval.evaluate import evaluate
from semeval.explain import generator_tokenizer, rank_evidence, select_context


from semeval.propaganda import explain_propaganda as explain


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/public/probel/explanations.jsonl")
    parser.add_argument("--taxonomy", default="data/public/probel/taxonomy.json")
    parser.add_argument("--gguf", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--out", default="artifacts/public-probel-explanations-v2")
    args = parser.parse_args()
    source_paths = ("experiments/probel_explanations.py", "semeval/propaganda.py", "semeval/explain.py", "semeval/evaluate.py", "semeval/features.py", "semeval/data.py")
    sources = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in source_paths}
    inputs = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in (("evaluation", args.data), ("taxonomy", args.taxonomy))}
    with open(args.gguf, "rb") as stream:
        model_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    signature = hashlib.sha256(json.dumps({"sources": sources, "inputs": inputs, "generator": model_hash}, sort_keys=True).encode()).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    rows, taxonomy = read_jsonl(args.data), json.loads(Path(args.taxonomy).read_text())
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    reports, outputs = [], {}
    for method in ("textrank", "semantic", "generate-semantic", "generate-lead"):
        path = output / (method + ".jsonl")
        predictions = read_jsonl(path) if path.exists() else []
        done = {row["article_id"] for row in predictions}
        if len(done) != len(predictions) or not done <= {row["article_id"] for row in rows} or any(row.get("signature") != signature or row.get("method") != method for row in predictions):
            raise ValueError("Checkpoint source/input/model/method mismatch or duplicate IDs")
        for row in rows:
            if row["article_id"] in done:
                continue
            started = time.perf_counter()
            prediction = None
            if method == "generate-lead":
                previous = next(value for value in outputs["generate-semantic"] if value["article_id"] == row["article_id"])
                lead = select_context(rank_evidence(row["text"], "", "lead"), generator_tokenizer())
                if not previous["fallback"] and previous["evidence_ids"] == [unit["id"] for unit in lead]:
                    # Same text, labels, language, evidence order and generator
                    # settings give an exactly equivalent prompt. Reuse visibly.
                    prediction = dict(previous, method=method, reused_equivalent_prompt_from="generate-semantic")
            if prediction is None:
                prediction = explain(row, method, taxonomy, args.endpoint)
            prediction.update(signature=signature, seconds=time.perf_counter() - started)
            predictions.append(prediction)
            write_jsonl(path, predictions)
        by_id = {row["article_id"]: row for row in predictions}
        predictions = [by_id[row["article_id"]] for row in rows]
        report = evaluate(rows, predictions, ".cache/evaluation-mbert", 9)
        report.update(name=method, generation_seconds=sum(row["seconds"] for row in predictions),
            scoring_provenance="Bilingual native propaganda explanations; references authored by OpenAI o1 and publisher human-evaluated. NOT human gold or SemEval narrative scores.")
        reports.append(report)
        outputs[method] = predictions
        write_json(output / (method + ".metrics.json"), report)
        print(method + " complete", flush=True)
    a, b = outputs["generate-lead"], outputs["generate-semantic"]
    comparisons = {"lead_vs_semantic_evidence": {
        "identical_selected_evidence_rate": sum(x["evidence_ids"] == y["evidence_ids"] for x, y in zip(a, b)) / len(rows),
        "equivalent_prompt_reuse_rate": sum("reused_equivalent_prompt_from" in row for row in a) / len(rows),
        "note": "Short news units may supply identical context to both variants; do not assume a retrieval effect."}}
    write_json(output / "summary.json", {"dataset": "ProBel news explanations pilot", "label_schema": "native",
        "commit": commit, "source_sha256": sources, "input_sha256": inputs, "generator_sha256": model_hash,
        "runs": reports, "paired_comparisons": comparisons, "organizer_scorer_verified": False,
        "limitations": "40 researcher-selected held-out news units, EN/AR, given native techniques; model-authored references; no human grounding review of our outputs; not dominant narrative explanations."})


if __name__ == "__main__":
    main()
