"""Independent model-based grounding diagnostics for the fixed blinded sample.

A pinned English NLI model supplies repeatable entailment proxies. Optional
blinded LLM ratings are a qualitative audit, never human gold or truth labels.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from experiments.common import provenance, sha256
from semeval.data import read_jsonl, write_json, write_jsonl
from semeval.explain import sentences
from semeval.review import item_hash

MODEL = "cross-encoder/nli-deberta-v3-small"
REVISION = "fa2804872c3b4bd748f38c0185cc85775361e735"
CACHE = ".cache/nli-evaluator"
CRITERIA = ("supported", "attribution", "relevance", "readability")


def download():
    from huggingface_hub import snapshot_download
    return snapshot_download(MODEL, revision=REVISION, local_dir=CACHE, max_workers=1,
        allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "spm.model"])


def windows(units):
    return [" ".join(unit["text"] for unit in units[start:start + 3]) for start in range(0, len(units), 2)]


class Entailment:
    def __init__(self, path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        torch.set_num_threads(4)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).eval()
        self.cache = {}
        # Model card specifies contradiction / entailment / neutral, in that order.
        if self.model.config.num_labels != 3:
            raise ValueError("Unexpected NLI label inventory")

    def scores(self, premises, hypotheses):
        import torch
        keys = [(premise, hypothesis) for hypothesis in hypotheses for premise in premises]
        missing = list(dict.fromkeys(key for key in keys if key not in self.cache))
        for start in range(0, len(missing), 8):
            batch = missing[start:start + 8]
            encoded = self.tokenizer([pair[0] for pair in batch], [pair[1] for pair in batch],
                padding=True, truncation="only_first", max_length=512, return_tensors="pt")
            with torch.inference_mode():
                values = torch.softmax(self.model(**encoded).logits, dim=1)[:, 1].tolist()
            self.cache.update(zip(batch, values))
        return [max(self.cache[(premise, hypothesis)] for premise in premises) for hypothesis in hypotheses]


def llm_summary(queue, path):
    ratings = read_jsonl(path)
    lookup = {row["review_id"]: row for row in queue}
    if len(ratings) != len(lookup) or {row["review_id"] for row in ratings} != set(lookup):
        raise ValueError("LLM audit must cover each blinded item exactly once")
    grouped = defaultdict(list)
    for rating in ratings:
        if rating.get("reviewer_type") != "llm" or any(type(rating.get(key)) is not int or rating[key] not in (0, 1, 2) for key in CRITERIA):
            raise ValueError("Invalid model-based rating; never substitute it for human review")
        source = lookup[rating["review_id"]]
        ids = {unit["id"] for unit in sentences(source["text"])}
        if not set(rating.get("evidence_ids", [])) <= ids:
            raise ValueError("LLM cited an invalid source sentence ID")
        grouped[source["variant"]].append(rating)
    runs = []
    for name, values in sorted(grouped.items()):
        runs.append({"name": name + " · blinded LLM audit", "task": 3, "primary_metric": "llm_fully_supported_rate",
            "metrics": {"count": len(values), "llm_fully_supported_rate": sum(row["supported"] == 2 for row in values) / len(values),
                "mean_ratings": {key: sum(row[key] for row in values) / len(values) for key in CRITERIA}},
            "scoring_provenance": "Single blinded Codex LLM judge; 0/1/2 source-support rubric; not human ratings, calibrated factuality or truth validation."})
    return {"runs": runs, "judge_models": sorted({row["judge_model"] for row in ratings}),
        "input_sha256": {"llm_ratings": sha256(path)},
        "limitations": "Exact judge runtime model identifier is unavailable. Qualitative model judgments are frozen ancillary evidence, not bitwise reproducible gold. Six source passages, five outputs each; no inter-judge agreement study."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", default="artifacts/extensions-long-explanations/review-queue.jsonl")
    parser.add_argument("--llm-ratings", default="artifacts/extensions-long-explanations/llm-ratings.jsonl")
    parser.add_argument("--out", default="artifacts/extensions-grounding")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()
    path = download()
    if args.download_only:
        return
    queue = read_jsonl(args.queue)
    if any(row["language"] != "en" for row in queue):
        raise ValueError("This NLI diagnostic is validated for English inputs only")
    output = Path(args.out)
    meta = provenance(["experiments/grounding.py", "semeval/explain.py"], {"review_queue": args.queue})
    signature = hashlib.sha256(json.dumps({"sources": meta["source_sha256"], "inputs": meta["input_sha256"], "revision": REVISION}, sort_keys=True).encode()).hexdigest()
    checkpoint = output / "nli-scores.jsonl"
    values = read_jsonl(checkpoint) if checkpoint.exists() else []
    if any(row.get("signature") != signature for row in values) or len({row["review_id"] for row in values}) != len(values):
        raise ValueError("NLI checkpoint uses different source/input/model or duplicate IDs")
    done = {row["review_id"] for row in values}
    if not done <= {row["review_id"] for row in queue}:
        raise ValueError("Unknown NLI checkpoint IDs")
    evaluator = None
    for row in queue:
        if row["review_id"] in done:
            continue
        evaluator = evaluator or Entailment(path)
        units = sentences(row["text"])
        hypotheses = [unit["text"] for unit in sentences(row["explanation"])]
        selected = [unit for unit in units if unit["id"] in row["evidence_ids"]]
        all_scores = evaluator.scores(windows(units), hypotheses)
        selected_scores = evaluator.scores(windows(selected), hypotheses) if selected else [0.] * len(hypotheses)
        values.append({"review_id": row["review_id"], "variant": row["variant"], "signature": signature,
            "item_sha256": item_hash(row), "source_support_rate": float(np.mean(np.asarray(all_scores) >= .5)),
            "selected_support_rate": float(np.mean(np.asarray(selected_scores) >= .5)),
            "mean_max_entailment": float(np.mean(all_scores)), "sentences": len(hypotheses)})
        write_jsonl(checkpoint, values)
        print(f"NLI diagnostics {len(values)}/{len(queue)}", flush=True)
    grouped = defaultdict(list)
    for row in values:
        grouped[row["variant"]].append(row)
    runs = [{"name": name + " · NLI diagnostic", "task": 3, "primary_metric": "nli_source_support_rate",
        "metrics": {"count": len(group), "nli_source_support_rate": sum(row["source_support_rate"] for row in group) / len(group),
            "selected_evidence_support_rate": sum(row["selected_support_rate"] for row in group) / len(group),
            "mean_max_entailment": sum(row["mean_max_entailment"] for row in group) / len(group)},
        "scoring_provenance": "Pinned DeBERTa NLI diagnostic; mean fraction of explanation sentences with max-window entailment >=0.5; not factuality or human validation."}
        for name, group in sorted(grouped.items())]
    audit = None
    if Path(args.llm_ratings).exists():
        audit = llm_summary(queue, args.llm_ratings)
        runs.extend(audit["runs"])
    write_json(output / "summary.json", {**meta, "dataset": "CARDS blinded grounding sample", "label_schema": "native",
        "experiment_id": "independent-model-grounding", "runs": runs, "llm_audit": audit,
        "nli_model": MODEL, "nli_revision": REVISION, "nli_weights_sha256": sha256(Path(CACHE) / "model.safetensors"),
        "protocol": {"window_sentences": 3, "stride": 2, "max_tokens": 512, "entailment_index": 1, "threshold": .5},
        "limitations": "Model-based source-support proxies only. Max-window scoring may reward copied text and miss discourse/attribution errors. LLM relevance and attribution ratings are ancillary. No human validation or world-truth judgment is claimed."})


if __name__ == "__main__":
    main()
