"""Validate real reviewer submissions and summarize blinded grounding ratings."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .data import read_jsonl, write_jsonl

CRITERIA = ("supported", "attribution", "relevance", "readability")


def item_hash(row):
    """Bind a judgment to the exact source, target and generated explanation."""
    fields = {key: row.get(key) for key in ("review_id", "text", "target_description", "explanation")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_review(payload, queue_path, ratings_path):
    queue = {row["review_id"]: row for row in read_jsonl(queue_path)}
    if payload.get("review_id") not in queue:
        raise ValueError("Unknown review item")
    if not isinstance(payload.get("reviewer"), str):
        raise ValueError("Enter a reviewer alias")
    reviewer = payload["reviewer"].strip()
    if not reviewer or len(reviewer) > 80:
        raise ValueError("Enter a reviewer alias (1–80 characters)")
    if payload.get("human_reviewer") is not True or payload.get("language_confident") is not True:
        raise ValueError("Confirm that you are a human reviewer able to assess this language")
    for key in CRITERIA:
        if type(payload.get(key)) is not int or payload[key] not in (0, 1, 2):
            raise ValueError("Every criterion requires an explicit score: 0, 1 or 2")
    record = {key: payload[key] for key in CRITERIA}
    record.update(review_id=payload["review_id"], reviewer=reviewer, human_reviewer=True,
        item_sha256=item_hash(queue[payload["review_id"]]),
        language_confident=True, comments=str(payload.get("comments", ""))[:2000],
        timestamp=datetime.now(timezone.utc).isoformat())
    path = Path(ratings_path)
    previous = read_jsonl(path) if path.exists() else []
    previous = [row for row in previous if (row["review_id"], row["reviewer"]) != (record["review_id"], reviewer)]
    write_jsonl(path, previous + [record])
    return {"saved": True, "reviewed_items": len({row["review_id"] for row in previous + [record]}), "total_items": len(queue)}


def summarize_reviews(queue, ratings):
    lookup = {row["review_id"]: row for row in queue}
    if len(lookup) != len(queue):
        raise ValueError("Review queue IDs must be unique")
    by_item = defaultdict(list)
    seen = set()
    for rating in ratings:
        pair = (rating["review_id"], rating["reviewer"])
        if pair in seen or pair[0] not in lookup or not rating.get("human_reviewer") or not rating.get("language_confident"):
            raise ValueError("Invalid or duplicate reviewer record")
        if rating.get("item_sha256") != item_hash(lookup[pair[0]]):
            raise ValueError("Review belongs to different source text or explanation")
        if any(type(rating.get(key)) is not int or rating[key] not in (0, 1, 2) for key in CRITERIA):
            raise ValueError("Invalid review score")
        seen.add(pair)
        by_item[pair[0]].append(rating)
    by_variant = defaultdict(list)
    for key, values in by_item.items():
        by_variant[lookup[key]["variant"]].append(values)
    runs = []
    for variant, items in sorted(by_variant.items()):
        grounding = sum(sum(row["supported"] == 2 for row in item) / len(item) for item in items) / len(items)
        metrics = {"count": len(items), "human_grounding_rate": grounding,
            "review_judgments": sum(len(item) for item in items),
            "mean_ratings": {criterion: sum(sum(row[criterion] for row in item) / len(item) for item in items) / len(items) for criterion in CRITERIA}}
        runs.append({"name": variant, "task": 3, "primary_metric": "human_grounding_rate", "metrics": metrics,
            "scoring_provenance": "Self-declared human reviewers; fully supported=2 on fixed 0/1/2 rubric; per-item mean across reviewers; small blinded source sample."})
    return {"runs": runs, "reviewed_items": len(by_item), "total_items": len(queue),
        "reviewer_count": len({row["reviewer"] for row in ratings}),
        "status": "complete" if len(by_item) == len(queue) and queue else "awaiting_human_review"}
